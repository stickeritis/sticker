use std::env::args;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process;
use std::sync::Arc;

use conllx::io::{ReadSentence, Reader, Writer};
use failure::Error;
use getopts::Options;
use stdinout::OrExit;
use threadpool::ThreadPool;

use sticker::tensorflow::{Tagger, TaggerGraph};
use sticker::{LayerEncoder, Numberer, SentVectorizer};
use sticker_utils::{CborRead, Config, SentProcessor, TomlRead};

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] CONFIG [INPUT] [OUTPUT]", program);
    print!("{}", opts.usage(&brief));
    process::exit(1);
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    opts.optopt(
        "t",
        "threads",
        "default server threadpool size (default: 4)",
        "N",
    );
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse options", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.len() != 2 {
        print_usage(&program, opts);
    }

    let n_threads = matches
        .opt_str("t")
        .as_ref()
        .map(|t| {
            t.parse()
                .or_exit(format!("Invalid number of threads: {}", t), 1)
        })
        .unwrap_or(4);

    let config_file = File::open(&matches.free[0]).or_exit(
        format!("Cannot open configuration file '{}'", &matches.free[0]),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(&matches.free[0])
        .or_exit("Cannot relativize paths in configuration", 1);

    let labels = load_labels(&config).or_exit(
        format!("Cannot load label file '{}'", config.labeler.labels),
        1,
    );

    // Parallel processing is useless without the same parallelism in Tensorflow.
    config.model.inter_op_parallelism_threads = n_threads;
    config.model.intra_op_parallelism_threads = n_threads;

    let embeddings = config
        .embeddings
        .load_embeddings()
        .or_exit("Cannot load embeddings", 1);
    let vectorizer = SentVectorizer::new(embeddings);

    let graph_reader = File::open(&config.model.graph).or_exit(
        format!(
            "Cannot open computation graph '{}' for reading",
            &config.model.graph
        ),
        1,
    );

    let graph = TaggerGraph::load_graph(graph_reader, &config.model)
        .or_exit("Cannot load computation graph", 1);
    let tagger = Arc::new(
        Tagger::load_weights(graph, labels, vectorizer, config.model.parameters.clone())
            .or_exit("Cannot construct tagger", 1),
    );

    let addr = &matches.free[1];
    let pool = ThreadPool::new(n_threads);

    let listener = TcpListener::bind(addr).or_exit(format!("Cannot listen on '{}'", addr), 1);

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let config = config.clone();
                let tagger = tagger.clone();
                pool.execute(move || handle_client(config, tagger, stream))
            }
            Err(err) => eprintln!("Error processing stream: {}", err),
        }
    }
}

fn handle_client(config: Config, tagger: Arc<Tagger<String>>, mut stream: TcpStream) {
    let peer_addr = stream
        .peer_addr()
        .map(|addr| addr.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    eprintln!("Accepted connection from {}", peer_addr);

    let conllx_stream = match stream.try_clone() {
        Ok(stream) => stream,
        Err(err) => {
            eprintln!("Cannot clone stream: {}", err);
            return;
        }
    };

    let reader = Reader::new(BufReader::new(&conllx_stream));
    let writer = Writer::new(BufWriter::new(&conllx_stream));

    let decoder = LayerEncoder::new(config.labeler.layer.clone());
    let mut sent_proc = SentProcessor::new(
        decoder,
        &*tagger,
        writer,
        config.model.batch_size,
        config.labeler.read_ahead,
    );

    for sentence in reader.sentences() {
        let sentence = match sentence {
            Ok(sentence) => sentence,
            Err(err) => {
                let _ = writeln!(stream, "! Cannot parse sentence: {}", err);
                return;
            }
        };
        if let Err(err) = sent_proc.process(sentence) {
            let _ = writeln!(stream, "! Error processing sentence: {}", err);
            return;
        }
    }

    eprintln!("Finished processing for {}", peer_addr);
}

fn load_labels(config: &Config) -> Result<Numberer<String>, Error> {
    let labels_path = Path::new(&config.labeler.labels);

    eprintln!("Loading labels from: {:?}", labels_path);

    let f = File::open(labels_path)?;
    Numberer::from_cbor_read(f)
}
