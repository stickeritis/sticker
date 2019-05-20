use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use clap::Arg;
use conllx::io::{ReadSentence, Reader, Writer};
use stdinout::OrExit;
use threadpool::ThreadPool;

use sticker::depparse::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::tensorflow::{Tagger, TaggerGraph};
use sticker::{CategoricalEncoder, LayerEncoder, Numberer, SentVectorizer, SentenceDecoder};
use sticker_utils::{
    sticker_app, CborRead, Config, EncoderType, LabelerType, SentProcessor, TomlRead,
};

static CONFIG: &str = "CONFIG";
static ADDR: &str = "ADDR";
static THREADS: &str = "THREADS";

pub struct ServerApp {
    config: String,
    addr: String,
    n_threads: usize,
}

impl ServerApp {
    fn new() -> Self {
        let matches = sticker_app("sticker-server")
            .arg(
                Arg::with_name(THREADS)
                    .short("t")
                    .long("threads")
                    .value_name("N")
                    .help("Number of threads")
                    .default_value("4"),
            )
            .arg(
                Arg::with_name(ADDR)
                    .help("Address to bind to (e.g. localhost:4000)")
                    .index(2)
                    .required(true),
            )
            .get_matches();

        let config = matches.value_of(CONFIG).unwrap().into();
        let addr = matches.value_of(ADDR).unwrap().into();
        let n_threads = matches
            .value_of(THREADS)
            .map(|v| v.parse().or_exit("Cannot parse number of threads", 1))
            .unwrap();

        ServerApp {
            config,
            addr,
            n_threads,
        }
    }
}

impl Default for ServerApp {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    let app = ServerApp::new();

    let config_file = File::open(&app.config).or_exit(
        format!("Cannot open configuration file '{}'", app.config),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(app.config)
        .or_exit("Cannot relativize paths in configuration", 1);

    // Parallel processing is useless without the same parallelism in Tensorflow.
    config.model.inter_op_parallelism_threads = app.n_threads;
    config.model.intra_op_parallelism_threads = app.n_threads;

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

    let pool = ThreadPool::new(app.n_threads);

    let listener =
        TcpListener::bind(&app.addr).or_exit(format!("Cannot listen on '{}'", app.addr), 1);

    match config.labeler.labeler_type {
        LabelerType::Sequence(ref layer) => serve_with_decoder(
            &config,
            pool,
            vectorizer,
            graph,
            LayerEncoder::new(layer.clone()),
            listener,
        ),
        LabelerType::Parser(EncoderType::RelativePOS) => serve_with_decoder(
            &config,
            pool,
            vectorizer,
            graph,
            RelativePOSEncoder,
            listener,
        ),
        LabelerType::Parser(EncoderType::RelativePosition) => serve_with_decoder(
            &config,
            pool,
            vectorizer,
            graph,
            RelativePositionEncoder,
            listener,
        ),
    }
}

fn serve_with_decoder<D>(
    config: &Config,
    pool: ThreadPool,
    vectorizer: SentVectorizer,
    graph: TaggerGraph,
    decoder: D,
    listener: TcpListener,
) where
    D: 'static + Clone + Send + SentenceDecoder + Sync,
    D::Encoding: Clone + Eq + Hash + Send + Sync,
    Numberer<D::Encoding>: CborRead,
{
    let labels = config.labeler.load_labels().or_exit(
        format!("Cannot load label file '{}'", config.labeler.labels),
        1,
    );

    let categorical_decoder = CategoricalEncoder::new(decoder, labels);

    let tagger = Arc::new(
        Tagger::load_weights(
            graph,
            categorical_decoder,
            vectorizer,
            config.model.parameters.clone(),
        )
        .or_exit("Cannot construct tagger", 1),
    );

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let config = config.clone();
                let tagger = tagger.clone();
                pool.execute(move || handle_client_with_decoder(config, tagger, stream))
            }
            Err(err) => eprintln!("Error processing stream: {}", err),
        }
    }
}

fn handle_client_with_decoder<D>(config: Config, tagger: Arc<Tagger<D>>, mut stream: TcpStream)
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
{
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

    let mut sent_proc = SentProcessor::new(
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
