use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use clap::Arg;
use conllx::io::{ReadSentence, Reader, Writer};
use stdinout::OrExit;
use threadpool::ThreadPool;

use sticker_utils::{sticker_app, Config, SentProcessor, TaggerSpeed, TaggerWrapper, TomlRead};

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

    let tagger = TaggerWrapper::new(&config).or_exit("Cannot construct tagger", 1);

    let pool = ThreadPool::new(app.n_threads);

    let listener =
        TcpListener::bind(&app.addr).or_exit(format!("Cannot listen on '{}'", app.addr), 1);

    serve(&config, Arc::new(tagger), pool, listener);
}

fn serve(config: &Config, tagger: Arc<TaggerWrapper>, pool: ThreadPool, listener: TcpListener) {
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

fn handle_client(config: Config, tagger: Arc<TaggerWrapper>, mut stream: TcpStream) {
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

    let mut speed = TaggerSpeed::new();

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

        speed.count_sentence()
    }

    eprintln!("Finished processing for {}", peer_addr);
}
