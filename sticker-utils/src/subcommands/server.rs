use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use clap::{App, Arg, ArgMatches};
use conllx::io::{ReadSentence, Reader, Writer};
use stdinout::OrExit;
use sticker::wrapper::{Config, Pipeline, TomlRead};
use threadpool::ThreadPool;

use crate::progress::TaggerSpeed;
use crate::sent_proc::SentProcessor;
use crate::traits::{StickerApp, StickerPipelineApp};

static ADDR: &str = "ADDR";
static THREADS: &str = "THREADS";

#[derive(Clone)]
pub struct ServerApp {
    batch_size: usize,
    configs: Vec<String>,
    addr: String,
    n_threads: usize,
    read_ahead: usize,
}

impl ServerApp {
    fn serve(&self, pipeline: Arc<Pipeline>, pool: ThreadPool, listener: TcpListener) {
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let app = self.clone();
                    let pipeline = pipeline.clone();
                    pool.execute(move || handle_client(app, pipeline, stream))
                }
                Err(err) => eprintln!("Error processing stream: {}", err),
            }
        }
    }
}

impl StickerPipelineApp for ServerApp {}

impl StickerApp for ServerApp {
    fn app() -> App<'static, 'static> {
        Self::pipeline_app("server")
            .about("Annotation server")
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
                    .long("addr")
                    .help("Address to bind to (e.g. localhost:4000)")
                    .default_value("localhost:4000"),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        let batch_size = matches
            .value_of(Self::BATCH_SIZE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse batch size", 1);
        let configs = matches
            .values_of(Self::CONFIGS)
            .unwrap()
            .map(ToOwned::to_owned)
            .collect();
        let addr = matches.value_of(ADDR).unwrap().into();
        let n_threads = matches
            .value_of(THREADS)
            .map(|v| v.parse().or_exit("Cannot parse number of threads", 1))
            .unwrap();
        let read_ahead = matches
            .value_of(Self::READ_AHEAD)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of batches to read ahead", 1);

        ServerApp {
            batch_size,
            addr,
            configs,
            n_threads,
            read_ahead,
        }
    }

    fn run(&self) {
        let mut configs = Vec::with_capacity(self.configs.len());
        for filename in &self.configs {
            let config_file = File::open(filename)
                .or_exit(format!("Cannot open configuration file '{}'", filename), 1);
            let mut config =
                Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
            config
                .relativize_paths(filename)
                .or_exit("Cannot relativize paths in configuration", 1);

            // Parallel processing is useless without the same parallelism in Tensorflow.
            config.model.inter_op_parallelism_threads = self.n_threads;
            config.model.intra_op_parallelism_threads = self.n_threads;

            configs.push(config);
        }

        let pipeline =
            Pipeline::new_from_configs(&configs).or_exit("Cannot construct tagging pipeline", 1);

        let pool = ThreadPool::new(self.n_threads);

        let listener =
            TcpListener::bind(&self.addr).or_exit(format!("Cannot listen on '{}'", self.addr), 1);

        self.serve(Arc::new(pipeline), pool, listener);
    }
}

fn handle_client(app: ServerApp, tagger: Arc<Pipeline>, mut stream: TcpStream) {
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

    let mut sent_proc = SentProcessor::new(&*tagger, writer, app.batch_size, app.read_ahead);

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
