use std::env::args;
use std::fs::File;
use std::path::Path;
use std::process;

use conllx::ReadSentence;
use failure::Error;
use getopts::Options;
use serde_derive::Serialize;
use stdinout::{Input, OrExit, Output};

use sticker::{Collector, NoopCollector, Numberer, SentVectorizer};
use sticker_utils::{CborWrite, Config, TomlRead};

/// Ad-hoc shapes structure, which can be used to construct the
/// Tensorflow parsing graph.
#[derive(Serialize)]
struct Shapes {
    n_labels: usize,
    token_embed_dims: usize,
}

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] CONFIG DATA SHAPES", program);
    print!("{}", opts.usage(&brief));
    process::exit(1);
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optflag("h", "help", "print this help menu");
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse options", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.is_empty() || matches.free.len() > 3 {
        print_usage(&program, opts);
        std::process::exit(1);
    }

    let config_file = File::open(&matches.free[0]).or_exit(
        format!("Cannot open configuration file '{}'", &matches.free[0]),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(&matches.free[0])
        .or_exit("Cannot relativize paths in configuration", 1);

    let input = Input::from(matches.free.get(1));
    let treebank_reader = conllx::Reader::new(
        input
            .buf_read()
            .or_exit("Cannot open corpus for reading", 1),
    );
    let output = Output::from(matches.free.get(2));
    let mut shapes_write = output.write().or_exit("Cannot create shapes file", 1);

    let labels = Numberer::new(1);

    let embeddings = config
        .embeddings
        .load_embeddings()
        .or_exit("Cannot load embeddings", 1);
    let vectorizer = SentVectorizer::new(embeddings);

    let mut collector = NoopCollector::new(config.labeler.layer.clone(), labels, vectorizer);

    for sentence in treebank_reader.sentences() {
        let sentence = sentence.or_exit("Cannot parse sentence", 1);
        collector
            .collect(&sentence)
            .or_exit("Cannot collect sentence", 1);
    }

    write_labels(&config, collector.labels()).or_exit("Cannot write labels", 1);

    let shapes = Shapes {
        n_labels: collector.labels().len(),
        token_embed_dims: collector
            .vectorizer()
            .layer_embeddings()
            .token_embeddings()
            .dims(),
    };

    write!(
        shapes_write,
        "{}",
        toml::to_string(&shapes).or_exit("Cannot write to shapes file", 1)
    )
    .or_exit("Cannot write shapes", 1);
}

fn write_labels(config: &Config, labels: &Numberer<String>) -> Result<(), Error> {
    let labels_path = Path::new(&config.labeler.labels);
    let mut f = File::create(labels_path)?;
    labels.to_cbor_write(&mut f)
}
