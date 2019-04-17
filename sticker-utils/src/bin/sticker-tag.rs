use std::env::args;
use std::fs::File;
use std::hash::Hash;
use std::io::BufWriter;
use std::process;

use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use getopts::Options;
use stdinout::{Input, OrExit, Output};

use sticker::depparse::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::tensorflow::{Tagger, TaggerGraph};
use sticker::{LayerEncoder, Numberer, SentVectorizer, SentenceDecoder};
use sticker_utils::{CborRead, Config, EncoderType, LabelerType, SentProcessor, TomlRead};

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
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse options", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.is_empty() || matches.free.len() > 3 {
        print_usage(&program, opts);
        return;
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
    let reader = Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(matches.free.get(2));
    let writer = Writer::new(BufWriter::new(
        output.write().or_exit("Cannot open output for writing", 1),
    ));

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

    match config.labeler.labeler_type {
        LabelerType::Sequence(ref layer) => process_with_decoder(
            &config,
            vectorizer,
            graph,
            LayerEncoder::new(layer.clone()),
            reader,
            writer,
        ),
        LabelerType::Parser(EncoderType::RelativePOS) => process_with_decoder(
            &config,
            vectorizer,
            graph,
            RelativePOSEncoder,
            reader,
            writer,
        ),
        LabelerType::Parser(EncoderType::RelativePosition) => process_with_decoder(
            &config,
            vectorizer,
            graph,
            RelativePositionEncoder,
            reader,
            writer,
        ),
    };
}

fn process_with_decoder<D, R, W>(
    config: &Config,
    vectorizer: SentVectorizer,
    graph: TaggerGraph,
    decoder: D,
    read: R,
    write: W,
) where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
    Numberer<D::Encoding>: CborRead,
    R: ReadSentence,
    W: WriteSentence,
{
    let labels = config.labeler.load_labels().or_exit(
        format!("Cannot load label file '{}'", config.labeler.labels),
        1,
    );

    let tagger = Tagger::load_weights(graph, labels, vectorizer, &config.model.parameters)
        .or_exit("Cannot construct tagger", 1);

    let mut sent_proc = SentProcessor::new(
        decoder,
        &tagger,
        write,
        config.model.batch_size,
        config.labeler.read_ahead,
    );

    for sentence in read.sentences() {
        let sentence = sentence.or_exit("Cannot parse sentence", 1);
        sent_proc
            .process(sentence)
            .or_exit("Error processing sentence", 1);
    }
}
