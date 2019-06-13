use std::fs::File;
use std::hash::Hash;
use std::io::BufWriter;

use clap::Arg;
use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, OrExit, Output};

use sticker::depparse::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::tensorflow::{Tagger, TaggerGraph};
use sticker::{CategoricalEncoder, LayerEncoder, Numberer, SentVectorizer, SentenceDecoder};
use sticker_utils::{
    sticker_app, CborRead, Config, EncoderType, LabelerType, SentProcessor, TaggerSpeed, TomlRead,
};

static CONFIG: &str = "CONFIG";
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct TagApp {
    config: String,
    input: Option<String>,
    output: Option<String>,
}

impl TagApp {
    fn new() -> Self {
        let matches = sticker_app("sticker-tag")
            .arg(Arg::with_name(INPUT).help("Input data").index(2))
            .arg(Arg::with_name(OUTPUT).help("Output data").index(3))
            .get_matches();

        let config = matches.value_of(CONFIG).unwrap().into();
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);

        TagApp {
            config,
            input,
            output,
        }
    }
}

impl Default for TagApp {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    let app = TagApp::new();

    let config_file = File::open(&app.config).or_exit(
        format!("Cannot open configuration file '{}'", app.config),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(app.config)
        .or_exit("Cannot relativize paths in configuration", 1);

    let input = Input::from(app.input);
    let reader = Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(app.output);
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

    let categorical_decoder = CategoricalEncoder::new(decoder, labels);

    let tagger = Tagger::load_weights(
        graph,
        categorical_decoder,
        vectorizer,
        &config.model.parameters,
    )
    .or_exit("Cannot construct tagger", 1);

    let mut speed = TaggerSpeed::new();

    let mut sent_proc = SentProcessor::new(
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

        speed.count_sentence()
    }
}
