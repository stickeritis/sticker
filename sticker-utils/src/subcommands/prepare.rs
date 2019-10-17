use std::fs::File;
use std::hash::Hash;
use std::io::Write;
use std::path::Path;

use clap::{App, Arg, ArgMatches};
use conllx::io::{ReadSentence, Reader};
use failure::Error;
use serde_derive::Serialize;
use stdinout::{Input, OrExit, Output};
use toml;

use sticker::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::encoder::layer::LayerEncoder;
use sticker::encoder::SentenceEncoder;
use sticker::serialization::CborWrite;
use sticker::wrapper::{Config, EncoderType, LabelerType, TomlRead};
use sticker::{Collector, Embeddings, NoopCollector, Numberer, SentVectorizer};

use crate::traits::{StickerApp, StickerTrainApp};

static TRAIN_DATA: &str = "TRAIN_DATA";
static SHAPES: &str = "SHAPES";

/// Ad-hoc shapes structure, which can be used to construct the
/// Tensorflow parsing graph.
#[derive(Serialize)]
struct Shapes {
    n_labels: usize,
    subwords: bool,
    token_embed_dims: usize,
    tag_embed_dims: usize,
}

pub struct PrepareApp {
    config: String,
    train_data: Option<String>,
    shapes: Option<String>,
}

impl StickerTrainApp for PrepareApp {}

impl StickerApp for PrepareApp {
    fn app() -> App<'static, 'static> {
        Self::train_app("prepare")
            .about("Prepare shape and label files for training")
            .arg(Arg::with_name(TRAIN_DATA).help("Training data").index(2))
            .arg(Arg::with_name(SHAPES).help("Shape output file").index(3))
    }

    fn parse(matches: &ArgMatches) -> Self {
        let config = matches.value_of(Self::CONFIG).unwrap().into();
        let train_data = matches.value_of(TRAIN_DATA).map(ToOwned::to_owned);
        let shapes = matches.value_of(SHAPES).map(ToOwned::to_owned);

        PrepareApp {
            config,
            train_data,
            shapes,
        }
    }

    fn run(&self) {
        let config_file = File::open(&self.config).or_exit(
            format!("Cannot open configuration file '{}'", &self.config),
            1,
        );
        let mut config =
            Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
        config
            .relativize_paths(&self.config)
            .or_exit("Cannot relativize paths in configuration", 1);

        let input = Input::from(self.train_data.as_ref());
        let treebank_reader = Reader::new(
            input
                .buf_read()
                .or_exit("Cannot open corpus for reading", 1),
        );
        let output = Output::from(self.shapes.as_ref());
        let shapes_write = output.write().or_exit("Cannot create shapes file", 1);

        let embeddings = config
            .input
            .embeddings
            .load_embeddings()
            .or_exit("Cannot load embeddings", 1);
        let vectorizer = SentVectorizer::new(embeddings, config.input.subwords);

        match config.labeler.labeler_type {
            LabelerType::Sequence(ref layer) => prepare_with_encoder(
                &config,
                vectorizer,
                LayerEncoder::new(layer.clone()),
                treebank_reader,
                shapes_write,
            ),
            LabelerType::Parser(EncoderType::RelativePOS) => prepare_with_encoder(
                &config,
                vectorizer,
                RelativePOSEncoder,
                treebank_reader,
                shapes_write,
            ),
            LabelerType::Parser(EncoderType::RelativePosition) => prepare_with_encoder(
                &config,
                vectorizer,
                RelativePositionEncoder,
                treebank_reader,
                shapes_write,
            ),
        };
    }
}

fn prepare_with_encoder<E, R, W>(
    config: &Config,
    vectorizer: SentVectorizer,
    encoder: E,
    read: R,
    shapes_write: W,
) where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    Numberer<E::Encoding>: CborWrite,
    R: ReadSentence,
    W: Write,
{
    let labels = Numberer::new(1);
    let mut collector = NoopCollector::new(encoder, labels, vectorizer);
    collect_sentences(&mut collector, read);
    write_labels(&config, collector.labels()).or_exit("Cannot write labels", 1);
    write_shapes(&config, shapes_write, &collector);
}

fn collect_sentences<E, R>(collector: &mut NoopCollector<E>, reader: R)
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    R: ReadSentence,
{
    for sentence in reader.sentences() {
        let sentence = sentence.or_exit("Cannot parse sentence", 1);
        collector
            .collect(&sentence)
            .or_exit("Cannot collect sentence", 1);
    }
}

fn write_labels<T>(config: &Config, labels: &Numberer<T>) -> Result<(), Error>
where
    T: Eq + Hash,
    Numberer<T>: CborWrite,
{
    let labels_path = Path::new(&config.labeler.labels);
    let mut f = File::create(labels_path)?;
    labels.to_cbor_write(&mut f)
}

fn write_shapes<W, E>(config: &Config, mut shapes_write: W, collector: &NoopCollector<E>)
where
    W: Write,
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    let shapes = Shapes {
        n_labels: collector.labels().len(),
        subwords: config.input.subwords,
        token_embed_dims: collector
            .vectorizer()
            .layer_embeddings()
            .token_embeddings()
            .dims(),
        tag_embed_dims: collector
            .vectorizer()
            .layer_embeddings()
            .tag_embeddings()
            .map(Embeddings::dims)
            .unwrap_or_default(),
    };

    write!(
        shapes_write,
        "{}",
        toml::to_string(&shapes).or_exit("Cannot write to shapes file", 1)
    )
    .or_exit("Cannot write shapes", 1);
}
