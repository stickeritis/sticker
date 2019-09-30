use std::io::BufWriter;
use std::iter::FromIterator;
use std::process;

use clap::{App, AppSettings, Arg};
use conllx::graph::Node;
use conllx::io::{ReadSentence, WriteSentence};
use conllx::token::Features;
use stdinout::{Input, OrExit, Output};

use sticker::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::encoder::SentenceEncoder;

static ENCODER: &str = "ENCODER";
static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub struct Dep2LabelApp {
    encoder: String,
    input: Option<String>,
    output: Option<String>,
}

impl Dep2LabelApp {
    fn new() -> Self {
        let matches = App::new("sticker-dep2label")
            .settings(DEFAULT_CLAP_SETTINGS)
            .arg(
                Arg::with_name(ENCODER)
                    .short("e")
                    .long("encoder")
                    .value_name("ENC")
                    .help("Dependency encoder")
                    .possible_values(&["rel_pos", "rel_position"])
                    .default_value("rel_pos"),
            )
            .arg(Arg::with_name(INPUT).help("Input data").index(1))
            .arg(Arg::with_name(OUTPUT).help("Output data").index(2))
            .get_matches();

        let encoder = matches.value_of(ENCODER).unwrap().into();
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);

        Dep2LabelApp {
            encoder,
            input,
            output,
        }
    }
}

impl Default for Dep2LabelApp {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    let app = Dep2LabelApp::new();
    let input = Input::from(app.input);
    let reader =
        conllx::io::Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(app.output);
    let writer = conllx::io::Writer::new(BufWriter::new(
        output.write().or_exit("Cannot open output for writing", 1),
    ));

    match app.encoder.as_str() {
        "rel_pos" => label_with_encoder(RelativePOSEncoder, reader, writer),
        "rel_position" => label_with_encoder(RelativePositionEncoder, reader, writer),
        unknown => {
            eprintln!("Unknown encoder: {}", unknown);
            process::exit(1);
        }
    }
}

fn label_with_encoder<E, R, W>(mut encoder: E, read: R, mut write: W)
where
    E: SentenceEncoder,
    E::Encoding: ToString,
    R: ReadSentence,
    W: WriteSentence,
{
    for sentence in read.sentences() {
        let mut sentence = sentence.or_exit("Cannot parse sentence", 1);

        let encoded = encoder
            .encode(&sentence)
            .or_exit("Cannot dependency-encode sentence", 1);

        for (token, encoding) in sentence.iter_mut().filter_map(Node::token_mut).zip(encoded) {
            let mut features = token.features().cloned().unwrap_or_default();
            features.insert("deplabel".to_owned(), Some(encoding.to_string()));
            token.set_features(Some(Features::from_iter(features.into_inner())));
        }

        write
            .write_sentence(&sentence)
            .or_exit("Cannot write sentence", 1);
    }
}
