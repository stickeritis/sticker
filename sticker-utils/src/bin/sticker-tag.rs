use std::io::BufWriter;

use clap::Arg;
use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, OrExit, Output};

use sticker_utils::app;
use sticker_utils::{Pipeline, SentProcessor, TaggerSpeed};

static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct TagApp {
    batch_size: usize,
    configs: Vec<String>,
    input: Option<String>,
    output: Option<String>,
    read_ahead: usize,
}

impl TagApp {
    fn new() -> Self {
        let matches = app::sticker_pipeline_app("sticker-tag")
            .arg(
                Arg::with_name(INPUT)
                    .help("Input data")
                    .long("input")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(OUTPUT)
                    .help("Output data")
                    .long("output")
                    .takes_value(true),
            )
            .get_matches();

        let batch_size = matches
            .value_of(app::BATCH_SIZE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse batch size", 1);
        let configs = matches
            .values_of(app::CONFIGS)
            .unwrap()
            .map(ToOwned::to_owned)
            .collect();
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);
        let read_ahead = matches
            .value_of(app::READ_AHEAD)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of batches to read ahead", 1);

        TagApp {
            batch_size,
            configs,
            input,
            output,
            read_ahead,
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

    let pipeline = Pipeline::new_from_config_filenames(&app.configs)
        .or_exit("Cannot construct tagging pipeline", 1);

    let input = Input::from(app.input.clone());
    let reader = Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(app.output.clone());
    let writer = Writer::new(BufWriter::new(
        output.write().or_exit("Cannot open output for writing", 1),
    ));

    process(&app, pipeline, reader, writer);
}

fn process<R, W>(app: &TagApp, pipeline: Pipeline, read: R, write: W)
where
    R: ReadSentence,
    W: WriteSentence,
{
    let mut speed = TaggerSpeed::new();

    let mut sent_proc = SentProcessor::new(&pipeline, write, app.batch_size, app.read_ahead);

    for sentence in read.sentences() {
        let sentence = sentence.or_exit("Cannot parse sentence", 1);
        sent_proc
            .process(sentence)
            .or_exit("Error processing sentence", 1);

        speed.count_sentence()
    }
}
