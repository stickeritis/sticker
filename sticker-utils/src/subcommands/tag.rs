use std::io::BufWriter;

use clap::{App, Arg, ArgMatches};
use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, OrExit, Output};
use sticker::wrapper::Pipeline;

use crate::progress::TaggerSpeed;
use crate::sent_proc::SentProcessor;
use crate::traits::{StickerApp, StickerPipelineApp};

static INPUT: &str = "INPUT";
static OUTPUT: &str = "OUTPUT";

pub struct TagApp {
    batch_size: usize,
    configs: Vec<String>,
    input: Option<String>,
    max_len: Option<usize>,
    output: Option<String>,
    read_ahead: usize,
}

impl TagApp {
    fn process<R, W>(&self, pipeline: Pipeline, read: R, write: W)
    where
        R: ReadSentence,
        W: WriteSentence,
    {
        let mut speed = TaggerSpeed::new();

        let mut sent_proc = SentProcessor::new(
            &pipeline,
            write,
            self.batch_size,
            self.max_len,
            self.read_ahead,
        );

        for sentence in read.sentences() {
            let sentence = sentence.or_exit("Cannot parse sentence", 1);
            sent_proc
                .process(sentence)
                .or_exit("Error processing sentence", 1);

            speed.count_sentence()
        }
    }
}

impl StickerPipelineApp for TagApp {}

impl StickerApp for TagApp {
    fn app() -> App<'static, 'static> {
        Self::pipeline_app("tag")
            .about("Annotate sentences with a model")
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
        let input = matches.value_of(INPUT).map(ToOwned::to_owned);
        let max_len = matches
            .value_of(Self::MAX_LEN)
            .map(|v| v.parse().or_exit("Cannot parse maximum sentence length", 1));
        let output = matches.value_of(OUTPUT).map(ToOwned::to_owned);
        let read_ahead = matches
            .value_of(Self::READ_AHEAD)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of batches to read ahead", 1);

        TagApp {
            batch_size,
            configs,
            input,
            max_len,
            output,
            read_ahead,
        }
    }

    fn run(&self) {
        let pipeline = Pipeline::from_config_filenames(&self.configs)
            .or_exit("Cannot construct tagging pipeline", 1);

        let input = Input::from(self.input.as_ref());
        let reader = Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

        let output = Output::from(self.output.as_ref());
        let writer = Writer::new(BufWriter::new(
            output.write().or_exit("Cannot open output for writing", 1),
        ));

        self.process(pipeline, reader, writer);
    }
}
