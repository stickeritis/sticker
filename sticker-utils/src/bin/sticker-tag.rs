use std::fs::File;
use std::io::BufWriter;

use clap::Arg;
use conllx::io::{ReadSentence, Reader, WriteSentence, Writer};
use stdinout::{Input, OrExit, Output};

use sticker_utils::{sticker_app, Config, SentProcessor, TaggerSpeed, TaggerWrapper, TomlRead};

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

    let tagger = TaggerWrapper::new(&config).or_exit("Cannot construct tagger", 1);

    process(&config, tagger, reader, writer);
}

fn process<R, W>(config: &Config, tagger: TaggerWrapper, read: R, write: W)
where
    R: ReadSentence,
    W: WriteSentence,
{
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
