extern crate conllx;

extern crate failure;

extern crate getopts;

extern crate stdinout;

extern crate sticker;

extern crate sticker_utils;

use std::env::args;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process;

use conllx::{ReadSentence, Sentence, WriteSentence};
use failure::Error;
use getopts::Options;
use stdinout::{Input, OrExit, Output};

use sticker::tensorflow::Tagger;
use sticker::{Numberer, SentVectorizer, Tag};
use sticker_utils::{CborRead, Config, TomlRead};

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

    if matches.free.len() == 0 || matches.free.len() > 3 {
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
    let reader = conllx::Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(matches.free.get(2));
    let writer = conllx::Writer::new(BufWriter::new(
        output.write().or_exit("Cannot open output for writing", 1),
    ));

    let labels = load_labels(&config).or_exit(
        format!("Cannot load label file '{}'", config.labeler.labels),
        1,
    );

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
    let tagger = Tagger::load_graph_with_weights(
        graph_reader,
        &config.model.parameters,
        vectorizer,
        labels,
        &config.model,
    )
    .or_exit("Cannot load computation graph", 1);

    let mut sent_proc = SentProcessor::new(tagger, writer, config.model.batch_size);

    for sentence in reader.sentences() {
        let sentence = sentence.or_exit("Cannot parse sentence", 1);
        sent_proc
            .process(sentence)
            .or_exit("Error processing sentence", 1);
    }
}

// Wrap the sentence processing in a data type. This has the benefit that
// we can use a destructor to write the last (possibly incomplete) batch.
struct SentProcessor<T, W>
where
    T: Tag,
    W: Write,
{
    tagger: T,
    writer: conllx::Writer<W>,
    batch_size: usize,
    batch_sents: Vec<Sentence>,
}

impl<T, W> SentProcessor<T, W>
where
    T: Tag,
    W: Write,
{
    pub fn new(tagger: T, writer: conllx::Writer<W>, batch_size: usize) -> Self {
        SentProcessor {
            tagger,
            writer,
            batch_size,
            batch_sents: Vec::new(),
        }
    }

    pub fn process(&mut self, sent: Sentence) -> Result<(), Error> {
        self.batch_sents.push(sent);

        if self.batch_sents.len() == self.batch_size {
            let labels = labels_to_owned(self.tagger.tag_sentences(&self.batch_sents)?);
            self.write_sent_labels(labels)?;
            self.batch_sents.clear();
        }

        Ok(())
    }

    fn write_sent_labels(&mut self, labels: Vec<Vec<String>>) -> Result<(), Error>
    where
        W: Write,
    {
        for (tokens, sent_labels) in self.batch_sents.iter_mut().zip(labels) {
            {
                for (token, label) in tokens.iter_mut().zip(sent_labels) {
                    token.set_pos(Some(label));
                }
            }

            self.writer.write_sentence(tokens)?;
        }

        Ok(())
    }
}

impl<T, W> Drop for SentProcessor<T, W>
where
    T: Tag,
    W: Write,
{
    fn drop(&mut self) {
        if !self.batch_sents.is_empty() {
            let labels = labels_to_owned(match self.tagger.tag_sentences(&self.batch_sents) {
                Ok(labels) => labels,
                Err(err) => {
                    eprintln!("Error tagging sentences: {}", err);
                    return;
                }
            });

            if let Err(err) = self.write_sent_labels(labels) {
                eprintln!("Error writing sentences: {}", err);
            }
        }
    }
}

fn load_labels(config: &Config) -> Result<Numberer<String>, Error> {
    let labels_path = Path::new(&config.labeler.labels);

    eprintln!("Loading labels from: {:?}", labels_path);

    let f = File::open(labels_path)?;
    Numberer::from_cbor_read(f)
}

fn labels_to_owned(labels: Vec<Vec<&str>>) -> Vec<Vec<String>> {
    labels
        .into_iter()
        .map(|sv| sv.into_iter().map(str::to_owned).collect())
        .collect()
}
