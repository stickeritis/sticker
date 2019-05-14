use std::env::args;
use std::fs::File;
use std::hash::Hash;
use std::io::BufReader;
use std::path::Path;
use std::process;

use conllx::io::{ReadSentence, Reader};
use failure::Error;
use getopts::Options;
use indicatif::{ProgressBar, ProgressStyle};
use stdinout::OrExit;
use sticker::depparse::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::tensorflow::{
    CollectedTensors, LearningRateSchedule, Tagger, TaggerGraph, TensorCollector,
};
use sticker::{Collector, LayerEncoder, Numberer, SentVectorizer, SentenceEncoder};
use sticker_utils::{CborRead, Config, EncoderType, FileProgress, LabelerType, TomlRead};

fn print_usage(program: &str, opts: Options) {
    let brief = format!(
        "Usage: {} [options] CONFIG TRAINING_DATA VALIDATION_DATA",
        program
    );
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

    if matches.free.len() != 3 {
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

    eprintln!("Vectorizing training instances...");
    let train_tensors = collect_tensors(&config, &matches.free[1]);
    eprintln!("Vectorizing validation instances...");
    let validation_tensors = collect_tensors(&config, &matches.free[2]);

    match config.labeler.labeler_type {
        LabelerType::Sequence(_) => {
            train_model_with_encoder::<LayerEncoder>(&config, train_tensors, validation_tensors)
        }
        LabelerType::Parser(EncoderType::RelativePOS) => {
            train_model_with_encoder::<RelativePOSEncoder>(
                &config,
                train_tensors,
                validation_tensors,
            )
        }
        LabelerType::Parser(EncoderType::RelativePosition) => {
            train_model_with_encoder::<RelativePositionEncoder>(
                &config,
                train_tensors,
                validation_tensors,
            )
        }
    }
    .or_exit("Error while training model", 1);
}

fn train_model_with_encoder<E>(
    config: &Config,
    train_tensors: CollectedTensors,
    validation_tensors: CollectedTensors,
) -> Result<(), Error>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    Numberer<E::Encoding>: CborRead,
{
    let labels: Numberer<E::Encoding> = config.labeler.load_labels().or_exit(
        format!(
            "Cannot load or create label file '{}'",
            config.labeler.labels
        ),
        1,
    );

    let embeddings = config
        .embeddings
        .load_embeddings()
        .or_exit("Cannot load embeddings", 1);
    let vectorizer = SentVectorizer::new(embeddings);

    let graph_read = BufReader::new(File::open(&config.model.graph)?);
    let graph = TaggerGraph::load_graph(graph_read, &config.model)?;
    let tagger =
        Tagger::random_weights(graph, labels, vectorizer).or_exit("Cannot construct tagger", 1);

    let mut best_epoch = 0;
    let mut best_acc = 0.0;
    let mut last_acc = 0.0;

    let mut lr_schedule = config.train.lr_schedule();

    for epoch in 0.. {
        let lr = lr_schedule.learning_rate(epoch, last_acc);

        let (loss, acc) = run_epoch(&tagger, &train_tensors, true, lr);

        eprintln!(
            "Epoch {} (train, lr: {:.4}): loss: {:.4}, acc: {:.4}",
            epoch, lr, loss, acc
        );

        let (loss, acc) = run_epoch(&tagger, &validation_tensors, false, lr);

        last_acc = acc;
        if acc > best_acc {
            best_epoch = epoch;
            best_acc = acc;
        }

        eprintln!(
            "Epoch {} (validation): loss: {:.4}, acc: {:.4}, best epoch: {}, best acc: {:.4}",
            epoch, loss, acc, best_epoch, best_acc
        );

        tagger
            .save(format!("epoch-{}", epoch))
            .or_exit(format!("Cannot save model for epoch {}", epoch), 1);

        if epoch - best_epoch == config.train.patience {
            eprintln!(
                "Lost my patience! Best epoch: {} with accuracy: {:.4}",
                best_epoch, best_acc
            );
            break;
        }
    }

    Ok(())
}

fn run_epoch<E>(
    tagger: &Tagger<E>,
    tensors: &CollectedTensors,
    is_training: bool,
    lr: f32,
) -> (f32, f32)
where
    E: Clone + Eq + Hash,
{
    let epoch_type = if is_training { "train" } else { "validation" };

    let mut instances = 0;
    let mut acc = 0f32;
    let mut loss = 0f32;

    let progress = ProgressBar::new(tensors.labels.len() as u64);
    progress.set_style(ProgressStyle::default_bar().template(&format!(
        "{{bar}} {} batch {{pos}}/{{len}}, {{msg}}",
        epoch_type
    )));

    for i in 0..tensors.labels.len() {
        let seq_lens = &tensors.sequence_lens[i];
        let tokens = &tensors.inputs[i];
        let labels = &tensors.labels[i];

        let batch_perf = if is_training {
            tagger.train(seq_lens, tokens, labels, lr)
        } else {
            tagger.validate(seq_lens, tokens, labels)
        };

        let n_tokens = seq_lens.view().iter().sum::<i32>();
        loss += n_tokens as f32 * batch_perf.loss;
        acc += n_tokens as f32 * batch_perf.accuracy;
        instances += n_tokens;

        progress.inc(1);
        progress.set_message(&format!("batch loss: {}", batch_perf.loss));
    }

    loss /= instances as f32;
    acc /= instances as f32;

    (loss, acc)
}

fn collect_tensors<P>(config: &Config, path: P) -> CollectedTensors
where
    P: AsRef<Path>,
{
    let embeddings = config
        .embeddings
        .load_embeddings()
        .or_exit("Cannot load embeddings", 1);
    let vectorizer = SentVectorizer::new(embeddings);

    match config.labeler.labeler_type {
        LabelerType::Sequence(ref layer) => {
            collect_tensors_with_encoder(config, vectorizer, LayerEncoder::new(layer.clone()), path)
        }
        LabelerType::Parser(EncoderType::RelativePOS) => {
            collect_tensors_with_encoder(config, vectorizer, RelativePOSEncoder, path)
        }
        LabelerType::Parser(EncoderType::RelativePosition) => {
            collect_tensors_with_encoder(config, vectorizer, RelativePositionEncoder, path)
        }
    }
}

fn collect_tensors_with_encoder<E, P>(
    config: &Config,
    vectorizer: SentVectorizer,
    encoder: E,
    path: P,
) -> CollectedTensors
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    Numberer<E::Encoding>: CborRead,
    P: AsRef<Path>,
{
    let labels = config.labeler.load_labels().or_exit(
        format!(
            "Cannot load or create label file '{}'",
            config.labeler.labels
        ),
        1,
    );

    let input_file = File::open(path.as_ref()).or_exit(
        format!(
            "Cannot open '{}' for reading",
            path.as_ref().to_string_lossy()
        ),
        1,
    );
    let reader = Reader::new(BufReader::new(
        FileProgress::new(input_file).or_exit("Cannot create file progress bar", 1),
    ));

    let mut collector = TensorCollector::new(config.model.batch_size, encoder, labels, vectorizer);
    for sentence in reader.sentences() {
        let sentence = sentence.or_exit("Cannot parse sentence", 1);
        collector
            .collect(&sentence)
            .or_exit("Cannot collect sentence", 1);
    }

    collector.into_parts()
}
