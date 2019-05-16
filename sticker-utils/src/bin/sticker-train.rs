use std::env::args;
use std::fs::File;
use std::hash::Hash;
use std::io::BufReader;
use std::process;

use failure::Error;
use getopts::Options;
use indicatif::ProgressStyle;
use stdinout::OrExit;
use sticker::depparse::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::tensorflow::{
    ConllxDataSet, DataSet, LearningRateSchedule, TaggerGraph, TaggerTrainer,
};
use sticker::{CategoricalEncoder, LayerEncoder, Numberer, SentVectorizer, SentenceEncoder};
use sticker_utils::{CborRead, Config, EncoderType, LabelerType, ReadProgress, TomlRead};

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

    let train_file = File::open(&matches.free[1]).or_exit("Cannot open train file for reading", 1);
    let validation_file =
        File::open(&matches.free[2]).or_exit("Cannot open validation file for reading", 1);

    match config.labeler.labeler_type {
        LabelerType::Sequence(ref layer) => train_model_with_encoder::<LayerEncoder>(
            &config,
            LayerEncoder::new(layer.clone()),
            train_file,
            validation_file,
        ),
        LabelerType::Parser(EncoderType::RelativePOS) => {
            train_model_with_encoder::<RelativePOSEncoder>(
                &config,
                RelativePOSEncoder,
                train_file,
                validation_file,
            )
        }
        LabelerType::Parser(EncoderType::RelativePosition) => {
            train_model_with_encoder::<RelativePositionEncoder>(
                &config,
                RelativePositionEncoder,
                train_file,
                validation_file,
            )
        }
    }
    .or_exit("Error while training model", 1);
}

fn train_model_with_encoder<E>(
    config: &Config,
    encoder: E,
    mut train_file: File,
    mut validation_file: File,
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

    let mut categorical_encoder = CategoricalEncoder::new(encoder, labels);

    let embeddings = config
        .embeddings
        .load_embeddings()
        .or_exit("Cannot load embeddings", 1);
    let vectorizer = SentVectorizer::new(embeddings);

    let graph_read = BufReader::new(File::open(&config.model.graph)?);
    let graph = TaggerGraph::load_graph(graph_read, &config.model)?;
    let trainer = TaggerTrainer::random_weights(graph).or_exit("Cannot construct trainer", 1);

    let mut best_epoch = 0;
    let mut best_acc = 0.0;
    let mut last_acc = 0.0;

    let mut lr_schedule = config.train.lr_schedule();

    for epoch in 0.. {
        let lr = lr_schedule.learning_rate(epoch, last_acc);

        let (loss, acc) = run_epoch(
            config,
            &mut categorical_encoder,
            &vectorizer,
            &trainer,
            &mut train_file,
            true,
            lr,
        );

        eprintln!(
            "Epoch {} (train, lr: {:.4}): loss: {:.4}, acc: {:.4}",
            epoch, lr, loss, acc
        );

        let (loss, acc) = run_epoch(
            config,
            &mut categorical_encoder,
            &vectorizer,
            &trainer,
            &mut validation_file,
            false,
            lr,
        );

        last_acc = acc;
        if acc > best_acc {
            best_epoch = epoch;
            best_acc = acc;
        }

        let epoch_status = if best_epoch == epoch { "🎉" } else { "" };

        eprintln!(
            "Epoch {} (validation): loss: {:.4}, acc: {:.4}, best epoch: {}, best acc: {:.4} {}",
            epoch, loss, acc, best_epoch, best_acc, epoch_status
        );

        trainer
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
    config: &Config,
    encoder: &mut CategoricalEncoder<E, E::Encoding>,
    vectorizer: &SentVectorizer,
    trainer: &TaggerTrainer,
    file: &mut File,
    is_training: bool,
    lr: f32,
) -> (f32, f32)
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    let epoch_type = if is_training { "train" } else { "validation" };

    let read_progress = ReadProgress::new(file).or_exit("Cannot create progress bar", 1);
    let progress_bar = read_progress.progress_bar().clone();
    progress_bar.set_style(ProgressStyle::default_bar().template(&format!(
        "[Time: {{elapsed_precise}}, ETA: {{eta_precise}}] {{bar}} {{percent}}% {} {{msg}}",
        epoch_type
    )));

    let mut dataset = ConllxDataSet::new(read_progress);

    let mut instances = 0;
    let mut acc = 0f32;
    let mut loss = 0f32;

    for batch in dataset
        .batches(encoder, vectorizer, config.model.batch_size)
        .or_exit("Cannot read batches", 1)
    {
        let (inputs, seq_lens, labels) = batch.or_exit("Cannot read batch", 1).into_parts();

        let batch_perf = if is_training {
            trainer.train(&seq_lens, &inputs, &labels, lr)
        } else {
            trainer.validate(&seq_lens, &inputs, &labels)
        };

        let n_tokens = seq_lens.view().iter().sum::<i32>();
        loss += n_tokens as f32 * batch_perf.loss;
        acc += n_tokens as f32 * batch_perf.accuracy;
        instances += n_tokens;

        progress_bar.set_message(&format!(
            "batch loss: {:.4}, batch accuracy: {:.4}",
            batch_perf.loss, batch_perf.accuracy
        ));
    }

    loss /= instances as f32;
    acc /= instances as f32;

    (loss, acc)
}
