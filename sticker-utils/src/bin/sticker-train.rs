use std::fs::File;
use std::hash::Hash;
use std::io::BufReader;
use std::usize;

use clap::Arg;
use failure::{Error, Fallible};
use indicatif::ProgressStyle;
use ordered_float::NotNan;
use stdinout::OrExit;
use sticker::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::encoder::layer::LayerEncoder;
use sticker::encoder::{CategoricalEncoder, SentenceEncoder};
use sticker::tensorflow::{
    ConllxDataSet, DataSet, LearningRateSchedule, PlateauLearningRate, TaggerGraph, TaggerTrainer,
};
use sticker::{Numberer, SentVectorizer};
use sticker_utils::{
    sticker_app, CborRead, CompletedUnit, Config, EncoderType, LabelerType, ReadProgress,
    SaveSchedule, SaveScheduler, TomlRead,
};

static CONFIG: &str = "CONFIG";
static INITIAL_LR: &str = "INITIAL_LR";
static LR_SCALE: &str = "LR_SCALE";
static LR_PATIENCE: &str = "LR_PATIENCE";
static MAX_LEN: &str = "MAX_LEN";
static CONTINUE: &str = "CONTINUE";
static PATIENCE: &str = "PATIENCE";
static TRAIN_DATA: &str = "TRAIN_DATA";
static VALIDATION_DATA: &str = "VALIDATION_DATA";
static LOGDIR: &str = "LOGDIR";

pub struct LrSchedule {
    pub initial_lr: NotNan<f32>,
    pub lr_scale: NotNan<f32>,
    pub lr_patience: usize,
}

pub struct TrainApp {
    config: String,
    lr_schedule: LrSchedule,
    max_len: usize,
    parameters: Option<String>,
    patience: usize,
    save_schedule: SaveSchedule,
    train_data: String,
    validation_data: String,
    logdir: Option<String>,
}

impl TrainApp {
    pub fn lr_schedule(&self) -> PlateauLearningRate {
        PlateauLearningRate::new(
            self.lr_schedule.initial_lr.into_inner(),
            self.lr_schedule.lr_scale.into_inner(),
            self.lr_schedule.lr_patience,
        )
    }
}

impl TrainApp {
    fn new() -> Self {
        let matches = sticker_app("sticker-train")
            .arg(
                Arg::with_name(CONTINUE)
                    .long("continue")
                    .takes_value(true)
                    .value_name("PARAMS")
                    .help("Continue training from parameter files (e.g.: epoch-50)"),
            )
            .arg(
                Arg::with_name(INITIAL_LR)
                    .long("lr")
                    .value_name("LR")
                    .help("Initial learning rate")
                    .default_value("0.01"),
            )
            .arg(
                Arg::with_name(LR_PATIENCE)
                    .long("lr-patience")
                    .value_name("N")
                    .help("Scale learning rate after N epochs without improvement")
                    .default_value("4"),
            )
            .arg(
                Arg::with_name(LR_SCALE)
                    .long("lr-scale")
                    .value_name("SCALE")
                    .help("Value to scale the learning rate by")
                    .default_value("0.5"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(PATIENCE)
                    .long("patience")
                    .value_name("N")
                    .help("Maximum number of epochs without improvement")
                    .default_value("15"),
            )
            .arg(
                Arg::with_name(TRAIN_DATA)
                    .help("Training data")
                    .index(2)
                    .required(true),
            )
            .arg(
                Arg::with_name(VALIDATION_DATA)
                    .help("Validation data")
                    .index(3)
                    .required(true),
            )
            .arg(
                Arg::with_name(LOGDIR)
                    .long("logdir")
                    .value_name("LOGDIR")
                    .takes_value(true)
                    .help("Write Tensorboard summaries to this directory."),
            )
            .get_matches();

        let config = matches.value_of(CONFIG).unwrap().into();
        let initial_lr = matches
            .value_of(INITIAL_LR)
            .unwrap()
            .parse()
            .or_exit("Cannot parse initial learning rate", 1);
        let lr_patience = matches
            .value_of(LR_PATIENCE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse learning rate patience", 1);
        let lr_scale = matches
            .value_of(LR_SCALE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse learning rate scale", 1);
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().or_exit("Cannot parse maximum sentence length", 1))
            .unwrap_or(usize::MAX);
        let parameters = matches.value_of(CONTINUE).map(ToOwned::to_owned);
        let patience = matches
            .value_of(PATIENCE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse patience", 1);
        let save_schedule = SaveSchedule::Epoch;
        let logdir = matches.value_of(LOGDIR).map(ToOwned::to_owned);
        let train_data = matches.value_of(TRAIN_DATA).unwrap().into();
        let validation_data = matches.value_of(VALIDATION_DATA).unwrap().into();

        TrainApp {
            config,
            parameters,
            patience,
            lr_schedule: LrSchedule {
                initial_lr,
                lr_patience,
                lr_scale,
            },
            max_len,
            save_schedule,
            train_data,
            validation_data,
            logdir,
        }
    }
}

impl Default for TrainApp {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    let app = TrainApp::new();

    let config_file = File::open(&app.config).or_exit(
        format!("Cannot open configuration file '{}'", app.config),
        1,
    );
    let mut config = Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
    config
        .relativize_paths(&app.config)
        .or_exit("Cannot relativize paths in configuration", 1);

    let train_file = File::open(&app.train_data).or_exit("Cannot open train file for reading", 1);
    let validation_file =
        File::open(&app.validation_data).or_exit("Cannot open validation file for reading", 1);

    let trainer = create_trainer(&config, &app).or_exit("Cannot construct trainer", 1);

    match config.labeler.labeler_type {
        LabelerType::Sequence(ref layer) => train_model_with_encoder::<LayerEncoder>(
            &config,
            &app,
            trainer,
            LayerEncoder::new(layer.clone()),
            train_file,
            validation_file,
        ),
        LabelerType::Parser(EncoderType::RelativePOS) => {
            train_model_with_encoder::<RelativePOSEncoder>(
                &config,
                &app,
                trainer,
                RelativePOSEncoder,
                train_file,
                validation_file,
            )
        }
        LabelerType::Parser(EncoderType::RelativePosition) => {
            train_model_with_encoder::<RelativePositionEncoder>(
                &config,
                &app,
                trainer,
                RelativePositionEncoder,
                train_file,
                validation_file,
            )
        }
    }
    .or_exit("Error while training model", 1);
}

fn create_trainer(config: &Config, app: &TrainApp) -> Fallible<TaggerTrainer> {
    let graph_read = BufReader::new(File::open(&config.model.graph)?);
    let graph = TaggerGraph::load_graph(graph_read, &config.model)?;

    let mut trainer = match app.parameters {
        Some(ref parameters) => TaggerTrainer::load_weights(graph, parameters),
        None => TaggerTrainer::random_weights(graph),
    }?;
    match &app.logdir {
        Some(logdir) => {
            trainer.init_logdir(logdir)?;
        }
        None => {}
    };
    Ok(trainer)
}

fn train_model_with_encoder<E>(
    config: &Config,
    app: &TrainApp,
    trainer: TaggerTrainer,
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

    let mut best_epoch = 0;
    let mut best_acc = 0.0;
    let mut last_acc = 0.0;

    let mut lr_schedule = app.lr_schedule();
    let mut save_scheduler = app.save_schedule.to_save_scheduler("");

    for epoch in 0.. {
        let lr = lr_schedule.learning_rate(epoch, last_acc);

        let (loss, acc) = run_epoch(
            config,
            app,
            &mut save_scheduler,
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

        save_scheduler.save(&trainer, CompletedUnit::Epoch).or_exit(
            format!("Cannot save model for epoch {}", save_scheduler.epoch()),
            1,
        );

        let (loss, acc) = run_epoch(
            config,
            app,
            &mut save_scheduler,
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

        if epoch - best_epoch == app.patience {
            eprintln!(
                "Lost my patience! Best epoch: {} with accuracy: {:.4}",
                best_epoch, best_acc
            );
            break;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_epoch<E>(
    config: &Config,
    app: &TrainApp,
    save_scheduler: &mut SaveScheduler,
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
        .batches(encoder, vectorizer, config.model.batch_size, app.max_len)
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

        if is_training {
            save_scheduler.save(trainer, CompletedUnit::Batch).or_exit(
                format!("Cannot save model for batch {}", save_scheduler.batch()),
                1,
            );
        }

        progress_bar.set_message(&format!(
            "batch loss: {:.4}, batch accuracy: {:.4}",
            batch_perf.loss, batch_perf.accuracy
        ));
    }

    loss /= instances as f32;
    acc /= instances as f32;

    (loss, acc)
}
