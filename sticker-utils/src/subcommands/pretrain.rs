use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, Seek, SeekFrom};
use std::usize;

use clap::{App, Arg, ArgMatches};
use failure::{Error, Fallible};
use indicatif::ProgressStyle;
use ordered_float::NotNan;
use stdinout::OrExit;
use sticker::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::encoder::layer::LayerEncoder;
use sticker::encoder::{CategoricalEncoder, SentenceEncoder};
use sticker::serialization::CborRead;
use sticker::tensorflow::{ConllxDataSet, DataSet, TaggerGraph, TaggerTrainer};
use sticker::wrapper::{Config, EncoderType, LabelerType, TomlRead};
use sticker::{Numberer, SentVectorizer};

use sticker_utils::{CompletedUnit, ReadProgress, SaveSchedule, SaveScheduler};

use crate::{StickerApp, StickerTrainApp};

static EPOCHS: &str = "EPOCHS";
static INITIAL_LR: &str = "INITIAL_LR";
static MAX_LEN: &str = "MAX_LEN";
static CONTINUE: &str = "CONTINUE";
static SAVE_BATCH: &str = "SAVE_BATCH";
static TRAIN_DATA: &str = "TRAIN_DATA";
static VALIDATION_DATA: &str = "VALIDATION_DATA";
static LOGDIR: &str = "LOGDIR";

pub struct PretrainApp {
    config: String,
    epochs: usize,
    initial_lr: NotNan<f32>,
    max_len: usize,
    parameters: Option<String>,
    save_schedule: SaveSchedule,
    train_data: String,
    validation_data: String,
    logdir: Option<String>,
}

impl PretrainApp {
    fn create_trainer(&self, config: &Config) -> Fallible<TaggerTrainer> {
        let graph_read = BufReader::new(File::open(&config.model.graph)?);
        let graph = TaggerGraph::load_graph(graph_read, &config.model)?;

        let mut trainer = match self.parameters {
            Some(ref parameters) => TaggerTrainer::load_weights(graph, parameters),
            None => TaggerTrainer::random_weights(graph),
        }?;
        match &self.logdir {
            Some(logdir) => {
                trainer.init_logdir(logdir)?;
            }
            None => {}
        };
        Ok(trainer)
    }

    fn train_model_with_encoder<E>(
        &self,
        config: &Config,
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
            .input
            .embeddings
            .load_embeddings()
            .or_exit("Cannot load embeddings", 1);
        let vectorizer = SentVectorizer::new(embeddings, config.input.subwords);

        let mut best_epoch = 0;
        let mut best_acc = 0.0;

        let mut save_scheduler = self.save_schedule.to_save_scheduler("pretrain-");

        let train_size = train_file.metadata()?.len() as usize;

        for epoch in 0..self.epochs {
            let (loss, acc) = self.run_epoch(
                config,
                &mut save_scheduler,
                &mut categorical_encoder,
                &vectorizer,
                &trainer,
                &mut train_file,
                true,
                train_size,
                epoch,
            )?;

            eprintln!("Epoch {} (train, loss: {:.4}, acc: {:.4}", epoch, loss, acc);

            save_scheduler.save(&trainer, CompletedUnit::Epoch).or_exit(
                format!("Cannot save model for epoch {}", save_scheduler.epoch()),
                1,
            );

            let (loss, acc) = self.run_epoch(
                config,
                &mut save_scheduler,
                &mut categorical_encoder,
                &vectorizer,
                &trainer,
                &mut validation_file,
                false,
                train_size,
                epoch,
            )?;

            if acc > best_acc {
                best_epoch = epoch;
                best_acc = acc;
            }

            let epoch_status = if best_epoch == epoch { "🎉" } else { "" };

            eprintln!(
            "Epoch {} (validation): loss: {:.4}, acc: {:.4}, best epoch: {}, best acc: {:.4} {}",
            epoch, loss, acc, best_epoch, best_acc, epoch_status
        );
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_epoch<E>(
        &self,
        config: &Config,
        save_scheduler: &mut SaveScheduler,
        encoder: &mut CategoricalEncoder<E, E::Encoding>,
        vectorizer: &SentVectorizer,
        trainer: &TaggerTrainer,
        file: &mut File,
        is_training: bool,
        train_size: usize,
        epoch: usize,
    ) -> Result<(f32, f32), Error>
    where
        E: SentenceEncoder,
        E::Encoding: Clone + Eq + Hash,
    {
        let epoch_type = if is_training { "train" } else { "validation" };

        let read_progress =
            ReadProgress::new(file.try_clone()?).or_exit("Cannot create progress bar", 1);
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
            .batches(encoder, vectorizer, config.model.batch_size, self.max_len)
            .or_exit("Cannot read batches", 1)
        {
            let tensors = batch.or_exit("Cannot read batch", 1).into_parts();

            let batch_perf = if is_training {
                let bytes_done = (epoch * train_size) + file.seek(SeekFrom::Current(0))? as usize;
                let lr_scale = 1f32 - (bytes_done as f32 / (self.epochs * train_size) as f32);
                let lr = lr_scale * self.initial_lr.into_inner();
                let batch_perf = trainer.train(
                    &tensors.seq_lens,
                    &tensors.inputs,
                    tensors.subwords.as_ref(),
                    &tensors.labels,
                    lr,
                );
                progress_bar.set_message(&format!(
                    "lr: {:.6}, loss: {:.4}, accuracy: {:.4}",
                    lr, batch_perf.loss, batch_perf.accuracy
                ));
                batch_perf
            } else {
                let batch_perf = trainer.validate(
                    &tensors.seq_lens,
                    &tensors.inputs,
                    tensors.subwords.as_ref(),
                    &tensors.labels,
                );
                progress_bar.set_message(&format!(
                    "batch loss: {:.4}, batch accuracy: {:.4}",
                    batch_perf.loss, batch_perf.accuracy
                ));
                batch_perf
            };

            let n_tokens = tensors.seq_lens.view().iter().sum::<i32>();
            loss += n_tokens as f32 * batch_perf.loss;
            acc += n_tokens as f32 * batch_perf.accuracy;
            instances += n_tokens;

            if is_training {
                save_scheduler.save(trainer, CompletedUnit::Batch).or_exit(
                    format!("Cannot save model for batch {}", save_scheduler.batch()),
                    1,
                );
            }
        }

        loss /= instances as f32;
        acc /= instances as f32;

        Ok((loss, acc))
    }
}

impl StickerTrainApp for PretrainApp {}

impl StickerApp for PretrainApp {
    fn app() -> App<'static, 'static> {
        Self::train_app("pretrain")
            .about("Pretrain a sticker model")
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
                Arg::with_name(EPOCHS)
                    .long("epochs")
                    .value_name("N")
                    .help("Number of epochs to pretrain")
                    .default_value("1"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(SAVE_BATCH)
                    .long("save-batch")
                    .takes_value(true)
                    .value_name("N")
                    .help("Save after N batches, in addition to saving after every epoch"),
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
    }

    fn parse(matches: &ArgMatches) -> Self {
        let config = matches.value_of(Self::CONFIG).unwrap().into();
        let epochs = matches
            .value_of(EPOCHS)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of training epochs", 1);
        let initial_lr = matches
            .value_of(INITIAL_LR)
            .unwrap()
            .parse()
            .or_exit("Cannot parse initial learning rate", 1);
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().or_exit("Cannot parse maximum sentence length", 1))
            .unwrap_or(usize::MAX);
        let parameters = matches.value_of(CONTINUE).map(ToOwned::to_owned);
        let save_schedule = matches
            .value_of(SAVE_BATCH)
            .map(|n| {
                SaveSchedule::EpochAndBatches(
                    n.parse()
                        .or_exit("Cannot parse number of batches after which to save", 1),
                )
            })
            .unwrap_or(SaveSchedule::Epoch);
        let logdir = matches.value_of(LOGDIR).map(ToOwned::to_owned);

        let train_data = matches.value_of(TRAIN_DATA).unwrap().into();
        let validation_data = matches.value_of(VALIDATION_DATA).unwrap().into();

        PretrainApp {
            config,
            epochs,
            initial_lr,
            max_len,
            parameters,
            save_schedule,
            train_data,
            validation_data,
            logdir,
        }
    }

    fn run(&self) {
        let config_file = File::open(&self.config).or_exit(
            format!("Cannot open configuration file '{}'", self.config),
            1,
        );
        let mut config =
            Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
        config
            .relativize_paths(&self.config)
            .or_exit("Cannot relativize paths in configuration", 1);

        let train_file =
            File::open(&self.train_data).or_exit("Cannot open train file for reading", 1);
        let validation_file =
            File::open(&self.validation_data).or_exit("Cannot open validation file for reading", 1);

        let trainer = self
            .create_trainer(&config)
            .or_exit("Cannot construct trainer", 1);

        match config.labeler.labeler_type {
            LabelerType::Sequence(ref layer) => self.train_model_with_encoder::<LayerEncoder>(
                &config,
                trainer,
                LayerEncoder::new(layer.clone()),
                train_file,
                validation_file,
            ),
            LabelerType::Parser(EncoderType::RelativePOS) => self
                .train_model_with_encoder::<RelativePOSEncoder>(
                    &config,
                    trainer,
                    RelativePOSEncoder,
                    train_file,
                    validation_file,
                ),
            LabelerType::Parser(EncoderType::RelativePosition) => self
                .train_model_with_encoder::<RelativePositionEncoder>(
                    &config,
                    trainer,
                    RelativePositionEncoder,
                    train_file,
                    validation_file,
                ),
        }
        .or_exit("Error while training model", 1);
    }
}
