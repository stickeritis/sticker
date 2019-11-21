use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, Read, Seek};
use std::usize;

use clap::{App, Arg, ArgMatches};
use failure::{Error, Fallible, ResultExt};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ordered_float::NotNan;
use stdinout::OrExit;
use sticker::encoder::categorical::ImmutableCategoricalEncoder;
use sticker::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::encoder::layer::LayerEncoder;
use sticker::encoder::lemma::EditTreeEncoder;
use sticker::encoder::SentenceEncoder;
use sticker::serialization::CborRead;
use sticker::tensorflow::{
    ConllxDataSet, DataSet, LabelTensor, RuntimeConfig, TaggerGraph, TaggerTrainer, TensorBuilder,
};
use sticker::wrapper::{Config, EncoderType, LabelerType, TomlRead};
use sticker::{Numberer, SentVectorizer};

use crate::progress::ReadProgress;
use crate::traits::{StickerApp, StickerTrainApp};
use crate::util::count_conllx_sentences;

static EPOCHS: &str = "EPOCHS";
static INITIAL_LR: &str = "INITIAL_LR";
static WARMUP: &str = "WARMUP";
static MAX_LEN: &str = "MAX_LEN";
static SHUFFLE: &str = "SHUFFLE";
static CONTINUE: &str = "CONTINUE";
static EVAL_STEPS: &str = "EVAL_STEPS";
static STEPS: &str = "N_STEPS";
static TRAIN_DATA: &str = "TRAIN_DATA";
static VALIDATION_DATA: &str = "VALIDATION_DATA";
static LOGDIR: &str = "LOGDIR";

#[derive(Clone)]
pub struct PretrainApp {
    batch_size: usize,
    config: String,
    eval_steps: usize,
    initial_lr: NotNan<f32>,
    warmup_steps: usize,
    max_len: Option<usize>,
    shuffle_buffer_size: Option<usize>,
    parameters: Option<String>,
    runtime_config: RuntimeConfig,
    train_data: String,
    train_duration: TrainDuration,
    validation_data: String,
    logdir: Option<String>,
}

impl PretrainApp {
    fn create_trainer(&self, config: &Config) -> Fallible<TaggerTrainer> {
        let graph_read = BufReader::new(File::open(&config.model.graph)?);
        let graph = TaggerGraph::load_graph(graph_read, &config.model)?;

        let mut trainer = match self.parameters {
            Some(ref parameters) => {
                TaggerTrainer::load_weights(graph, &self.runtime_config, parameters)
            }
            None => TaggerTrainer::random_weights(graph, &self.runtime_config),
        }?;
        match &self.logdir {
            Some(logdir) => {
                trainer.init_logdir(logdir)?;
            }
            None => {}
        };
        Ok(trainer)
    }

    fn open_dataset(file: &File) -> Fallible<ConllxDataSet<impl Read + Seek>> {
        let read = BufReader::new(file.try_clone()?);
        Ok(ConllxDataSet::new(read))
    }

    fn open_dataset_with_progress(
        file: &File,
        phase: &str,
    ) -> Fallible<(ConllxDataSet<impl Read + Seek>, ProgressBar)> {
        let read_progress =
            ReadProgress::new(file.try_clone()?).or_exit("Cannot open file for reading", 1);
        let progress_bar = read_progress.progress_bar().clone();
        progress_bar.set_style(ProgressStyle::default_bar().template(&format!(
            "[Time: {{elapsed_precise}}, ETA: {{eta_precise}}] {{bar}} {{percent}}% {} {{msg}}",
            phase
        )));

        Ok((ConllxDataSet::new(read_progress), progress_bar))
    }

    fn train_model_with_encoder<E>(
        &self,
        config: &Config,
        trainer: TaggerTrainer,
        encoder: E,
        train_file: File,
        validation_file: File,
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

        let categorical_encoder = ImmutableCategoricalEncoder::new(encoder, labels);

        let embeddings = config
            .input
            .embeddings
            .load_embeddings()
            .or_exit("Cannot load embeddings", 1);
        let vectorizer = SentVectorizer::new(embeddings, config.input.subwords);

        let mut best_step = 0;
        let mut best_acc = 0.0;

        let mut global_step = 0;
        let n_steps = self.train_duration.to_steps(&train_file, self.batch_size)?;

        let train_progress = ProgressBar::new(n_steps as u64);
        train_progress.set_style(ProgressStyle::default_bar().template(
            "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}% train {msg}",
        ));

        // Continue performing steps until we are done.
        while global_step < n_steps - 1 {
            let mut train_dataset = Self::open_dataset(&train_file)?;
            let train_batches = train_dataset.batches(
                &categorical_encoder,
                &vectorizer,
                self.batch_size,
                self.max_len,
                self.shuffle_buffer_size,
            )?;

            for steps in &train_batches.chunks(self.eval_steps) {
                let (_, _, global_step_after_steps) =
                    self.train_steps(&train_progress, &trainer, steps, global_step, n_steps)?;
                global_step = global_step_after_steps;

                let (loss, acc) = self.validation_epoch(
                    &trainer,
                    &validation_file,
                    &categorical_encoder,
                    &vectorizer,
                    global_step,
                )?;

                if acc > best_acc {
                    best_step = global_step;
                    best_acc = acc;

                    trainer
                        .save(format!("{}step-{}", "pretrain-", global_step))
                        .context(format!("Cannot save model for step {}", global_step))?;
                }

                let step_status = if best_step == global_step { "🎉" } else { "" };

                eprintln!("Step {} (validation): loss: {:.4}, acc: {:.4}, best step: {}, best acc: {:.4} {}\n",
			  global_step, loss, acc, best_step, best_acc, step_status);

                if global_step >= n_steps - 1 {
                    break;
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn train_steps(
        &self,
        progress: &ProgressBar,
        trainer: &TaggerTrainer,
        batches: impl Iterator<Item = Fallible<TensorBuilder<LabelTensor>>>,
        mut global_step: usize,
        n_steps: usize,
    ) -> Result<(f32, f32, usize), Error> {
        let mut instances = 0;
        let mut acc = 0f32;
        let mut loss = 0f32;
        for batch in batches {
            let tensors = batch.or_exit("Cannot read batch", 1).into_parts();

            let lr = if global_step < self.warmup_steps {
                (self.initial_lr.into_inner() / (self.warmup_steps as f32)) * global_step as f32
            } else {
                let lr_scale = 1f32 - (global_step as f32 / n_steps as f32);
                lr_scale * self.initial_lr.into_inner()
            };

            let batch_perf = trainer.train(
                &tensors.seq_lens,
                &tensors.inputs,
                tensors.subwords.as_ref(),
                &tensors.labels,
                lr,
            );

            progress.set_message(&format!(
                "step: {}, lr: {:.6}, loss: {:.4}, accuracy: {:.4}",
                global_step, lr, batch_perf.loss, batch_perf.accuracy
            ));
            progress.inc(1);

            global_step += 1;

            let n_tokens = tensors.seq_lens.view().iter().sum::<i32>();
            loss += n_tokens as f32 * batch_perf.loss;
            acc += n_tokens as f32 * batch_perf.accuracy;
            instances += n_tokens;
        }

        loss /= instances as f32;
        acc /= instances as f32;

        Ok((loss, acc, global_step))
    }

    #[allow(clippy::too_many_arguments)]
    fn validation_epoch<E>(
        &self,
        trainer: &TaggerTrainer,
        file: &File,
        categorical_encoder: &ImmutableCategoricalEncoder<E, E::Encoding>,
        vectorizer: &SentVectorizer,
        global_step: usize,
    ) -> Result<(f32, f32), Error>
    where
        E: SentenceEncoder,
        E::Encoding: Clone + Eq + Hash,
    {
        let (mut dataset, progress) = Self::open_dataset_with_progress(&file, "validation")?;
        let batches = dataset.batches(
            &categorical_encoder,
            &vectorizer,
            self.batch_size,
            self.max_len,
            None,
        )?;

        let (loss, acc) = self.validation_steps(&progress, trainer, batches, global_step)?;

        progress.finish_and_clear();

        Ok((loss, acc))
    }

    #[allow(clippy::too_many_arguments)]
    fn validation_steps(
        &self,
        progress: &ProgressBar,
        trainer: &TaggerTrainer,
        batches: impl Iterator<Item = Fallible<TensorBuilder<LabelTensor>>>,
        global_step: usize,
    ) -> Result<(f32, f32), Error> {
        let mut instances = 0;
        let mut acc = 0f32;
        let mut loss = 0f32;

        for batch in batches {
            let tensors = batch.or_exit("Cannot read batch", 1).into_parts();

            let batch_perf = trainer.validate(
                &tensors.seq_lens,
                &tensors.inputs,
                tensors.subwords.as_ref(),
                &tensors.labels,
            );

            progress.set_message(&format!(
                "step: {}, batch loss: {:.4}, batch accuracy: {:.4}",
                global_step, batch_perf.loss, batch_perf.accuracy
            ));

            let n_tokens = tensors.seq_lens.view().iter().sum::<i32>();
            loss += n_tokens as f32 * batch_perf.loss;
            acc += n_tokens as f32 * batch_perf.accuracy;
            instances += n_tokens;
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
                Arg::with_name(EPOCHS)
                    .long("epochs")
                    .takes_value(true)
                    .value_name("N")
                    .help("Train for N epochs")
                    .default_value("2"),
            )
            .arg(
                Arg::with_name(INITIAL_LR)
                    .long("lr")
                    .value_name("LR")
                    .help("Initial learning rate")
                    .default_value("0.01"),
            )
            .arg(
                Arg::with_name(WARMUP)
                    .long("warmup")
                    .value_name("N")
                    .help(
                        "For the first N timesteps, the learning rate is linearly scaled up to LR.",
                    )
                    .default_value("2000"),
            )
            .arg(
                Arg::with_name(MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .takes_value(true)
                    .help("Ignore sentences longer than N tokens"),
            )
            .arg(
                Arg::with_name(SHUFFLE)
                    .long("shuffle_buffer")
                    .value_name("N")
                    .help("Size of the buffer used for shuffling.")
                    .takes_value(true),
            )
            .arg(
                Arg::with_name(EVAL_STEPS)
                    .long("eval_steps")
                    .takes_value(true)
                    .value_name("N")
                    .help("Evaluate after N steps, save the model on improvement")
                    .default_value("1000"),
            )
            .arg(
                Arg::with_name(STEPS)
                    .long("steps")
                    .value_name("N")
                    .help("Train for N steps")
                    .takes_value(true)
                    .overrides_with(EPOCHS),
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
        let batch_size = matches
            .value_of(Self::BATCH_SIZE)
            .unwrap()
            .parse()
            .or_exit("Cannot parse batch size", 1);
        let initial_lr = matches
            .value_of(INITIAL_LR)
            .unwrap()
            .parse()
            .or_exit("Cannot parse initial learning rate", 1);
        let inter_op_threads = matches
            .value_of(Self::INTER_OP_THREADS)
            .unwrap()
            .parse()
            .or_exit("Cannot number of inter op threads", 1);
        let intra_op_threads = matches
            .value_of(Self::INTRA_OP_THREADS)
            .unwrap()
            .parse()
            .or_exit("Cannot number of intra op threads", 1);
        let warmup_steps = matches
            .value_of(WARMUP)
            .unwrap()
            .parse()
            .or_exit("Cannot parse warmup", 1);
        let max_len = matches
            .value_of(MAX_LEN)
            .map(|v| v.parse().or_exit("Cannot parse maximum sentence length", 1));
        let shuffle_buffer_size = matches
            .value_of(SHUFFLE)
            .map(|v| v.parse().or_exit("Cannot parse shuffle buffer size.", 1));
        let parameters = matches.value_of(CONTINUE).map(ToOwned::to_owned);
        let eval_steps = matches
            .value_of(EVAL_STEPS)
            .unwrap()
            .parse()
            .or_exit("Cannot parse number of batches after which to save", 1);
        let logdir = matches.value_of(LOGDIR).map(ToOwned::to_owned);

        // If steps is present, it overrides epochs.
        let train_duration = if let Some(steps) = matches.value_of(STEPS) {
            let steps = steps
                .parse()
                .or_exit("Cannot parse the number of training steps", 1);
            TrainDuration::Steps(steps)
        } else {
            let epochs = matches
                .value_of(EPOCHS)
                .unwrap()
                .parse()
                .or_exit("Cannot parse number of training epochs", 1);
            TrainDuration::Epochs(epochs)
        };

        let train_data = matches.value_of(TRAIN_DATA).unwrap().into();
        let validation_data = matches.value_of(VALIDATION_DATA).unwrap().into();

        PretrainApp {
            batch_size,
            config,
            eval_steps,
            initial_lr,
            warmup_steps,
            max_len,
            shuffle_buffer_size,
            parameters,
            runtime_config: RuntimeConfig {
                inter_op_threads,
                intra_op_threads,
            },
            train_data,
            train_duration,
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
            LabelerType::Lemma => self.train_model_with_encoder::<EditTreeEncoder>(
                &config,
                trainer,
                EditTreeEncoder,
                train_file,
                validation_file,
            ),
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrainDuration {
    Epochs(usize),
    Steps(usize),
}

impl TrainDuration {
    fn to_steps(&self, train_file: &File, batch_size: usize) -> Fallible<usize> {
        use TrainDuration::*;

        match *self {
            Epochs(epochs) => {
                eprintln!("Counting number of steps in an epoch...");
                let read_progress =
                    ReadProgress::new(train_file.try_clone()?).or_exit("Cannot open train file", 1);

                let progress_bar = read_progress.progress_bar().clone();
                progress_bar
                    .set_style(ProgressStyle::default_bar().template(
                        "[Time: {elapsed_precise}, ETA: {eta_precise}] {bar} {percent}%",
                    ));

                let n_sentences = count_conllx_sentences(BufReader::new(read_progress))?;

                progress_bar.finish_and_clear();

                // Compute number of steps of the given batch size.
                let steps_per_epoch = (n_sentences + batch_size - 1) / batch_size;
                eprintln!(
                    "sentences: {}, steps_per epoch: {}, total_steps: {}",
                    n_sentences,
                    steps_per_epoch,
                    epochs * steps_per_epoch
                );
                Ok(epochs * steps_per_epoch)
            }
            Steps(steps) => Ok(steps),
        }
    }
}
