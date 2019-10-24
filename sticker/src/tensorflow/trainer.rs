use std::path::Path;

use failure::{err_msg, Error, Fallible};
use ndarray::{Ix1, Ix2, Ix3};
use ndarray_tensorflow::NdTensor;
use tensorflow::{Session, SessionOptions, SessionRunArgs, Tensor};

use super::tagger::TaggerGraph;
use super::util::{prepare_path, status_to_error};
use crate::ModelPerformance;

/// Trainer for a sequence labeling model.
pub struct TaggerTrainer {
    graph: TaggerGraph,
    session: Session,
    summaries: bool,
}

impl TaggerTrainer {
    /// Create a new trainer with loaded weights.
    ///
    /// This constructor will load the model parameters (such as weights) from
    /// the file specified in `parameters_path`.
    pub fn load_weights<P>(graph: TaggerGraph, parameters_path: P) -> Fallible<Self>
    where
        P: AsRef<Path>,
    {
        // Restore parameters.
        let path_tensor = prepare_path(parameters_path)?.into();
        let mut args = SessionRunArgs::new();
        args.add_feed(&graph.save_path_op, 0, &path_tensor);
        args.add_target(&graph.restore_op);
        let session = Self::new_session(&graph)?;
        session.run(&mut args).map_err(status_to_error)?;

        Ok(TaggerTrainer {
            graph,
            session,
            summaries: false,
        })
    }

    /// Create a new session with randomized weights.
    pub fn random_weights(graph: TaggerGraph) -> Result<Self, Error> {
        // Initialize parameters.
        let mut args = SessionRunArgs::new();
        args.add_target(&graph.init_op);
        let session = Self::new_session(&graph)?;
        session
            .run(&mut args)
            .expect("Cannot initialize parameters");

        Ok(TaggerTrainer {
            graph,
            session,
            summaries: false,
        })
    }

    fn new_session(graph: &TaggerGraph) -> Result<Session, Error> {
        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(
                &graph
                    .model_config
                    .to_protobuf(graph.has_auto_mixed_precision()?)?,
            )
            .map_err(status_to_error)?;

        Session::new(&session_opts, &graph.graph).map_err(status_to_error)
    }

    /// Save the model parameters.
    ///
    /// The model parameters are stored as the given path.
    pub fn save<P>(&self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        // Add leading directory component if absent.
        let path_tensor = prepare_path(path)?.into();

        // Call the save op.
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.save_path_op, 0, &path_tensor);
        args.add_target(&self.graph.save_op);
        self.session.run(&mut args).map_err(status_to_error)
    }

    /// Initializes the Tensorboard logdir.
    ///
    /// This method initializes the logdir and adds the graph summary
    /// to the event file.
    pub fn init_logdir<P>(&mut self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        let path_tensor = prepare_path(path)?.into();
        self.summaries = true;

        // First init the summary writer..
        let mut args = SessionRunArgs::new();
        args.add_feed(
            self.graph
                .logdir_op
                .as_ref()
                .ok_or_else(|| err_msg("Graph does not support writing of summaries"))?,
            0,
            &path_tensor,
        );
        args.add_target(
            self.graph
                .summary_init_op
                .as_ref()
                .ok_or_else(|| err_msg("Graph does not support writing of summaries"))?,
        );
        self.session.run(&mut args).map_err(status_to_error)?;

        // ..and then write the graph.
        let mut args = SessionRunArgs::new();
        args.add_target(
            self.graph
                .graph_write_op
                .as_ref()
                .ok_or_else(|| err_msg("Graph does not support writing of summaries"))?,
        );
        self.session.run(&mut args).map_err(status_to_error)
    }

    /// Train on a batch of inputs and labels.
    pub fn train(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
        subwords: Option<&NdTensor<String, Ix2>>,
        labels: &NdTensor<i32, Ix2>,
        learning_rate: f32,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = true;

        let mut lr = Tensor::new(&[]);
        lr[0] = learning_rate;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.is_training_op, 0, &is_training);
        args.add_feed(&self.graph.lr_op, 0, &lr);
        args.add_target(&self.graph.train_op);
        if self.summaries {
            args.add_target(
                &self
                    .graph
                    .train_summary_op
                    .as_ref()
                    .expect("Summaries requested from a graph that does not support summaries."),
            );
        }
        self.validate_(seq_lens, inputs, subwords, labels, args)
    }

    /// Perform validation using a batch of inputs and labels.
    pub fn validate(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
        subwords: Option<&NdTensor<String, Ix2>>,
        labels: &NdTensor<i32, Ix2>,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.is_training_op, 0, &is_training);
        if self.summaries {
            args.add_target(
                self.graph
                    .val_summary_op
                    .as_ref()
                    .expect("Summaries requested from a graph that does not support summaries."),
            );
        }
        self.validate_(seq_lens, inputs, subwords, labels, args)
    }

    fn validate_<'l>(
        &'l self,
        seq_lens: &'l NdTensor<i32, Ix1>,
        inputs: &'l NdTensor<f32, Ix3>,
        subwords: Option<&'l NdTensor<String, Ix2>>,
        labels: &'l NdTensor<i32, Ix2>,
        mut args: SessionRunArgs<'l>,
    ) -> ModelPerformance {
        // Add inputs.
        args.add_feed(&self.graph.inputs_op, 0, inputs.inner_ref());
        args.add_feed(&self.graph.seq_lens_op, 0, seq_lens.inner_ref());

        if let Some(subwords) = subwords {
            args.add_feed(
                self.graph
                    .subwords_op
                    .as_ref()
                    .expect("Subwords used in a graph without support for subwords"),
                0,
                subwords.inner_ref(),
            );
        }

        // Add gold labels.
        args.add_feed(&self.graph.labels_op, 0, labels.inner_ref());

        let accuracy_token = args.request_fetch(&self.graph.accuracy_op, 0);
        let loss_token = args.request_fetch(&self.graph.loss_op, 0);

        self.session.run(&mut args).expect("Cannot run graph");

        ModelPerformance {
            loss: args.fetch(loss_token).expect("Unable to retrieve loss")[0],
            accuracy: args
                .fetch(accuracy_token)
                .expect("Unable to retrieve accuracy")[0],
        }
    }
}
