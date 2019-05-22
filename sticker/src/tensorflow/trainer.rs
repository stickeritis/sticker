use std::path::Path;

use failure::Error;
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
}

impl TaggerTrainer {
    /// Create a new session with randomized weights.
    pub fn random_weights(graph: TaggerGraph) -> Result<Self, Error> {
        // Initialize parameters.
        let mut args = SessionRunArgs::new();
        args.add_target(&graph.init_op);
        let session = Self::new_session(&graph)?;
        session
            .run(&mut args)
            .expect("Cannot initialize parameters");

        Ok(TaggerTrainer { graph, session })
    }

    fn new_session(graph: &TaggerGraph) -> Result<Session, Error> {
        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(&graph.model_config.to_protobuf()?)
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

    /// Train on a batch of inputs and labels.
    pub fn train(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
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

        self.validate_(seq_lens, inputs, labels, args)
    }

    /// Perform validation using a batch of inputs and labels.
    pub fn validate(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
        labels: &NdTensor<i32, Ix2>,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.is_training_op, 0, &is_training);

        self.validate_(seq_lens, inputs, labels, args)
    }

    fn validate_<'l>(
        &'l self,
        seq_lens: &'l NdTensor<i32, Ix1>,
        inputs: &'l NdTensor<f32, Ix3>,
        labels: &'l NdTensor<i32, Ix2>,
        mut args: SessionRunArgs<'l>,
    ) -> ModelPerformance {
        // Add inputs.
        args.add_feed(&self.graph.inputs_op, 0, inputs.inner_ref());
        args.add_feed(&self.graph.seq_lens_op, 0, seq_lens.inner_ref());

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
