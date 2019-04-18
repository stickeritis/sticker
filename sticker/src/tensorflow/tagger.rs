use std::borrow::Borrow;
use std::cmp::min;
use std::hash::Hash;
use std::io::Read;
use std::iter::FromIterator;
use std::path::Path;

use conllx::graph::Sentence;
use failure::{err_msg, format_err, Error};
use itertools::Itertools;
use protobuf::Message;
use serde_derive::{Deserialize, Serialize};
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Status,
    Tensor,
};
use tf_proto::ConfigProto;

use super::tensor::TensorBuilder;
use crate::{ModelPerformance, Numberer, SentVectorizer, Tag};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelConfig {
    /// Model batch size, should be kept constant between training and
    /// prediction.
    pub batch_size: usize,

    /// The filename of the Tensorflow graph.
    pub graph: String,

    /// Operation names for the frozen tensorflow graph.
    pub op_names: OpNames,

    /// Thread pool size for parallel processing within a computation
    /// graph op.
    pub intra_op_parallelism_threads: usize,

    /// Thread pool size for parallel processing of independent computation
    /// graph ops.
    pub inter_op_parallelism_threads: usize,

    /// The filename of the trained graph parameters.
    pub parameters: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OpNames {
    pub init_op: String,

    pub restore_op: String,
    pub save_op: String,
    pub save_path_op: String,

    pub is_training_op: String,
    pub lr_op: String,

    pub inputs_op: String,
    pub seq_lens_op: String,

    pub loss_op: String,
    pub accuracy_op: String,
    pub labels_op: String,
    pub top_k_predicted_op: String,

    pub train_op: String,
}

pub struct TaggerGraph {
    graph: Graph,
    model_config: ModelConfig,

    init_op: Operation,
    restore_op: Operation,
    save_op: Operation,
    save_path_op: Operation,
    lr_op: Operation,
    is_training_op: Operation,
    inputs_op: Operation,
    seq_lens_op: Operation,

    loss_op: Operation,
    accuracy_op: Operation,
    labels_op: Operation,
    top_k_predicted_op: Operation,

    train_op: Operation,
}

impl TaggerGraph {
    pub fn load_graph<R>(mut graph_read: R, model_config: &ModelConfig) -> Result<Self, Error>
    where
        R: Read,
    {
        let mut data = Vec::new();
        graph_read.read_to_end(&mut data)?;

        let opts = ImportGraphDefOptions::new();
        let mut graph = Graph::new();
        graph
            .import_graph_def(&data, &opts)
            .map_err(status_to_error)?;

        let op_names = &model_config.op_names;

        let restore_op = Self::add_op(&graph, &op_names.restore_op)?;
        let save_op = Self::add_op(&graph, &op_names.save_op)?;
        let save_path_op = Self::add_op(&graph, &op_names.save_path_op)?;
        let init_op = Self::add_op(&graph, &op_names.init_op)?;

        let is_training_op = Self::add_op(&graph, &op_names.is_training_op)?;
        let lr_op = Self::add_op(&graph, &op_names.lr_op)?;

        let inputs_op = Self::add_op(&graph, &op_names.inputs_op)?;
        let seq_lens_op = Self::add_op(&graph, &op_names.seq_lens_op)?;

        let loss_op = Self::add_op(&graph, &op_names.loss_op)?;
        let accuracy_op = Self::add_op(&graph, &op_names.accuracy_op)?;
        let labels_op = Self::add_op(&graph, &op_names.labels_op)?;
        let top_k_predicted_op = Self::add_op(&graph, &op_names.top_k_predicted_op)?;

        let train_op = Self::add_op(&graph, &op_names.train_op)?;

        Ok(TaggerGraph {
            graph,
            model_config: model_config.clone(),

            init_op,
            restore_op,
            save_op,
            save_path_op,
            is_training_op,
            lr_op,
            inputs_op,
            seq_lens_op,

            loss_op,
            accuracy_op,
            labels_op,
            top_k_predicted_op,

            train_op,
        })
    }

    fn add_op(graph: &Graph, name: &str) -> Result<Operation, Error> {
        graph
            .operation_by_name_required(name)
            .map_err(status_to_error)
    }
}

pub struct Tagger<T>
where
    T: Eq + Hash,
{
    graph: TaggerGraph,
    labels: Numberer<T>,
    session: Session,
    vectorizer: SentVectorizer,
}

impl<T> Tagger<T>
where
    T: Clone + Eq + Hash,
{
    /// Create a new session with randomized weights.
    pub fn random_weights(
        graph: TaggerGraph,
        labels: Numberer<T>,
        vectorizer: SentVectorizer,
    ) -> Result<Tagger<T>, Error> {
        // Initialize parameters.
        let mut args = SessionRunArgs::new();
        args.add_target(&graph.init_op);
        let session = Self::new_session(&graph)?;
        session
            .run(&mut args)
            .expect("Cannot initialize parameters");

        Ok(Tagger {
            graph,
            labels,
            session,
            vectorizer,
        })
    }

    /// Load a tagger with weights.
    ///
    /// This constructor will load the model parameters (such as weights) from
    /// the file specified in `parameters_path`.
    pub fn load_weights<P>(
        graph: TaggerGraph,
        labels: Numberer<T>,
        vectorizer: SentVectorizer,
        parameters_path: P,
    ) -> Result<Tagger<T>, Error>
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

        Ok(Tagger {
            graph,
            labels,
            session,
            vectorizer,
        })
    }

    fn new_session(graph: &TaggerGraph) -> Result<Session, Error> {
        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(&tf_model_config_to_protobuf(&graph.model_config)?)
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

    fn input_dims(&self) -> usize {
        self.vectorizer.layer_embeddings().token_embeddings().dims()
            + self
                .vectorizer
                .layer_embeddings()
                .tag_embeddings()
                .as_ref()
                .map(|e| e.dims())
                .unwrap_or_default()
    }

    fn prepare_batch(&self, sentences: &[impl Borrow<Sentence>]) -> Result<TensorBuilder, Error> {
        // Find maximum sentence size.
        let max_seq_len = sentences
            .iter()
            .map(|s| s.borrow().len() - 1)
            .max()
            .unwrap_or(0);

        let inputs_dims = self.input_dims();

        let mut builder = TensorBuilder::new(sentences.len(), max_seq_len, inputs_dims);

        // Fill the batch.
        for sentence in sentences {
            let input = self.vectorizer.realize(sentence.borrow())?;
            builder.add(&input);
        }

        Ok(builder)
    }

    fn tag_sentences_(
        &self,
        sentences: &[impl Borrow<Sentence>],
    ) -> Result<Vec<Vec<Vec<&T>>>, Error> {
        let builder = self.prepare_batch(sentences)?;

        // Tag the batch
        let tag_tensor = self.tag_sequences(
            builder.seq_lens(),
            builder.inputs(),
            &self.graph.top_k_predicted_op,
        )?;

        // Convert label numbers to labels.
        let max_seq_len = tag_tensor.dims()[1] as usize;
        let k = tag_tensor.dims()[2] as usize;
        let mut labels = Vec::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            let seq_len = min(max_seq_len, sentence.borrow().len() - 1);
            let offset = idx * max_seq_len * k;
            let seq = &tag_tensor[offset..offset + seq_len * k];

            labels.push(
                seq.iter()
                    .chunks(k)
                    .into_iter()
                    .map(|c| c.map(|&label| self.labels.value(label as usize).unwrap()))
                    .map(Vec::from_iter)
                    .collect(),
            );
        }

        Ok(labels)
    }

    fn tag_sequences(
        &self,
        seq_lens: &Tensor<i32>,
        inputs: &Tensor<f32>,
        predicted_op: &Operation,
    ) -> Result<Tensor<i32>, Error> {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();

        args.add_feed(&self.graph.is_training_op, 0, &is_training);

        // Sequence inputs
        args.add_feed(&self.graph.seq_lens_op, 0, seq_lens);
        args.add_feed(&self.graph.inputs_op, 0, inputs);

        let predictions_token = args.request_fetch(predicted_op, 0);

        self.session.run(&mut args).map_err(status_to_error)?;

        Ok(args.fetch(predictions_token).map_err(status_to_error)?)
    }

    pub fn train(
        &self,
        seq_lens: &Tensor<i32>,
        inputs: &Tensor<f32>,
        labels: &Tensor<i32>,
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

    pub fn validate(
        &self,
        seq_lens: &Tensor<i32>,
        inputs: &Tensor<f32>,
        labels: &Tensor<i32>,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.graph.is_training_op, 0, &is_training);

        self.validate_(seq_lens, inputs, labels, args)
    }

    fn validate_<'l>(
        &'l self,
        seq_lens: &'l Tensor<i32>,
        inputs: &'l Tensor<f32>,
        labels: &'l Tensor<i32>,
        mut args: SessionRunArgs<'l>,
    ) -> ModelPerformance {
        // Add inputs.
        args.add_feed(&self.graph.inputs_op, 0, inputs);
        args.add_feed(&self.graph.seq_lens_op, 0, seq_lens);

        // Add gold labels.
        args.add_feed(&self.graph.labels_op, 0, labels);

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

impl<T> Tag<T> for Tagger<T>
where
    T: Clone + Eq + Hash,
{
    /// Tag sentences, returning the top-k results for every token.
    fn tag_sentences(
        &self,
        sentences: &[impl Borrow<Sentence>],
    ) -> Result<Vec<Vec<Vec<&T>>>, Error> {
        self.tag_sentences_(sentences)
    }
}

fn tf_model_config_to_protobuf(model_config: &ModelConfig) -> Result<Vec<u8>, Error> {
    let mut config_proto = ConfigProto::new();
    config_proto.intra_op_parallelism_threads = model_config.intra_op_parallelism_threads as i32;
    config_proto.inter_op_parallelism_threads = model_config.inter_op_parallelism_threads as i32;

    let mut bytes = Vec::new();
    config_proto.write_to_vec(&mut bytes)?;

    Ok(bytes)
}

/// Tensorflow requires a path that contains a directory component.
fn prepare_path<P>(path: P) -> Result<String, Error>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let path = if path.components().count() == 1 {
        Path::new("./").join(path)
    } else {
        path.to_owned()
    };

    path.to_str()
        .ok_or_else(|| err_msg("Filename contains non-unicode characters"))
        .map(ToOwned::to_owned)
}

/// tensorflow::Status is not Sync, which is required by failure.
fn status_to_error(status: Status) -> Error {
    format_err!("{}", status)
}
