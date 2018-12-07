use std::cmp::min;
use std::io::Read;
use std::path::Path;

use conllx::Sentence;
use failure::{err_msg, format_err, Error};
use protobuf::Message;
use serde_derive::{Deserialize, Serialize};
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Status,
    Tensor,
};
use tf_proto::ConfigProto;

use super::tensor::TensorBuilder;
use crate::{ModelPerformance, Numberer, SentVectorizer, Tag};

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Model {
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

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct OpNames {
    pub init_op: String,

    pub restore_op: String,
    pub save_op: String,
    pub save_path_op: String,

    pub is_training_op: String,
    pub lr_op: String,

    pub tokens_op: String,
    pub seq_lens_op: String,

    pub loss_op: String,
    pub accuracy_op: String,
    pub labels_op: String,
    pub predicted_op: String,

    pub train_op: String,
}

pub struct Tagger {
    session: Session,

    vectorizer: SentVectorizer,
    labels: Numberer<String>,

    init_op: Operation,
    restore_op: Operation,
    save_op: Operation,
    save_path_op: Operation,
    lr_op: Operation,
    is_training_op: Operation,
    tokens_op: Operation,
    seq_lens_op: Operation,

    loss_op: Operation,
    accuracy_op: Operation,
    labels_op: Operation,
    predicted_op: Operation,

    train_op: Operation,
}

impl Tagger {
    pub fn load_graph<R>(
        r: R,
        vectorizer: SentVectorizer,
        labels: Numberer<String>,
        model: &Model,
    ) -> Result<Self, Error>
    where
        R: Read,
    {
        let mut tagger = Self::load_graph_(r, vectorizer, labels, model)?;

        // Initialize parameters.
        let mut args = SessionRunArgs::new();
        args.add_target(&tagger.init_op);
        tagger
            .session
            .run(&mut args)
            .expect("Cannot initialize parameters");

        Ok(tagger)
    }

    /// Load a Tensorflow graph with weights.
    ///
    /// This constructor will load the model parameters (such as weights) from
    /// the file specified in `parameters_path`.
    pub fn load_graph_with_weights<P, R>(
        graph_read: R,
        parameters_path: P,
        vectorizer: SentVectorizer,
        labels: Numberer<String>,
        model: &Model,
    ) -> Result<Self, Error>
    where
        P: AsRef<Path>,
        R: Read,
    {
        let mut model = Self::load_graph_(graph_read, vectorizer, labels, model)?;

        // Restore parameters.
        let path_tensor = prepare_path(parameters_path)?.into();
        let mut args = SessionRunArgs::new();
        args.add_feed(&model.save_path_op, 0, &path_tensor);
        args.add_target(&model.restore_op);
        model.session.run(&mut args).map_err(status_to_error)?;

        Ok(model)
    }

    fn load_graph_<R>(
        mut r: R,
        vectorizer: SentVectorizer,
        labels: Numberer<String>,
        model: &Model,
    ) -> Result<Self, Error>
    where
        R: Read,
    {
        let mut data = Vec::new();
        r.read_to_end(&mut data)?;

        let opts = ImportGraphDefOptions::new();
        let mut graph = Graph::new();
        graph
            .import_graph_def(&data, &opts)
            .map_err(status_to_error)?;

        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(&tf_model_to_protobuf(&model)?)
            .map_err(status_to_error)?;
        let session = Session::new(&session_opts, &graph).map_err(status_to_error)?;

        let op_names = &model.op_names;

        let restore_op = Self::add_op(&graph, &op_names.restore_op)?;
        let save_op = Self::add_op(&graph, &op_names.save_op)?;
        let save_path_op = Self::add_op(&graph, &op_names.save_path_op)?;
        let init_op = Self::add_op(&graph, &op_names.init_op)?;

        let is_training_op = Self::add_op(&graph, &op_names.is_training_op)?;
        let lr_op = Self::add_op(&graph, &op_names.lr_op)?;

        let tokens_op = Self::add_op(&graph, &op_names.tokens_op)?;
        let seq_lens_op = Self::add_op(&graph, &op_names.seq_lens_op)?;

        let loss_op = Self::add_op(&graph, &op_names.loss_op)?;
        let accuracy_op = Self::add_op(&graph, &op_names.accuracy_op)?;
        let labels_op = Self::add_op(&graph, &op_names.labels_op)?;
        let predicted_op = Self::add_op(&graph, &op_names.predicted_op)?;

        let train_op = Self::add_op(&graph, &op_names.train_op)?;

        Ok(Tagger {
            session,

            vectorizer,
            labels,

            init_op,
            restore_op,
            save_op,
            save_path_op,
            is_training_op,
            lr_op,
            tokens_op,
            seq_lens_op,

            loss_op,
            accuracy_op,
            labels_op,
            predicted_op,

            train_op,
        })
    }

    fn add_op(graph: &Graph, name: &str) -> Result<Operation, Error> {
        graph
            .operation_by_name_required(name)
            .map_err(status_to_error)
    }

    /// Save the model parameters.
    ///
    /// The model parameters are stored as the given path.
    pub fn save<P>(&mut self, path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        // Add leading directory component if absent.
        let path_tensor = prepare_path(path)?.into();

        // Call the save op.
        let mut args = SessionRunArgs::new();
        args.add_feed(&self.save_path_op, 0, &path_tensor);
        args.add_target(&self.save_op);
        self.session.run(&mut args).map_err(status_to_error)
    }

    fn tag_sequences(
        &mut self,
        seq_lens: &Tensor<i32>,
        tokens: &Tensor<f32>,
    ) -> Result<Tensor<i32>, Error> {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();

        args.add_feed(&self.is_training_op, 0, &is_training);

        // Sequence inputs
        args.add_feed(&self.seq_lens_op, 0, seq_lens);
        args.add_feed(&self.tokens_op, 0, tokens);

        let predictions_token = args.request_fetch(&self.predicted_op, 0);

        self.session.run(&mut args).map_err(status_to_error)?;

        Ok(args.fetch(predictions_token).map_err(status_to_error)?)
    }

    pub fn validate(
        &mut self,
        seq_lens: &Tensor<i32>,
        tokens: &Tensor<f32>,
        labels: &Tensor<i32>,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();

        args.add_feed(&self.is_training_op, 0, &is_training);

        self.validate_(seq_lens, tokens, labels, args)
    }

    pub fn train(
        &mut self,
        seq_lens: &Tensor<i32>,
        tokens: &Tensor<f32>,
        labels: &Tensor<i32>,
        learning_rate: f32,
    ) -> ModelPerformance {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = true;

        let mut lr = Tensor::new(&[]);
        lr[0] = learning_rate;

        let mut args = SessionRunArgs::new();
        args.add_feed(&self.is_training_op, 0, &is_training);
        args.add_feed(&self.lr_op, 0, &lr);
        args.add_target(&self.train_op);

        self.validate_(seq_lens, tokens, labels, args)
    }

    fn validate_<'l>(
        &'l mut self,
        seq_lens: &'l Tensor<i32>,
        tokens: &'l Tensor<f32>,
        labels: &'l Tensor<i32>,
        mut args: SessionRunArgs<'l>,
    ) -> ModelPerformance {
        // Add inputs.
        args.add_feed(&self.tokens_op, 0, tokens);
        args.add_feed(&self.seq_lens_op, 0, seq_lens);

        // Add gold labels.
        args.add_feed(&self.labels_op, 0, labels);

        let accuracy_token = args.request_fetch(&self.accuracy_op, 0);
        let loss_token = args.request_fetch(&self.loss_op, 0);

        self.session.run(&mut args).expect("Cannot run graph");

        ModelPerformance {
            loss: args.fetch(loss_token).expect("Unable to retrieve loss")[0],
            accuracy: args
                .fetch(accuracy_token)
                .expect("Unable to retrieve accuracy")[0],
        }
    }
}

impl Tag for Tagger {
    fn tag_sentences(&mut self, sentences: &[Sentence]) -> Result<Vec<Vec<&str>>, Error> {
        // Find maximum sentence size.
        let max_seq_len = sentences.iter().map(|s| s.len()).max().unwrap_or(0);

        let token_dims = self.vectorizer.layer_embeddings().token_embeddings().dims();

        let mut builder = TensorBuilder::new(sentences.len(), max_seq_len, token_dims);

        // Fill the batch.
        for sentence in sentences {
            let input = self.vectorizer.realize(sentence)?;
            builder.add(&input);
        }

        // Tag the batch
        let tag_tensor = self.tag_sequences(builder.seq_lens(), builder.tokens())?;

        // Convert label numbers to labels.
        let numberer = &self.labels;
        let mut labels = Vec::new();
        for (idx, sentence) in sentences.iter().enumerate() {
            let seq_len = min(max_seq_len, sentence.len());
            let offset = idx * max_seq_len;
            let seq = &tag_tensor[offset..offset + seq_len];

            labels.push(
                seq.iter()
                    .map(|&label| numberer.value(label as usize).unwrap().as_str())
                    .collect(),
            );
        }

        Ok(labels)
    }
}

fn tf_model_to_protobuf(model: &Model) -> Result<Vec<u8>, Error> {
    let mut config_proto = ConfigProto::new();
    config_proto.intra_op_parallelism_threads = model.intra_op_parallelism_threads as i32;
    config_proto.inter_op_parallelism_threads = model.inter_op_parallelism_threads as i32;

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
        .ok_or(err_msg("Filename contains non-unicode characters"))
        .map(ToOwned::to_owned)
}

/// tensorflow::Status is not Sync, which is required by failure.
fn status_to_error(status: Status) -> Error {
    format_err!("{}", status)
}
