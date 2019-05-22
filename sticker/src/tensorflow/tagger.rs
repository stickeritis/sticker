use std::borrow::{Borrow, BorrowMut};
use std::cmp::min;
use std::hash::Hash;
use std::io::Read;
use std::iter::FromIterator;
use std::path::Path;

use conllx::graph::Sentence;
use failure::{Error, Fallible};
use itertools::Itertools;
use ndarray::{Ix1, Ix3};
use ndarray_tensorflow::NdTensor;
use protobuf::Message;
use serde_derive::{Deserialize, Serialize};
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
};
use tf_proto::ConfigProto;

use super::tensor::{NoLabels, TensorBuilder};
use super::util::{prepare_path, status_to_error};
use crate::encoder::{CategoricalEncoder, SentenceDecoder};
use crate::{EncodingProb, SentVectorizer, Tag};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Model batch size, should be kept constant between training and
    /// prediction.
    pub batch_size: usize,

    /// The filename of the Tensorflow graph.
    pub graph: String,

    /// Thread pool size for parallel processing within a computation
    /// graph op.
    pub intra_op_parallelism_threads: usize,

    /// Thread pool size for parallel processing of independent computation
    /// graph ops.
    pub inter_op_parallelism_threads: usize,

    /// The filename of the trained graph parameters.
    pub parameters: String,
}

impl ModelConfig {
    pub fn to_protobuf(&self) -> Result<Vec<u8>, Error> {
        let mut config_proto = ConfigProto::new();
        config_proto.intra_op_parallelism_threads = self.intra_op_parallelism_threads as i32;
        config_proto.inter_op_parallelism_threads = self.inter_op_parallelism_threads as i32;

        let mut bytes = Vec::new();
        config_proto.write_to_vec(&mut bytes)?;

        Ok(bytes)
    }
}

mod op_names {
    pub const INIT_OP: &str = "init";

    pub const RESTORE_OP: &str = "save/restore_all";
    pub const SAVE_OP: &str = "save/control_dependency";
    pub const SAVE_PATH_OP: &str = "save/Const";

    pub const IS_TRAINING_OP: &str = "model/is_training";
    pub const LR_OP: &str = "model/lr";

    pub const INPUTS_OP: &str = "model/inputs";
    pub const SEQ_LENS_OP: &str = "model/seq_lens";

    pub const LOSS_OP: &str = "model/tag_loss";
    pub const ACCURACY_OP: &str = "model/tag_accuracy";
    pub const LABELS_OP: &str = "model/tags";
    pub const TOP_K_PREDICTED_OP: &str = "model/tag_top_k_predictions";
    pub const TOP_K_PROBS_OP: &str = "model/tag_top_k_probs";

    pub const TRAIN_OP: &str = "model/train";
}

pub struct TaggerGraph {
    pub(crate) graph: Graph,
    pub(crate) model_config: ModelConfig,

    pub(crate) init_op: Operation,
    pub(crate) restore_op: Operation,
    pub(crate) save_op: Operation,
    pub(crate) save_path_op: Operation,
    pub(crate) lr_op: Operation,
    pub(crate) is_training_op: Operation,
    pub(crate) inputs_op: Operation,
    pub(crate) seq_lens_op: Operation,

    pub(crate) loss_op: Operation,
    pub(crate) accuracy_op: Operation,
    pub(crate) labels_op: Operation,
    pub(crate) top_k_predicted_op: Operation,
    pub(crate) top_k_probs_op: Operation,

    pub(crate) train_op: Operation,
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

        let restore_op = Self::add_op(&graph, op_names::RESTORE_OP)?;
        let save_op = Self::add_op(&graph, op_names::SAVE_OP)?;
        let save_path_op = Self::add_op(&graph, op_names::SAVE_PATH_OP)?;
        let init_op = Self::add_op(&graph, op_names::INIT_OP)?;

        let is_training_op = Self::add_op(&graph, op_names::IS_TRAINING_OP)?;
        let lr_op = Self::add_op(&graph, op_names::LR_OP)?;

        let inputs_op = Self::add_op(&graph, op_names::INPUTS_OP)?;
        let seq_lens_op = Self::add_op(&graph, op_names::SEQ_LENS_OP)?;

        let loss_op = Self::add_op(&graph, op_names::LOSS_OP)?;
        let accuracy_op = Self::add_op(&graph, op_names::ACCURACY_OP)?;
        let labels_op = Self::add_op(&graph, op_names::LABELS_OP)?;
        let top_k_predicted_op = Self::add_op(&graph, op_names::TOP_K_PREDICTED_OP)?;
        let top_k_probs_op = Self::add_op(&graph, op_names::TOP_K_PROBS_OP)?;

        let train_op = Self::add_op(&graph, op_names::TRAIN_OP)?;

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
            top_k_probs_op,

            train_op,
        })
    }

    fn add_op(graph: &Graph, name: &str) -> Result<Operation, Error> {
        graph
            .operation_by_name_required(name)
            .map_err(status_to_error)
    }
}

pub struct Tagger<D>
where
    D: SentenceDecoder,
    D::Encoding: Eq + Hash,
{
    graph: TaggerGraph,
    decoder: CategoricalEncoder<D, D::Encoding>,
    session: Session,
    vectorizer: SentVectorizer,
}

impl<D> Tagger<D>
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
{
    /// Load a tagger with weights.
    ///
    /// This constructor will load the model parameters (such as weights) from
    /// the file specified in `parameters_path`.
    pub fn load_weights<P>(
        graph: TaggerGraph,
        decoder: CategoricalEncoder<D, D::Encoding>,
        vectorizer: SentVectorizer,
        parameters_path: P,
    ) -> Result<Self, Error>
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
            decoder,
            session,
            vectorizer,
        })
    }

    fn new_session(graph: &TaggerGraph) -> Result<Session, Error> {
        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(&graph.model_config.to_protobuf()?)
            .map_err(status_to_error)?;

        Session::new(&session_opts, &graph.graph).map_err(status_to_error)
    }

    fn prepare_batch(
        &self,
        sentences: &[impl Borrow<Sentence>],
    ) -> Result<TensorBuilder<NoLabels>, Error> {
        // Find maximum sentence size.
        let max_seq_len = sentences
            .iter()
            .map(|s| s.borrow().len() - 1)
            .max()
            .unwrap_or(0);

        let mut builder =
            TensorBuilder::new(sentences.len(), max_seq_len, self.vectorizer.input_len());

        // Fill the batch.
        for sentence in sentences {
            let input = self.vectorizer.realize(sentence.borrow())?;
            builder.add_without_labels(&input);
        }

        Ok(builder)
    }

    fn tag_sentences_(&self, sentences: &mut [impl BorrowMut<Sentence>]) -> Fallible<()> {
        let builder = self.prepare_batch(sentences)?;

        // Tag the batch
        let (tag_tensor, probs_tensor) = self.tag_sequences(
            builder.seq_lens(),
            builder.inputs(),
            &self.graph.top_k_predicted_op,
            &self.graph.top_k_probs_op,
        )?;

        // Decode label numbers.
        let max_seq_len = tag_tensor.dims()[1] as usize;
        let k = tag_tensor.dims()[2] as usize;
        for (idx, sentence) in sentences.iter_mut().map(BorrowMut::borrow_mut).enumerate() {
            // BorrowMut derives from borrow, but the borrowed type cannot
            // be inferred with a borrow() call here?
            let seq_len = min(max_seq_len, sentence.len() - 1);
            let offset = idx * max_seq_len * k;
            let seq = &tag_tensor[offset..offset + seq_len * k];
            let probs = &probs_tensor[offset..offset + seq_len * k];

            // Get the label numbers with their probabilities.
            let encoded = seq
                .iter()
                .zip(probs)
                .chunks(k)
                .into_iter()
                .map(|c| {
                    c.map(|(&label, &prob)| EncodingProb::new_from_owned(label as usize, prob))
                })
                .map(Vec::from_iter)
                .collect::<Vec<_>>();

            // And decode the labels.
            self.decoder.decode(&encoded, sentence)?;
        }

        Ok(())
    }

    fn tag_sequences(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
        predicted_op: &Operation,
        probs_op: &Operation,
    ) -> Result<(Tensor<i32>, Tensor<f32>), Error> {
        let mut is_training = Tensor::new(&[]);
        is_training[0] = false;

        let mut args = SessionRunArgs::new();

        args.add_feed(&self.graph.is_training_op, 0, &is_training);

        // Sequence inputs
        args.add_feed(&self.graph.seq_lens_op, 0, seq_lens.inner_ref());
        args.add_feed(&self.graph.inputs_op, 0, inputs.inner_ref());

        let probs_token = args.request_fetch(probs_op, 0);
        let predictions_token = args.request_fetch(predicted_op, 0);

        self.session.run(&mut args).map_err(status_to_error)?;

        Ok((
            args.fetch(predictions_token).map_err(status_to_error)?,
            args.fetch(probs_token).map_err(status_to_error)?,
        ))
    }
}

impl<D> Tag for Tagger<D>
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
{
    /// Tag sentences, returning the top-k results for every token.
    fn tag_sentences(&self, sentences: &mut [impl BorrowMut<Sentence>]) -> Fallible<()> {
        self.tag_sentences_(sentences)
    }
}
