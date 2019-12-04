use std::borrow::{Borrow, BorrowMut};
use std::cmp::min;
use std::hash::Hash;
use std::io::Read;
use std::iter::FromIterator;
use std::path::Path;

use conllx::graph::Sentence;
use failure::{err_msg, Error, Fallible};
use itertools::Itertools;
use ndarray::{Ix1, Ix2, Ix3};
use ndarray_tensorflow::NdTensor;
use serde_derive::{Deserialize, Serialize};
use sticker_encoders::categorical::ImmutableCategoricalEncoder;
use sticker_encoders::{EncodingProb, SentenceDecoder};
use tensorflow::{
    Graph, ImportGraphDefOptions, Operation, Session, SessionOptions, SessionRunArgs, Tensor,
};

use super::tensor::{NoLabels, TensorBuilder};
use super::util::{prepare_path, status_to_error, ConfigProtoBuilder};
use crate::{SentVectorizer, Tag, TopK, TopKLabels};

/// Graph metadata
///
/// This only contains the fields that we are interested in. Since we
/// do not deny unknown fields, these are silently ignored.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct Metadata {
    #[serde(default)]
    pub auto_mixed_precision: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// Model batch size (unused).
    pub batch_size: Option<usize>,

    /// Only allocate as much GPU memory as needed.
    #[serde(default)]
    pub gpu_allow_growth: bool,

    /// The filename of the Tensorflow graph.
    pub graph: String,

    /// Thread pool size for parallel processing within a computation
    /// graph op.
    pub intra_op_parallelism_threads: Option<usize>,

    /// Thread pool size for parallel processing of independent computation
    /// graph ops.
    pub inter_op_parallelism_threads: Option<usize>,

    /// The filename of the trained graph parameters.
    pub parameters: String,
}

/// Tensorflow runtime configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeConfig {
    /// Number of inter op parallelism threads.
    pub inter_op_threads: usize,

    /// Number of intra op parallelism threads.
    pub intra_op_threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            inter_op_threads: 4,
            intra_op_threads: 4,
        }
    }
}

impl ModelConfig {
    pub fn to_protobuf(
        &self,
        auto_mixed_precision: bool,
        runtime_config: &RuntimeConfig,
    ) -> Result<Vec<u8>, Error> {
        ConfigProtoBuilder::default()
            .inter_op_parallelism_threads(runtime_config.inter_op_threads)
            .intra_op_parallelism_threads(runtime_config.intra_op_threads)
            .gpu_allow_growth(self.gpu_allow_growth)
            .auto_mixed_precision(auto_mixed_precision)
            .protobuf()
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            batch_size: None,
            gpu_allow_growth: true,
            graph: String::new(),
            inter_op_parallelism_threads: None,
            intra_op_parallelism_threads: None,
            parameters: String::new(),
        }
    }
}

mod op_names {
    pub const GRAPH_METADATA_OP: &str = "graph_metadata";

    pub const INIT_OP: &str = "init";
    pub const RESTORE_OP: &str = "save/restore_all";
    pub const SAVE_OP: &str = "save/control_dependency";
    pub const SAVE_PATH_OP: &str = "save/Const";

    pub const IS_TRAINING_OP: &str = "model/is_training";
    pub const LR_OP: &str = "model/lr";

    pub const INPUTS_OP: &str = "model/inputs";
    pub const SUBWORDS_OP: &str = "model/subwords";
    pub const SEQ_LENS_OP: &str = "model/seq_lens";

    pub const LOSS_OP: &str = "model/tag_loss";
    pub const ACCURACY_OP: &str = "model/tag_accuracy";
    pub const LABELS_OP: &str = "model/tags";
    pub const TOP_K_PREDICTED_OP: &str = "model/tag_top_k_predictions";
    pub const TOP_K_PROBS_OP: &str = "model/tag_top_k_probs";

    pub const TRAIN_OP: &str = "model/train";
    pub const WRITE_GRAPH_OP: &str = "graph_write";
    pub const LOGDIR_OP: &str = "logdir";
    pub const SUMMARY_INIT_OP: &str = "summary_init";

    pub const VAL_SUMMARIES_OP: &str = "model/summaries/val";
    pub const TRAIN_SUMMARIES_OP: &str = "model/summaries/train";
}

pub struct TaggerGraph {
    pub(crate) graph: Graph,
    pub(crate) model_config: ModelConfig,

    pub(crate) graph_metadata_op: Option<Operation>,

    pub(crate) graph_write_op: Option<Operation>,
    pub(crate) logdir_op: Option<Operation>,
    pub(crate) summary_init_op: Option<Operation>,
    pub(crate) train_summary_op: Option<Operation>,
    pub(crate) val_summary_op: Option<Operation>,

    pub(crate) init_op: Operation,
    pub(crate) restore_op: Operation,
    pub(crate) save_op: Operation,
    pub(crate) save_path_op: Operation,
    pub(crate) lr_op: Operation,
    pub(crate) is_training_op: Operation,
    pub(crate) inputs_op: Operation,
    pub(crate) subwords_op: Option<Operation>,
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

        let graph_metadata_op = Self::add_op(&graph, op_names::GRAPH_METADATA_OP).ok();

        let restore_op = Self::add_op(&graph, op_names::RESTORE_OP)?;
        let save_op = Self::add_op(&graph, op_names::SAVE_OP)?;
        let save_path_op = Self::add_op(&graph, op_names::SAVE_PATH_OP)?;
        let init_op = Self::add_op(&graph, op_names::INIT_OP)?;

        let is_training_op = Self::add_op(&graph, op_names::IS_TRAINING_OP)?;
        let lr_op = Self::add_op(&graph, op_names::LR_OP)?;

        let inputs_op = Self::add_op(&graph, op_names::INPUTS_OP)?;
        let subwords_op = Self::add_op(&graph, op_names::SUBWORDS_OP).ok();
        let seq_lens_op = Self::add_op(&graph, op_names::SEQ_LENS_OP)?;

        let loss_op = Self::add_op(&graph, op_names::LOSS_OP)?;
        let accuracy_op = Self::add_op(&graph, op_names::ACCURACY_OP)?;
        let labels_op = Self::add_op(&graph, op_names::LABELS_OP)?;
        let top_k_predicted_op = Self::add_op(&graph, op_names::TOP_K_PREDICTED_OP)?;
        let top_k_probs_op = Self::add_op(&graph, op_names::TOP_K_PROBS_OP)?;

        let train_op = Self::add_op(&graph, op_names::TRAIN_OP)?;

        let graph_write_op = Self::add_op(&graph, op_names::WRITE_GRAPH_OP).ok();
        let logdir_op = Self::add_op(&graph, op_names::LOGDIR_OP).ok();
        let summary_init_op = Self::add_op(&graph, op_names::SUMMARY_INIT_OP).ok();

        let train_summary_op = Self::add_op(&graph, op_names::TRAIN_SUMMARIES_OP).ok();
        let val_summary_op = Self::add_op(&graph, op_names::VAL_SUMMARIES_OP).ok();

        Ok(TaggerGraph {
            graph,
            model_config: model_config.clone(),

            graph_metadata_op,

            graph_write_op,
            logdir_op,
            summary_init_op,

            train_summary_op,
            val_summary_op,

            init_op,
            restore_op,
            save_op,
            save_path_op,
            is_training_op,
            lr_op,
            inputs_op,
            subwords_op,
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

    pub(crate) fn has_auto_mixed_precision(&self) -> Fallible<bool> {
        let metadata_str = match self.metadata()? {
            Some(metadata) => metadata,
            None => return Ok(false),
        };

        let metadata: Metadata = toml::from_str(&metadata_str)?;

        Ok(metadata.auto_mixed_precision)
    }

    pub fn metadata(&self) -> Fallible<Option<String>> {
        let metadata = match self.graph_metadata_op {
            Some(ref graph_metadata_op) => {
                let mut args = SessionRunArgs::new();
                let metadata_token = args.request_fetch(graph_metadata_op, 0);
                let session = self.metadata_session()?;
                session.run(&mut args).map_err(status_to_error)?;
                let metadata: Tensor<String> =
                    args.fetch(metadata_token).map_err(status_to_error)?;
                Some(metadata[0].clone())
            }
            None => None,
        };

        Ok(metadata)
    }

    fn metadata_session(&self) -> Fallible<Session> {
        let config_proto = ConfigProtoBuilder::default().gpu_count(0).protobuf()?;

        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(&config_proto)
            .map_err(status_to_error)?;

        Session::new(&session_opts, &self.graph).map_err(status_to_error)
    }
}

pub struct Tagger<D>
where
    D: Send + SentenceDecoder + Sync,
    D::Encoding: Clone + Eq + Hash,
{
    graph: TaggerGraph,
    decoder: ImmutableCategoricalEncoder<D, D::Encoding>,
    session: Session,
    vectorizer: SentVectorizer,
}

impl<D> Tagger<D>
where
    D: Send + SentenceDecoder + Sync,
    D::Encoding: Clone + Eq + Hash,
{
    /// Load a tagger with weights.
    ///
    /// This constructor will load the model parameters (such as weights) from
    /// the file specified in `parameters_path`.
    pub fn load_weights<P>(
        graph: TaggerGraph,
        runtime_config: &RuntimeConfig,
        decoder: ImmutableCategoricalEncoder<D, D::Encoding>,
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
        let session = Self::new_session(&graph, runtime_config)?;
        session.run(&mut args).map_err(status_to_error)?;

        Ok(Tagger {
            graph,
            decoder,
            session,
            vectorizer,
        })
    }

    fn new_session(graph: &TaggerGraph, runtime_config: &RuntimeConfig) -> Result<Session, Error> {
        let mut session_opts = SessionOptions::new();
        session_opts
            .set_config(&graph.model_config.to_protobuf(false, runtime_config)?)
            .map_err(status_to_error)?;

        Session::new(&session_opts, &graph.graph).map_err(status_to_error)
    }

    /// Get the top-k numeric labels for the sequences.
    fn top_k_numeric_<'a, S>(
        &self,
        sentences: &'a [S],
    ) -> Fallible<impl 'a + Iterator<Item = Vec<Vec<EncodingProb<usize>>>>>
    where
        S: Borrow<Sentence>,
    {
        let builder = self.prepare_batch(sentences)?;

        // Tag the batch
        let (tag_tensor, probs_tensor) = self.tag_sequences(
            builder.seq_lens(),
            builder.inputs(),
            builder.subwords(),
            &self.graph.top_k_predicted_op,
            &self.graph.top_k_probs_op,
        )?;

        // Decode label numbers.
        let max_seq_len = tag_tensor.dims()[1] as usize;
        let k = tag_tensor.dims()[2] as usize;

        Ok(sentences
            .iter()
            .map(Borrow::borrow)
            .enumerate()
            .map(move |(idx, sentence)| {
                // BorrowMut derives from borrow, but the borrowed type cannot
                // be inferred with a borrow() call here?
                let seq_len = min(max_seq_len, sentence.len() - 1);
                let offset = idx * max_seq_len * k;
                let seq = &tag_tensor[offset..offset + seq_len * k];
                let probs = &probs_tensor[offset..offset + seq_len * k];

                // Get the label numbers with their probabilities.
                seq.iter()
                    .zip(probs)
                    .chunks(k)
                    .into_iter()
                    .map(|c| c.map(|(&label, &prob)| EncodingProb::new(label as usize, prob)))
                    .map(Vec::from_iter)
                    .collect::<Vec<_>>()
            }))
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

        let mut builder = TensorBuilder::new(
            sentences.len(),
            max_seq_len,
            self.vectorizer.input_len(),
            self.vectorizer.has_subwords(),
        );

        // Fill the batch.
        for sentence in sentences {
            let input = self.vectorizer.realize(sentence.borrow())?;
            builder.add_without_labels(input);
        }

        Ok(builder)
    }

    fn tag_sequences(
        &self,
        seq_lens: &NdTensor<i32, Ix1>,
        inputs: &NdTensor<f32, Ix3>,
        subwords: Option<&NdTensor<String, Ix2>>,
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

        if let Some(subwords) = subwords {
            args.add_feed(
                self.graph.subwords_op.as_ref().ok_or_else(|| {
                    err_msg("Subwords used in a graph without support for subwords")
                })?,
                0,
                subwords.inner_ref(),
            );
        }

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
    D: Send + SentenceDecoder + Sync,
    D::Encoding: Clone + Eq + Hash,
{
    fn tag_sentences(&self, sentences: &mut [impl BorrowMut<Sentence>]) -> Fallible<()> {
        // We have to collect the results into a Vec, otherwise we
        // simultaneously have an immutable and a mutable borrow of
        // sentences.
        let top_k_numeric = self.top_k_numeric_(sentences)?.collect::<Vec<_>>();

        for (top_k, sentence) in top_k_numeric.into_iter().zip(sentences.iter_mut()) {
            self.decoder.decode(&top_k, sentence.borrow_mut())?;
        }

        Ok(())
    }
}

impl<D> TopK<D> for Tagger<D>
where
    D: Send + SentenceDecoder + Sync,
    D::Encoding: Clone + Eq + Hash,
{
    fn top_k(
        &self,
        sentences: &[impl Borrow<Sentence>],
    ) -> Fallible<TopKLabels<EncodingProb<D::Encoding>>> {
        self.top_k_numeric_(sentences)?
            .map(|top_k| self.decoder.decode_without_inner(&top_k))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufReader, Cursor, Read};
    use std::path::Path;

    use flate2::read::GzDecoder;

    use super::TaggerGraph;

    fn load_graph(path: impl AsRef<Path>) {
        let f = File::open(path).expect("Cannot open test graph.");
        let mut decoder = GzDecoder::new(BufReader::new(f));
        let mut data = Vec::new();
        decoder
            .read_to_end(&mut data)
            .expect("Cannot decompress test graph.");

        let model_config = Default::default();
        TaggerGraph::load_graph(Cursor::new(data), &model_config).expect("Cannot load graph");
    }

    #[test]
    fn load_sticker_0_4_conv_graph() {
        load_graph("testdata/sticker-0.4-conv.graph.gz");
    }

    #[test]
    fn load_sticker_0_4_rnn_graph() {
        load_graph("testdata/sticker-0.4-rnn.graph.gz");
    }

    #[test]
    fn load_sticker_0_4_transformer_graph() {
        load_graph("testdata/sticker-0.4-transformer.graph.gz");
    }
}
