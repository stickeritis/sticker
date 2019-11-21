use std::fs::File;
use std::hash::Hash;

use conllx::graph::Sentence;
use failure::{Fallible, ResultExt};

use crate::encoder::categorical::CategoricalEncoder;
use crate::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use crate::encoder::layer::LayerEncoder;
use crate::encoder::lemma::EditTreeEncoder;
use crate::encoder::SentenceDecoder;
use crate::serialization::CborRead;
use crate::tensorflow::{RuntimeConfig, Tagger as TFTagger, TaggerGraph};
use crate::wrapper::{Config, EncoderType, LabelerType};
use crate::{Numberer, SentVectorizer, Tag, TopK, TopKLabels};

/// The `Tag` trait is not object-safe, since the `tag_sentences`
/// method has a type parameter to accept a slice of mutably
/// borrowable `Sentence`s. However, in the `Tagger` wrapper, we want
/// to box different taggers to hide their type parameters. `TagRef`
/// is a helper trait that is implemented for any `Tag`, but is
/// object-safe by specializing `tag_sentences_ref` for `Sentence`
/// references.
trait TagRef {
    fn tag_sentences_ref(&self, sentences: &mut [&mut Sentence]) -> Fallible<()>;
}

impl<T> TagRef for T
where
    T: Tag,
{
    /// Tag sentences.
    fn tag_sentences_ref(&self, sentences: &mut [&mut Sentence]) -> Fallible<()> {
        self.tag_sentences(sentences)
    }
}

/// The `TopK` trait is not object-safe, since the `top_k` method has
/// a type parameter to accept a slice of mutably borrowable
/// `Sentence`s. However, in the `Tagger` wrapper, we want to box
/// different taggers to hide their type parameters. `TopKRef` is a
/// helper trait that is implemented for any `TopK`, but is
/// object-safe by specializing `top_k_ref` for `Sentence` references.
trait TopKRef {
    /// Get the top-k labels for all tokens.
    ///
    /// *k* is fixed in the model graph.
    fn top_k_ref(&self, sentences: &[&Sentence]) -> Fallible<TopKLabels<(String, f32)>>;
}

impl<D> TopKRef for TFTagger<D>
where
    TFTagger<D>: TopK<D>,
    D: Send + SentenceDecoder + Sync,
    D::Encoding: Clone + Eq + Hash + ToString,
{
    fn top_k_ref(&self, sentences: &[&Sentence]) -> Fallible<TopKLabels<(String, f32)>> {
        let top_k = self.top_k(sentences)?;

        // vec.into() does not seem to work? meh!
        Ok(top_k
            .into_iter()
            .map(|s| {
                s.into_iter()
                    .map(|t| t.into_iter().map(Into::into).collect())
                    .collect()
            })
            .collect())
    }
}

/// Auxiliary trait that bundles the `TagRef` and `TopKRef` traits.
trait TaggerRef: TagRef + TopKRef {}

impl<T> TaggerRef for T where T: TagRef + TopKRef {}

/// A convenience wrapper for the `tensorflow::Tagger`.
///
/// This wrapper loads embeddings, initializes the vectorizer,
/// initializes the graph, and loads labels given a `Config` struct.
pub struct Tagger {
    inner: Box<dyn TaggerRef + Send + Sync>,
}

impl Tagger {
    /// Create a tagger from the given configuration.
    pub fn new(config: &Config, runtime_config: &RuntimeConfig) -> Fallible<Self> {
        let embeddings = config
            .input
            .embeddings
            .load_embeddings()
            .with_context(|e| format!("Cannot load embeddings: {}", e))?;
        let vectorizer = SentVectorizer::new(embeddings, config.input.subwords);

        let graph_reader = File::open(&config.model.graph).with_context(|e| {
            format!(
                "Cannot open computation graph '{}' for reading: {}",
                &config.model.graph, e
            )
        })?;

        let graph = TaggerGraph::load_graph(graph_reader, &config.model)
            .with_context(|e| format!("Cannot load computation graph: {}", e))?;

        match config.labeler.labeler_type {
            LabelerType::Lemma => {
                Self::new_with_decoder(&config, runtime_config, vectorizer, graph, EditTreeEncoder)
            }
            LabelerType::Sequence(ref layer) => Self::new_with_decoder(
                &config,
                runtime_config,
                vectorizer,
                graph,
                LayerEncoder::new(layer.clone()),
            ),
            LabelerType::Parser(EncoderType::RelativePOS) => Self::new_with_decoder(
                &config,
                runtime_config,
                vectorizer,
                graph,
                RelativePOSEncoder,
            ),
            LabelerType::Parser(EncoderType::RelativePosition) => Self::new_with_decoder(
                &config,
                runtime_config,
                vectorizer,
                graph,
                RelativePositionEncoder,
            ),
        }
    }

    fn new_with_decoder<D>(
        config: &Config,
        runtime_config: &RuntimeConfig,
        vectorizer: SentVectorizer,
        graph: TaggerGraph,
        decoder: D,
    ) -> Fallible<Self>
    where
        D: 'static + Send + SentenceDecoder + Sync,
        D::Encoding: Clone + Eq + Hash + Send + Sync + ToString,
        Numberer<D::Encoding>: CborRead,
    {
        let labels = config.labeler.load_labels().with_context(|e| {
            format!("Cannot load label file '{}': {}", config.labeler.labels, e)
        })?;

        let categorical_decoder = CategoricalEncoder::new(decoder, labels);

        let tagger = TFTagger::load_weights(
            graph,
            runtime_config,
            categorical_decoder,
            vectorizer,
            &config.model.parameters,
        )
        .with_context(|e| format!("Cannot construct tagger: {}", e))?;

        Ok(Tagger {
            inner: Box::new(tagger),
        })
    }

    /// Tag sentences.
    pub fn tag_sentences(&self, sentences: &mut [&mut Sentence]) -> Fallible<()> {
        self.inner.tag_sentences_ref(sentences)
    }

    /// Get the top-k labels for all tokens.
    ///
    /// *k* is fixed in the model graph.
    pub fn top_k(&self, sentences: &[&Sentence]) -> Fallible<TopKLabels<(String, f32)>> {
        self.inner.top_k_ref(sentences)
    }
}
