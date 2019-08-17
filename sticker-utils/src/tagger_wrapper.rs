use std::fs::File;
use std::hash::Hash;

use conllx::graph::Sentence;
use failure::{Fallible, ResultExt};

use crate::config::{Config, EncoderType, LabelerType};
use crate::serialization::CborRead;
use sticker::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::encoder::layer::LayerEncoder;
use sticker::encoder::{CategoricalEncoder, SentenceDecoder};
use sticker::tensorflow::{Tagger, TaggerGraph};
use sticker::Tag;
use sticker::{Numberer, SentVectorizer};

/// The `Tag` trait is not object-safe, since the `tag_sentences`
/// method has a type parameter to accept a slice of mutably
/// borrowable `Sentence`s. However, in TaggerWrapper, we want to box
/// different taggers to hide their type parameters. `TagRef` is a
/// helper trait that is implemented for any `Tag`, but is object-safe
/// by specializing `tag_sentences_ref` for `Sentence` references.
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

/// A covenience wrapper for `Tagger`.
///
/// This wrapper loads embeddings, initializes the vectorizer,
/// initializes the graph, and loads labels given a `Config` struct.
pub struct TaggerWrapper {
    inner: Box<dyn TagRef + Send + Sync>,
}

impl TaggerWrapper {
    /// Create a tagger from the given configuration.
    pub fn new(config: &Config) -> Fallible<Self> {
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
            LabelerType::Sequence(ref layer) => {
                Self::new_with_decoder(&config, vectorizer, graph, LayerEncoder::new(layer.clone()))
            }
            LabelerType::Parser(EncoderType::RelativePOS) => {
                Self::new_with_decoder(&config, vectorizer, graph, RelativePOSEncoder)
            }
            LabelerType::Parser(EncoderType::RelativePosition) => {
                Self::new_with_decoder(&config, vectorizer, graph, RelativePositionEncoder)
            }
        }
    }

    fn new_with_decoder<D>(
        config: &Config,
        vectorizer: SentVectorizer,
        graph: TaggerGraph,
        decoder: D,
    ) -> Fallible<Self>
    where
        D: 'static + Send + SentenceDecoder + Sync,
        D::Encoding: Clone + Eq + Hash + Send + Sync,
        Numberer<D::Encoding>: CborRead,
    {
        let labels = config.labeler.load_labels().with_context(|e| {
            format!("Cannot load label file '{}': {}", config.labeler.labels, e)
        })?;

        let categorical_decoder = CategoricalEncoder::new(decoder, labels);

        let tagger = Tagger::load_weights(
            graph,
            categorical_decoder,
            vectorizer,
            &config.model.parameters,
        )
        .with_context(|e| format!("Cannot construct tagger: {}", e))?;

        Ok(TaggerWrapper {
            inner: Box::new(tagger),
        })
    }

    /// Tag sentences.
    pub fn tag_sentences(&self, sentences: &mut [&mut Sentence]) -> Fallible<()> {
        self.inner.tag_sentences_ref(sentences)
    }
}
