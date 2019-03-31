use conllx::Token;
use failure::Error;
use finalfusion::{
    embeddings::Embeddings as FiFuEmbeddings,
    storage::StorageWrap,
    storage::{CowArray, CowArray1},
    vocab::VocabWrap,
};
use ndarray::Array1;

pub struct Embeddings {
    embeddings: FiFuEmbeddings<VocabWrap, StorageWrap>,
    unknown: Array1<f32>,
}

impl Embeddings {
    pub fn dims(&self) -> usize {
        self.embeddings.dims()
    }

    pub fn embedding(&self, word: &str) -> CowArray1<f32> {
        self.embeddings
            .embedding(word)
            .unwrap_or_else(|| CowArray::Borrowed(self.unknown.view()))
    }
}

impl From<FiFuEmbeddings<VocabWrap, StorageWrap>> for Embeddings {
    fn from(embeddings: FiFuEmbeddings<VocabWrap, StorageWrap>) -> Self {
        let mut unknown = Array1::zeros(embeddings.dims());

        for (_, embed) in &embeddings {
            unknown += &embed.as_view();
        }

        let l2norm = unknown.dot(&unknown).sqrt();
        if l2norm != 0f32 {
            unknown /= l2norm;
        }

        Embeddings {
            embeddings,
            unknown,
        }
    }
}

/// Sentence represented as a vector.
///
/// This data type represents a sentence as vectors (`Vec`) of tokens
/// and tag indices. Such a vector is typically the input to a
/// sequence labeling graph.
#[derive(Default)]
pub struct SentVec {
    pub tokens: Vec<f32>,
}

impl SentVec {
    /// Construct a new sentence vector.
    pub fn new() -> Self {
        SentVec::default()
    }

    /// Construct a sentence vector with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        SentVec {
            tokens: Vec::with_capacity(capacity),
        }
    }

    /// Get the embedding representation of a sentence.
    ///
    /// The vector contains the concatenation of the embeddings of the
    /// tokens in the sentence.
    pub fn into_inner(self) -> Vec<f32> {
        self.tokens
    }
}

/// Embeddings for annotation layers.
///
/// This data structure bundles embedding matrices for the input
/// annotation layers: tokens and part-of-speech.
pub struct LayerEmbeddings {
    token_embeddings: Embeddings,
}

impl LayerEmbeddings {
    /// Construct `LayerEmbeddings` from the given embeddings.
    pub fn new(token_embeddings: Embeddings) -> Self {
        LayerEmbeddings { token_embeddings }
    }

    /// Get the token embedding matrix.
    pub fn token_embeddings(&self) -> &Embeddings {
        &self.token_embeddings
    }
}

/// Vectorizer for sentences.
///
/// An `SentVectorizer` vectorizes sentences.
pub struct SentVectorizer {
    layer_embeddings: LayerEmbeddings,
}

impl SentVectorizer {
    /// Construct an input vectorizer.
    ///
    /// The vectorizer is constructed from the embedding matrices. The layer
    /// embeddings are used to find the indices into the embedding matrix for
    /// layer values.
    pub fn new(layer_embeddings: LayerEmbeddings) -> Self {
        SentVectorizer { layer_embeddings }
    }

    /// Get the layer embeddings.
    pub fn layer_embeddings(&self) -> &LayerEmbeddings {
        &self.layer_embeddings
    }

    /// Vectorize a sentence.
    pub fn realize(&self, sentence: &[Token]) -> Result<SentVec, Error> {
        let mut input = SentVec::with_capacity(sentence.len());

        for token in sentence {
            let form = token.form();

            input.tokens.extend_from_slice(
                &self
                    .layer_embeddings
                    .token_embeddings
                    .embedding(form)
                    .as_view()
                    .as_slice()
                    .expect("Non-contiguous embedding"),
            );
        }

        Ok(input)
    }
}
