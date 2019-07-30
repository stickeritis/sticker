use conllx::graph::{Node, Sentence};
use failure::{format_err, Error};
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

/// Embeddings for annotation layers.
///
/// This data structure bundles embedding matrices for the input
/// annotation layers: tokens and part-of-speech.
pub struct LayerEmbeddings {
    token_embeddings: Embeddings,
    tag_embeddings: Option<Embeddings>,
}

impl LayerEmbeddings {
    /// Construct `LayerEmbeddings` from the given embeddings.
    pub fn new(token_embeddings: Embeddings, tag_embeddings: Option<Embeddings>) -> Self {
        LayerEmbeddings {
            token_embeddings,
            tag_embeddings,
        }
    }

    /// Get the tag embedding matrix.
    pub fn tag_embeddings(&self) -> Option<&Embeddings> {
        self.tag_embeddings.as_ref()
    }

    /// Get the token embedding matrix.
    pub fn token_embeddings(&self) -> &Embeddings {
        &self.token_embeddings
    }
}

pub struct InputVector {
    pub sequence: Vec<f32>,
    pub subwords: Option<Vec<String>>,
}

/// Vectorizer for sentences.
///
/// An `SentVectorizer` vectorizes sentences.
pub struct SentVectorizer {
    layer_embeddings: LayerEmbeddings,
    subwords: bool,
}

impl SentVectorizer {
    /// Construct an input vectorizer.
    ///
    /// The vectorizer is constructed from the embedding matrices. The layer
    /// embeddings are used to find the indices into the embedding matrix for
    /// layer values.
    pub fn new(layer_embeddings: LayerEmbeddings, subwords: bool) -> Self {
        SentVectorizer {
            layer_embeddings,
            subwords,
        }
    }

    /// Does the vectorizer produce representations for subwords?
    pub fn has_subwords(&self) -> bool {
        self.subwords
    }

    /// Get the length of the input representation.
    pub fn input_len(&self) -> usize {
        self.layer_embeddings.token_embeddings().dims()
            + self
                .layer_embeddings
                .tag_embeddings()
                .as_ref()
                .map(|e| e.dims())
                .unwrap_or_default()
    }

    /// Get the layer embeddings.
    pub fn layer_embeddings(&self) -> &LayerEmbeddings {
        &self.layer_embeddings
    }

    /// Vectorize a sentence.
    pub fn realize(&self, sentence: &Sentence) -> Result<InputVector, Error> {
        let input_size = self.layer_embeddings.token_embeddings.dims()
            + self
                .layer_embeddings
                .tag_embeddings
                .as_ref()
                .map(Embeddings::dims)
                .unwrap_or_default();
        let mut input = Vec::with_capacity(sentence.len() * input_size);

        let mut subwords = if self.subwords {
            Some(Vec::with_capacity(sentence.len()))
        } else {
            None
        };

        for token in sentence.iter().filter_map(Node::token) {
            let form = token.form();

            if let Some(ref mut subwords) = subwords {
                subwords.push(form.to_owned());
            }

            input.extend_from_slice(
                &self
                    .layer_embeddings
                    .token_embeddings
                    .embedding(form)
                    .as_view()
                    .as_slice()
                    .expect("Non-contiguous embedding"),
            );

            if let Some(tag_embeddings) = &self.layer_embeddings.tag_embeddings {
                let pos_tag = token
                    .pos()
                    .ok_or_else(|| format_err!("Token without a tag: {}", token.form()))?;

                input.extend_from_slice(
                    tag_embeddings
                        .embedding(pos_tag)
                        .as_view()
                        .as_slice()
                        .expect("Non-contiguous embedding"),
                );
            }
        }

        Ok(InputVector {
            sequence: input,
            subwords,
        })
    }
}
