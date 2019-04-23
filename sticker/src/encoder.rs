use std::borrow::{Borrow, Cow};

use conllx::graph::{Node, Sentence};
use failure::{format_err, Error};

use crate::{Layer, LayerValue};

/// An encoding with its probability.
pub struct EncodingProb<'a, E>
where
    E: ToOwned,
{
    encoding: Cow<'a, E>,
    prob: f32,
}

impl<E> EncodingProb<'static, E>
where
    E: ToOwned,
{
    /// Create an encoding with its probability.
    ///
    /// This constructor takes an owned encoding.
    #[allow(dead_code)]
    pub(crate) fn new_from_owned(encoding: E::Owned, prob: f32) -> Self {
        EncodingProb {
            encoding: Cow::Owned(encoding),
            prob,
        }
    }
}

impl<'a, E> EncodingProb<'a, E>
where
    E: ToOwned,
{
    /// Create an encoding with its probability.
    ///
    /// This constructor takes a borrowed encoding.
    pub(crate) fn new(encoding: &'a E, prob: f32) -> Self {
        EncodingProb {
            encoding: Cow::Borrowed(encoding),
            prob,
        }
    }

    /// Get the encoding.
    pub fn encoding(&self) -> &E {
        self.encoding.borrow()
    }

    /// Get the probability of the encoding.
    pub fn prob(&self) -> f32 {
        self.prob
    }
}

/// Trait for sentence decoders.
///
/// A sentence decoder adds a representation to each token in a
/// sentence, such as a part-of-speech tag or a topological field.
pub trait SentenceDecoder {
    type Encoding: ToOwned;

    fn decode<'a, S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        S: AsRef<[EncodingProb<'a, Self::Encoding>]>,
        Self::Encoding: 'a;
}

/// Trait for sentence encoders.
///
/// A sentence encoder extracts a representation of each token in a
/// sentence, such as a part-of-speech tag or a topological field.
pub trait SentenceEncoder {
    type Encoding;

    /// Encode the given sentence.
    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Error>;
}

/// Encode sentences using a CoNLL-X layer.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerEncoder {
    layer: Layer,
}

impl LayerEncoder {
    /// Construct a new layer encoder of the given layer.
    pub fn new(layer: Layer) -> Self {
        LayerEncoder { layer }
    }
}

impl SentenceDecoder for LayerEncoder {
    type Encoding = String;

    fn decode<'a, S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        S: AsRef<[EncodingProb<'a, Self::Encoding>]>,
        Self::Encoding: 'a,
    {
        assert_eq!(
            labels.len(),
            sentence.len() - 1,
            "Labels and sentence length mismatch"
        );

        for (token, token_labels) in sentence
            .iter_mut()
            .filter_map(Node::token_mut)
            .zip(labels.iter())
        {
            if let Some(label) = token_labels.as_ref().get(0) {
                token.set_value(&self.layer, label.encoding().as_str());
            }
        }

        Ok(())
    }
}

impl SentenceEncoder for LayerEncoder {
    type Encoding = String;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Error> {
        let mut encoding = Vec::with_capacity(sentence.len() - 1);
        for token in sentence.iter().filter_map(Node::token) {
            let label = token
                .value(&self.layer)
                .ok_or_else(|| format_err!("Token without a label: {}", token.form()))?;
            encoding.push(label.to_owned());
        }

        Ok(encoding)
    }
}
