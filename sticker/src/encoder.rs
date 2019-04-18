use std::borrow::Borrow;

use conllx::graph::{Node, Sentence};
use failure::{format_err, Error};

use crate::{Layer, LayerValue};

/// Trait for sentence decoders.
///
/// A sentence decoder adds a representation to each token in a
/// sentence, such as a part-of-speech tag or a topological field.
pub trait SentenceDecoder {
    type Encoding;

    fn decode<S, E>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        E: Borrow<Self::Encoding>,
        S: AsRef<[E]>;
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

    fn decode<S, E>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        E: Borrow<Self::Encoding>,
        S: AsRef<[E]>,
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
                token.set_value(&self.layer, label.borrow().as_str());
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
