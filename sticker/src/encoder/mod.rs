//! Label encoders.

use std::borrow::{Borrow, Cow};

use conllx::graph::Sentence;
use failure::Error;

pub mod categorical;

pub mod deprel;

pub mod layer;

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

impl<'a, E> From<EncodingProb<'a, E>> for (String, f32)
where
    E: Clone + ToString,
{
    fn from(prob: EncodingProb<E>) -> Self {
        (prob.encoding().to_string(), prob.prob())
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
    fn encode(&mut self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Error>;
}
