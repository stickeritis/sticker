//! Label encoders.

use std::borrow::{Borrow, Cow};
use std::hash::Hash;

use conllx::graph::Sentence;
use failure::Error;

use crate::Numberer;

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

/// An encoder wrapper that encodes/decodes to a categorical label.
pub struct CategoricalEncoder<E, V>
where
    V: Eq + Hash,
{
    inner: E,
    numberer: Numberer<V>,
}

impl<E, V> CategoricalEncoder<E, V>
where
    V: Eq + Hash,
{
    pub fn new(encoder: E, numberer: Numberer<V>) -> Self {
        CategoricalEncoder {
            inner: encoder,
            numberer,
        }
    }
}

impl<E> SentenceEncoder for CategoricalEncoder<E, E::Encoding>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    type Encoding = usize;

    fn encode(&mut self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Error> {
        let encoding = self.inner.encode(sentence)?;
        let categorical_encoding = encoding.into_iter().map(|e| self.numberer.add(e)).collect();
        Ok(categorical_encoding)
    }
}

impl<D> SentenceDecoder for CategoricalEncoder<D, D::Encoding>
where
    D: SentenceDecoder,
    D::Encoding: Clone + Eq + Hash,
{
    type Encoding = usize;

    fn decode<'a, S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        S: AsRef<[EncodingProb<'a, Self::Encoding>]>,
    {
        let encoding = labels
            .iter()
            .map(|encoding_probs| {
                encoding_probs
                    .as_ref()
                    .iter()
                    .map(|encoding_prob| {
                        EncodingProb::new(
                            self.numberer
                                .value(*encoding_prob.encoding())
                                .expect("Unknown label"),
                            encoding_prob.prob(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        self.inner.decode(&encoding, sentence)
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    use conllx::io::Reader;

    use super::layer::LayerEncoder;
    use super::{CategoricalEncoder, EncodingProb, SentenceDecoder, SentenceEncoder};
    use crate::{Layer, Numberer};

    static NON_PROJECTIVE_DATA: &'static str = "testdata/nonprojective.conll";

    fn test_encoding<P, E, C>(path: P, mut encoder_decoder: E)
    where
        P: AsRef<Path>,
        E: SentenceEncoder<Encoding = C> + SentenceDecoder<Encoding = C>,
        C: 'static + Clone,
    {
        let f = File::open(path).unwrap();
        let reader = Reader::new(BufReader::new(f));

        for sentence in reader {
            let sentence = sentence.unwrap();

            // Encode
            let encodings = encoder_decoder
                .encode(&sentence)
                .unwrap()
                .into_iter()
                .map(|e| [EncodingProb::new_from_owned(e, 1.)])
                .collect::<Vec<_>>();

            // Decode
            let mut test_sentence = sentence.clone();
            encoder_decoder
                .decode(&encodings, &mut test_sentence)
                .unwrap();

            assert_eq!(sentence, test_sentence);
        }
    }

    #[test]
    fn categorical_encoder() {
        let numberer = Numberer::new(1);
        let encoder = LayerEncoder::new(Layer::Pos);
        let categorical_encoder = CategoricalEncoder::new(encoder, numberer);
        test_encoding(NON_PROJECTIVE_DATA, categorical_encoder);
    }
}
