use std::hash::Hash;

use conllx::graph::Sentence;
use failure::Error;

use crate::{Numberer, SentVectorizer, SentenceEncoder};

/// Data types collects (and typically stores) vectorized sentences.
pub trait Collector {
    fn collect(&mut self, sentence: &Sentence) -> Result<(), Error>;

    fn vectorizer(&self) -> &SentVectorizer;
}

/// Collector that does not store the vectorized sentences.
///
/// This collector can be used to construct lookup tables as a
/// side-effect of vectorizing the input.
pub struct NoopCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Eq + Hash,
{
    encoder: E,
    numberer: Numberer<E::Encoding>,
    vectorizer: SentVectorizer,
}

impl<E> NoopCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Eq + Hash,
{
    pub fn new(
        encoder: E,
        numberer: Numberer<E::Encoding>,
        vectorizer: SentVectorizer,
    ) -> NoopCollector<E> {
        NoopCollector {
            encoder,
            numberer,
            vectorizer,
        }
    }

    pub fn labels(&self) -> &Numberer<E::Encoding> {
        &self.numberer
    }
}

impl<E> Collector for NoopCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    fn collect(&mut self, sentence: &Sentence) -> Result<(), Error> {
        self.vectorizer.realize(sentence)?;

        for encoding in self.encoder.encode(sentence)? {
            self.numberer.add(encoding);
        }

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
