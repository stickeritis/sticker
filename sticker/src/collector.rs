use conllx::Token;

use crate::{Layer, LayerValue, Numberer, SentVectorizer};
use failure::{format_err, Error};

/// Data types collects (and typically stores) vectorized sentences.
pub trait Collector {
    fn collect(&mut self, sentence: &[Token]) -> Result<(), Error>;

    fn vectorizer(&self) -> &SentVectorizer;
}

/// Collector that does not store the vectorized sentences.
///
/// This collector can be used to construct lookup tables as a
/// side-effect of vectorizing the input.
pub struct NoopCollector {
    layer: Layer,
    numberer: Numberer<String>,
    vectorizer: SentVectorizer,
}

impl NoopCollector {
    pub fn new(
        layer: Layer,
        numberer: Numberer<String>,
        vectorizer: SentVectorizer,
    ) -> NoopCollector {
        NoopCollector {
            layer,
            numberer,
            vectorizer,
        }
    }

    pub fn labels(&self) -> &Numberer<String> {
        &self.numberer
    }
}

impl Collector for NoopCollector {
    fn collect(&mut self, sentence: &[Token]) -> Result<(), Error> {
        self.vectorizer.realize(sentence)?;

        for token in sentence {
            let label = token
                .value(&self.layer)
                .ok_or_else(|| format_err!("Token without a tag: {}", token))?;
            self.numberer.add(label.to_owned());
        }

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
