use conllx::Token;

use crate::{Numberer, SentVectorizer};
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
    numberer: Numberer<String>,
    vectorizer: SentVectorizer,
}

impl NoopCollector {
    pub fn new(numberer: Numberer<String>, vectorizer: SentVectorizer) -> NoopCollector {
        NoopCollector {
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
            let pos_tag = token
                .pos()
                .ok_or_else(|| format_err!("Token without a part-of-speech tag: {}", token))?;
            self.numberer.add(pos_tag.to_owned());
        }

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
