use conllx::Sentence;

use failure::Error;

/// Trait for part-of-speech taggers.
pub trait Tag {
    fn tag_sentences(&self, sentences: &[impl AsRef<Sentence>]) -> Result<Vec<Vec<&str>>, Error>;
}

/// Results of validation.
#[derive(Clone, Copy, Debug)]
pub struct ModelPerformance {
    /// Model loss.
    pub loss: f32,

    /// Model accuracy
    ///
    /// The accuracy is the fraction of correctly predicted transitions.
    pub accuracy: f32,
}
