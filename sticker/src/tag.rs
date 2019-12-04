use std::borrow::{Borrow, BorrowMut};

use conllx::graph::Sentence;
use failure::Fallible;
use sticker_encoders::{EncodingProb, SentenceDecoder};

/// Trait for sequence taggers.
pub trait Tag {
    /// Tag sentences.
    fn tag_sentences(&self, sentences: &mut [impl BorrowMut<Sentence>]) -> Fallible<()>;
}

pub type TopKLabels<'a, L> = Vec<Vec<Vec<L>>>;

/// Trait for predicting the top-k labels for tokens.
pub trait TopK<D>
where
    D: SentenceDecoder,
{
    /// Get the top-k labels for all tokens.
    ///
    /// *k* is fixed in the model graph.
    fn top_k(
        &self,
        sentences: &[impl Borrow<Sentence>],
    ) -> Fallible<TopKLabels<EncodingProb<D::Encoding>>>;
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

#[cfg(test)]
mod tests {
    use conllx::token::{Features, Token, TokenBuilder};
    use sticker_encoders::layer::{Layer, LayerValue};

    #[test]
    fn layer() {
        let token: Token = TokenBuilder::new("test")
            .cpos("CP")
            .pos("P")
            .features(Features::from("a:b|c:d"))
            .into();

        assert_eq!(token.value(&Layer::CPos), Some("CP"));
        assert_eq!(token.value(&Layer::Pos), Some("P"));
        assert_eq!(token.value(&Layer::Feature("a".to_owned())), Some("b"));
        assert_eq!(token.value(&Layer::Feature("c".to_owned())), Some("d"));
        assert_eq!(token.value(&Layer::Feature("e".to_owned())), None);
    }

    #[test]
    fn set_layer() {
        let mut token: Token = TokenBuilder::new("test").into();
        token.set_value(&Layer::CPos, "CP");
        token.set_value(&Layer::Pos, "P");
        token.set_value(&Layer::Feature("a".to_owned()), "b");

        assert_eq!(token.value(&Layer::CPos), Some("CP"));
        assert_eq!(token.value(&Layer::Pos), Some("P"));
        assert_eq!(token.value(&Layer::Feature("a".to_owned())), Some("b"));
        assert_eq!(token.value(&Layer::Feature("c".to_owned())), None);
    }
}
