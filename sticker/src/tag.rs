use std::borrow::BorrowMut;

use conllx::graph::Sentence;
use conllx::token::{Features, Token};
use failure::Fallible;
use serde_derive::{Deserialize, Serialize};

/// Tagging layer.
#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Layer {
    CPos,
    Pos,
    Feature(String),
}

/// Layer values.
pub trait LayerValue {
    /// Set a layer value.
    fn set_value(&mut self, layer: &Layer, value: impl Into<String>);

    /// Get a layer value.
    fn value(&self, layer: &Layer) -> Option<&str>;
}

impl LayerValue for Token {
    /// Set the layer to the given value.
    fn set_value(&mut self, layer: &Layer, value: impl Into<String>) {
        let value = value.into();

        match layer {
            Layer::CPos => {
                self.set_cpos(Some(value));
            }
            Layer::Pos => {
                self.set_pos(Some(value));
            }
            Layer::Feature(ref feature) => {
                if self.features().is_none() {
                    self.set_features(Some(Features::default()));
                }

                self.features_mut()
                    .unwrap()
                    .insert(feature.clone(), Some(value));
            }
        };
    }

    /// Look up the layer value in a token.
    fn value(&self, layer: &Layer) -> Option<&str> {
        match layer {
            Layer::CPos => self.cpos(),
            Layer::Pos => self.pos(),
            Layer::Feature(ref feature) => {
                self.features()?.get(feature)?.as_ref().map(String::as_str)
            }
        }
    }
}

/// Trait for sequence taggers.
pub trait Tag {
    fn tag_sentences(&self, sentences: &mut [impl BorrowMut<Sentence>]) -> Fallible<()>;
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
    use crate::{Layer, LayerValue};

    use conllx::token::{Features, Token, TokenBuilder};

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
