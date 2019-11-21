use std::fmt;

use conllx::graph::{Node, Sentence};
use edit_tree::{Apply, TreeNode};
use failure::{Error, Fail};
use serde_derive::{Deserialize, Serialize};

use super::{EncodingProb, SentenceDecoder, SentenceEncoder};

/// Lemma encoding error.
#[derive(Clone, Debug, Eq, Fail, PartialEq)]
pub enum EncodeError {
    /// The token does not have a lemma.
    #[fail(display = "token without a lemma: '{}'", form)]
    MissingLemma { form: String },
}

/// Encoding of a lemmatization as an edit tree.
#[derive(Clone, Deserialize, Debug, Eq, Hash, PartialEq, Serialize)]
pub struct EditTree {
    inner: TreeNode<char>,
}

impl EditTree {
    /// Pretty print an edit tree.
    ///
    /// This is a pretty printer for edit trees that converts them to
    /// an S-expr. It is not optimized for efficieny and does a lot of
    /// string allocations.
    fn pretty_print(node: &TreeNode<char>) -> String {
        match node {
            TreeNode::MatchNode {
                pre,
                suf,
                left,
                right,
            } => {
                let left_str = left
                    .as_ref()
                    .map(|left| Self::pretty_print(left))
                    .unwrap_or_else(|| "()".to_string());
                let right_str = right
                    .as_ref()
                    .map(|right| Self::pretty_print(right))
                    .unwrap_or_else(|| "()".to_string());

                format!("(match {} {} {} {})", pre, suf, left_str, right_str)
            }
            TreeNode::ReplaceNode {
                replacee,
                replacement,
            } => format!(
                "(replace \"{}\" \"{}\")",
                replacee.iter().collect::<String>(),
                replacement.iter().collect::<String>(),
            ),
        }
    }
}

impl fmt::Display for EditTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", Self::pretty_print(&self.inner))
    }
}

/// Edit tree-based lemma encoder.
///
/// This encoder encodes a lemma as an edit tree that is applied to an
/// unlemmatized form.
pub struct EditTreeEncoder;

impl SentenceDecoder for EditTreeEncoder {
    type Encoding = EditTree;

    fn decode<S>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        S: AsRef<[EncodingProb<Self::Encoding>]>,
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
                let form = token.form().chars().collect::<Vec<_>>();
                let lemma = match label.encoding().inner.apply(&form) {
                    Some(lemma) => lemma.into_iter().collect::<String>(),
                    None => continue,
                };

                token.set_lemma(Some(lemma));
            }
        }

        Ok(())
    }
}

impl SentenceEncoder for EditTreeEncoder {
    type Encoding = EditTree;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Error> {
        let mut encoding = Vec::with_capacity(sentence.len() - 1);

        for token in sentence.iter().filter_map(Node::token) {
            let lemma = token.lemma().ok_or_else(|| EncodeError::MissingLemma {
                form: token.form().to_owned(),
            })?;

            let tree = TreeNode::create_tree(
                &token.form().chars().collect::<Vec<_>>(),
                &lemma.chars().collect::<Vec<_>>(),
            );

            encoding.push(EditTree { inner: tree });
        }

        Ok(encoding)
    }
}

#[cfg(test)]
mod tests {
    use conllx::graph::Sentence;
    use conllx::token::{Token, TokenBuilder};
    use edit_tree::TreeNode;

    use super::{EditTree, EditTreeEncoder};
    use crate::encoder::{EncodingProb, SentenceDecoder, SentenceEncoder};

    #[test]
    fn display_edit_tree() {
        let tree = EditTree {
            inner: TreeNode::create_tree(&['l', 'o', 'o', 'p', 't'], &['l', 'o', 'p', 'e', 'n']),
        };

        assert_eq!(
            tree.to_string(),
            "(match 0 3 () (match 1 1 (replace \"o\" \"\") (replace \"t\" \"en\")))"
        );
    }

    #[test]
    fn encoder_decoder_roundtrip() {
        let mut sent_encode = Sentence::new();
        sent_encode.push(TokenBuilder::new("hij").lemma("hij").into());
        sent_encode.push(TokenBuilder::new("heeft").lemma("hebben").into());
        sent_encode.push(TokenBuilder::new("gefietst").lemma("fietsen").into());

        let encoder = EditTreeEncoder;
        let labels = encoder
            .encode(&sent_encode)
            .unwrap()
            .into_iter()
            .map(|encoding| vec![EncodingProb::new(encoding, 1.0)])
            .collect::<Vec<_>>();

        let mut sent_decode = Sentence::new();
        sent_decode.push(Token::new("hij"));
        sent_decode.push(Token::new("heeft"));
        sent_decode.push(Token::new("gefietst"));

        encoder.decode(&labels, &mut sent_decode).unwrap();
    }
}
