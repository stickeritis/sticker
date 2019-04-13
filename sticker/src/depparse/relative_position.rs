use std::borrow::Borrow;

use conllx::graph::{DepTriple, Sentence};
use failure::Error;
use serde_derive::{Deserialize, Serialize};

use super::{DecodeError, DependencyEncoding, EncodeError};
use crate::{SentenceDecoder, SentenceEncoder, SentenceTopKDecoder};

/// Relative head position.
///
/// The position of the head relative to the dependent token.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct RelativePosition(isize);

impl ToString for DependencyEncoding<RelativePosition> {
    fn to_string(&self) -> String {
        format!("{}/{}", self.label, self.head.0)
    }
}

/// Relative position encoder.
///
/// This encoder encodes dependency relations as token labels. The
/// dependency relation is encoded as-is. The position of the head
/// is encoded relative to the (dependent) token.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RelativePositionEncoder;

impl RelativePositionEncoder {
    fn decode_idx(
        idx: usize,
        sentence_len: usize,
        encoding: &DependencyEncoding<RelativePosition>,
    ) -> Result<DepTriple<String>, DecodeError> {
        let DependencyEncoding {
            label,
            head: RelativePosition(head),
        } = encoding;

        let head_idx = idx as isize + head;
        if head_idx < 0 || head_idx >= sentence_len as isize {
            return Err(DecodeError::PositionOutOfBounds);
        }

        Ok(DepTriple::new(
            (idx as isize + head) as usize,
            Some(label.clone()),
            idx,
        ))
    }
}

impl SentenceEncoder for RelativePositionEncoder {
    type Encoding = DependencyEncoding<RelativePosition>;

    fn encode(&self, sentence: &Sentence) -> Result<Vec<Self::Encoding>, Error> {
        let mut encoded = Vec::with_capacity(sentence.len());
        for idx in 0..sentence.len() {
            let token = &sentence[idx];
            if token.is_root() {
                continue;
            }

            let triple = sentence
                .dep_graph()
                .head(idx)
                .ok_or(EncodeError::MissingHead)?;
            let relation = triple.relation().ok_or(EncodeError::MissingRelation)?;

            encoded.push(DependencyEncoding {
                label: relation.to_owned(),
                head: RelativePosition(triple.head() as isize - triple.dependent() as isize),
            });
        }

        Ok(encoded)
    }
}

impl SentenceDecoder for RelativePositionEncoder {
    type Encoding = DependencyEncoding<RelativePosition>;

    fn decode<E>(&self, labels: &[E], sentence: &mut Sentence) -> Result<(), Error>
    where
        E: Borrow<Self::Encoding>,
    {
        // This rewrapping has some potential overhead due to the auxiliary
        // Vec. But this avoids having two separate implementations for
        // decoding.
        let labels = labels.iter().map(|e| [e.borrow()]).collect::<Vec<_>>();
        self.decode_top_k(&labels, sentence)
    }
}

impl SentenceTopKDecoder for RelativePositionEncoder {
    type Encoding = DependencyEncoding<RelativePosition>;

    fn decode_top_k<S, E>(&self, labels: &[S], sentence: &mut Sentence) -> Result<(), Error>
    where
        E: Borrow<Self::Encoding>,
        S: AsRef<[E]>,
    {
        let token_indices: Vec<_> = (0..sentence.len())
            .filter(|&idx| sentence[idx].is_token())
            .collect();

        for (idx, encodings) in token_indices.into_iter().zip(labels) {
            for encoding in encodings.as_ref() {
                if let Ok(triple) =
                    RelativePositionEncoder::decode_idx(idx, sentence.len(), encoding.borrow())
                {
                    sentence.dep_graph_mut().add_deprel(triple);
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use conllx::graph::{DepTriple, Sentence};
    use conllx::token::TokenBuilder;

    use super::{RelativePosition, RelativePositionEncoder};
    use crate::depparse::{DecodeError, DependencyEncoding};
    use crate::SentenceTopKDecoder;

    // Small tests for the relative position encoder. Automatic
    // testing is performed in the module tests.

    #[test]
    fn position_out_of_bounds() {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("a").pos("A").into());
        sent.push(TokenBuilder::new("b").pos("B").into());

        assert_eq!(
            RelativePositionEncoder::decode_idx(
                1,
                sent.len(),
                &DependencyEncoding {
                    label: "X".into(),
                    head: RelativePosition(-2),
                },
            ),
            Err(DecodeError::PositionOutOfBounds)
        )
    }

    #[test]
    fn backoff() {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("a").pos("A").into());

        let decoder = RelativePositionEncoder;
        let labels = vec![vec![
            DependencyEncoding {
                label: "ROOT".into(),
                head: RelativePosition(-2),
            },
            DependencyEncoding {
                label: "ROOT".into(),
                head: RelativePosition(-1),
            },
        ]];

        decoder.decode_top_k(&labels, &mut sent).unwrap();

        assert_eq!(
            sent.dep_graph().head(1),
            Some(DepTriple::new(0, Some("ROOT"), 1))
        );
    }
}
