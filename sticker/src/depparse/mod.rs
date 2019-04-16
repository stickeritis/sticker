use serde_derive::{Deserialize, Serialize};

mod errors;
pub use crate::depparse::errors::*;

mod relative_position;
pub use crate::depparse::relative_position::*;

/// Encoding of a dependency relation as a token label.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct DependencyEncoding<H> {
    head: H,
    label: String,
}

impl<H> DependencyEncoding<H> {
    /// Get the head representation.
    pub fn head(&self) -> &H {
        &self.head
    }

    /// Get the dependency label.
    pub fn label(&self) -> &str {
        &self.label
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::path::Path;

    use conllx::graph::{Node, Sentence};
    use conllx::io::Reader;

    use super::RelativePositionEncoder;
    use crate::{SentenceDecoder, SentenceEncoder};

    static NON_PROJECTIVE_DATA: &'static str = "testdata/nonprojective.conll";

    fn copy_sentence_without_deprels(sentence: &Sentence) -> Sentence {
        let mut copy = Sentence::new();
        for token in sentence.iter().filter_map(Node::token) {
            copy.push(token.clone());
        }

        copy
    }

    fn test_encoding<P, E, C>(path: P, encoder_decoder: E)
    where
        P: AsRef<Path>,
        E: SentenceEncoder<Encoding = C> + SentenceDecoder<Encoding = C>,
    {
        let f = File::open(path).unwrap();
        let reader = Reader::new(BufReader::new(f));

        for sentence in reader {
            let sentence = sentence.unwrap();

            // Encode
            let encodings = encoder_decoder.encode(&sentence).unwrap();

            // Decode
            let mut test_sentence = copy_sentence_without_deprels(&sentence);
            encoder_decoder
                .decode(&encodings, &mut test_sentence)
                .unwrap();

            assert_eq!(sentence, test_sentence);
        }
    }

    #[test]
    fn relative_position() {
        let encoder = RelativePositionEncoder;
        test_encoding(NON_PROJECTIVE_DATA, encoder);
    }
}