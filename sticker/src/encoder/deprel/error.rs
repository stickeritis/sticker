use failure::Fail;

/// Encoder errors.
#[derive(Clone, Debug, Eq, Fail, PartialEq)]
pub enum EncodeError {
    /// The token does not have a head.
    #[fail(display = "missing head: '{}'", form)]
    MissingHead { form: String },

    /// The token does not have a part-of-speech.
    #[fail(display = "missing part-of-speech tag: '{}'", form)]
    MissingPOS { form: String },

    /// The token does not have a dependency relation.
    #[fail(display = "missing dependency relation: '{}'", form)]
    MissingRelation { form: String },
}

/// Decoder errors.
#[derive(Clone, Copy, Debug, Eq, Fail, PartialEq)]
pub(crate) enum DecodeError {
    /// The head position is out of bounds.
    #[fail(display = "position out of bounds")]
    PositionOutOfBounds,

    /// The head part-of-speech tag does not occur in the sentence.
    #[fail(display = "unknown part-of-speech tag")]
    InvalidPOS,
}
