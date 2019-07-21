use failure::Fail;

#[derive(Clone, Debug, Eq, Fail, PartialEq)]
pub enum EncodeError {
    /// The token does not have a label.
    #[fail(display = "token without a label: '{}'", form)]
    MissingLabel { form: String },
}
