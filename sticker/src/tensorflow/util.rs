use std::path::Path;

use failure::{err_msg, format_err, Error, Fallible};
use tensorflow::Status;

/// Tensorflow requires a path that contains a directory component.
pub(crate) fn prepare_path<P>(path: P) -> Fallible<String>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let path = if path.components().count() == 1 {
        Path::new("./").join(path)
    } else {
        path.to_owned()
    };

    path.to_str()
        .ok_or_else(|| err_msg("Filename contains non-unicode characters"))
        .map(ToOwned::to_owned)
}

/// tensorflow::Status is not Sync, which is required by failure.
pub(crate) fn status_to_error(status: Status) -> Error {
    format_err!("{}", status)
}
