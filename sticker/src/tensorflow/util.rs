use std::path::Path;

use failure::{err_msg, format_err, Error, Fallible};
use protobuf::Message;
use sticker_tf_proto::ConfigProto;
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

pub struct ConfigProtoBuilder {
    proto: ConfigProto,
}

impl ConfigProtoBuilder {
    pub fn new() -> Self {
        let mut proto = ConfigProto::new();
        proto.intra_op_parallelism_threads = 1;
        proto.inter_op_parallelism_threads = 1;

        ConfigProtoBuilder { proto }
    }

    pub fn gpu_count(mut self, n: usize) -> Self {
        self.proto.device_count.insert("GPU".to_owned(), n as i32);
        self
    }

    pub fn gpu_allow_growth(mut self, allow_growth: bool) -> Self {
        self.proto.mut_gpu_options().set_allow_growth(allow_growth);
        self
    }

    pub fn inter_op_parallelism_threads(mut self, n_threads: usize) -> Self {
        self.proto.inter_op_parallelism_threads = n_threads as i32;
        self
    }

    pub fn intra_op_parallelism_threads(mut self, n_threads: usize) -> Self {
        self.proto.intra_op_parallelism_threads = n_threads as i32;
        self
    }

    pub fn protobuf(&self) -> Fallible<Vec<u8>> {
        let mut bytes = Vec::new();
        self.proto.write_to_vec(&mut bytes)?;
        Ok(bytes)
    }
}

impl Default for ConfigProtoBuilder {
    fn default() -> Self {
        ConfigProtoBuilder::new()
    }
}
