use std::path::Path;

use failure::{err_msg, format_err, Error, Fallible};
use protobuf::Message;
use rand::Rng;
use sticker_tf_proto::{ConfigProto, RewriterConfig_Toggle};
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

    pub fn auto_mixed_precision(mut self, mixed_precision: bool) -> Self {
        let toggle = if mixed_precision {
            RewriterConfig_Toggle::ON
        } else {
            RewriterConfig_Toggle::OFF
        };

        self.proto
            .mut_graph_options()
            .mut_rewrite_options()
            .set_auto_mixed_precision(toggle);

        self
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

pub struct RandomRemoveVec<T, R> {
    inner: Vec<T>,
    rng: R,
}

impl<T, R> RandomRemoveVec<T, R>
where
    R: Rng,
{
    /// Create a shuffler with the given capacity.
    pub fn with_capacity(capacity: usize, rng: R) -> Self {
        RandomRemoveVec {
            inner: Vec::with_capacity(capacity + 1),
            rng,
        }
    }

    /// Check whether the shuffler is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Push an element into the shuffler.
    pub fn push(&mut self, value: T) {
        self.inner.push(value);
    }

    /// Get the number of elements in the shuffler.
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T, R> RandomRemoveVec<T, R>
where
    R: Rng,
{
    /// Randomly remove an element from the shuffler.
    pub fn remove_random(&mut self) -> Option<T> {
        if self.inner.is_empty() {
            None
        } else {
            Some(
                self.inner
                    .swap_remove(self.rng.gen_range(0, self.inner.len())),
            )
        }
    }

    /// Add `replacement` to the inner and randomly remove an element.
    ///
    /// `replacement` could also be drawn randomly.
    pub fn push_and_remove_random(&mut self, replacement: T) -> T {
        self.inner.push(replacement);
        self.inner
            .swap_remove(self.rng.gen_range(0, self.inner.len()))
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};
    use rand_xorshift::XorShiftRng;

    use super::RandomRemoveVec;

    #[test]
    fn random_remove_vec() {
        let mut rng = XorShiftRng::seed_from_u64(42);
        let mut elems = RandomRemoveVec::with_capacity(3, XorShiftRng::seed_from_u64(42));
        elems.push(1);
        elems.push(2);
        elems.push(3);

        // Before: [1 2 3]
        assert_eq!(rng.gen_range(0, 4 as usize), 1);
        assert_eq!(elems.push_and_remove_random(4), 2);

        // Before: [1 4 3]
        assert_eq!(rng.gen_range(0, 4 as usize), 2);
        assert_eq!(elems.push_and_remove_random(5), 3);

        // Before: [1 4 5]
        assert_eq!(rng.gen_range(0, 4 as usize), 1);
        assert_eq!(elems.push_and_remove_random(6), 4);

        // Before: [1 6 5]
        assert_eq!(rng.gen_range(0, 3 as usize), 1);
        assert_eq!(elems.remove_random().unwrap(), 6);

        // Before: [1 5]
        assert_eq!(rng.gen_range(0, 2 as usize), 0);
        assert_eq!(elems.remove_random().unwrap(), 1);

        // Before: [5]
        assert_eq!(rng.gen_range(0, 1 as usize), 0);
        assert_eq!(elems.remove_random().unwrap(), 5);

        // Exhausted
        assert_eq!(elems.remove_random(), None);

        // The buffer is empty, so always return the next number
        assert_eq!(elems.push_and_remove_random(7), 7);
        assert_eq!(elems.push_and_remove_random(8), 8);
    }
}
