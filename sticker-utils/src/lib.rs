mod config;
pub use crate::config::{Config, Embedding, Embeddings, Labeler, Train};

mod progress;
pub use crate::progress::FileProgress;

mod serialization;
pub use crate::serialization::{CborRead, CborWrite, TomlRead};

#[cfg(test)]
#[macro_use]
extern crate lazy_static;

#[cfg(test)]
mod config_tests;
