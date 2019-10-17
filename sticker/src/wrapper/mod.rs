//! High-level wrapper for the tagger.
//!
//! This module contains high-level wrappers around sticker's
//! functionality. `Tagger` provides a simple interface to
//! `tensorflow::Tagger`, that only requires a `Config` struct to be
//! initialized. `Pipeline` uses multiple `TaggerWrapper`s to form a
//! pipeline.

mod config;
pub use config::{Config, EncoderType, LabelerType, TomlRead};

mod pipeline;
pub use pipeline::Pipeline;

mod tagger;
pub use tagger::Tagger;
