use std::fs::File;
use std::hash::Hash;
use std::io::{BufReader, Read};
use std::path::Path;

use failure::{format_err, Error};
use finalfusion::embeddings::Embeddings as FiFuEmbeddings;
use finalfusion::prelude::*;
use numberer::Numberer;
use serde_derive::{Deserialize, Serialize};
use sticker_encoders::layer::Layer;

use crate::serialization::CborRead;
use crate::tensorflow::ModelConfig;
use crate::LayerEmbeddings;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub labeler: Labeler,
    pub input: Input,
    pub model: ModelConfig,
}

impl Config {
    /// Make configuration paths relative to the configuration file.
    pub fn relativize_paths<P>(&mut self, config_path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        let config_path = config_path.as_ref();

        self.labeler.labels = relativize_path(config_path, &self.labeler.labels)?;
        self.input.embeddings.word.filename =
            relativize_path(config_path, &self.input.embeddings.word.filename)?;
        if let Some(ref mut embeddings) = self.input.embeddings.tag {
            embeddings.filename = relativize_path(config_path, &embeddings.filename)?;
        }
        self.model.graph = relativize_path(config_path, &self.model.graph)?;
        self.model.parameters = relativize_path(config_path, &self.model.parameters)?;

        Ok(())
    }
}

pub trait TomlRead {
    fn from_toml_read<R>(read: R) -> Result<Config, Error>
    where
        R: Read;
}

impl TomlRead for Config {
    fn from_toml_read<R>(mut read: R) -> Result<Self, Error>
    where
        R: Read,
    {
        let mut data = String::new();
        read.read_to_string(&mut data)?;
        let config: Config = toml::from_str(&data)?;

        if config.model.batch_size.is_some() {
            eprintln!("The model.batch_size option is deprecated and not used anymore");
        }

        if config.model.intra_op_parallelism_threads.is_some() {
            eprintln!("The model.intra_op_parallelism option is deprecated and not used anymore");
        }

        if config.model.inter_op_parallelism_threads.is_some() {
            eprintln!("The model.inter_op_parallelism option is deprecated and not used anymore");
        }

        if config.labeler.read_ahead.is_some() {
            eprintln!("The labeler.read_ahead option is deprecated and not used anymore");
        }

        Ok(config)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Embeddings {
    pub word: Embedding,
    pub tag: Option<Embedding>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Embedding {
    pub filename: String,
    pub alloc: EmbeddingAlloc,
}

impl Embeddings {
    pub fn load_embeddings(&self) -> Result<LayerEmbeddings, Error> {
        let token_embeddings = self.load_layer_embeddings(&self.word)?;
        let tag_embeddings = match &self.tag {
            Some(embed) => Some(self.load_layer_embeddings(embed)?),
            None => None,
        };
        Ok(LayerEmbeddings::new(token_embeddings, tag_embeddings))
    }

    pub fn load_layer_embeddings(
        &self,
        embeddings: &Embedding,
    ) -> Result<crate::Embeddings, Error> {
        let f = File::open(&embeddings.filename)?;
        let embeds: FiFuEmbeddings<VocabWrap, StorageWrap> = match embeddings.alloc {
            EmbeddingAlloc::Read => ReadEmbeddings::read_embeddings(&mut BufReader::new(f))?,
            EmbeddingAlloc::Mmap => MmapEmbeddings::mmap_embeddings(&mut BufReader::new(f))?,
        };

        Ok(embeds.into())
    }
}

#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum EmbeddingAlloc {
    Mmap,
    Read,
}

#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum EncoderType {
    RelativePosition,
    RelativePOS,
}

/// Input configuration
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub embeddings: Embeddings,
    pub subwords: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Labeler {
    pub labels: String,

    // Not used anymore, changed from usize to Option<usize> in order
    // to read older configuration files.
    pub read_ahead: Option<usize>,

    pub labeler_type: LabelerType,
}

impl Labeler {
    pub fn load_labels<E>(&self) -> Result<Numberer<E>, Error>
    where
        E: Eq + Hash,
        Numberer<E>: CborRead,
    {
        let labels_path = Path::new(&self.labels);

        eprintln!("Loading labels from: {:?}", labels_path);

        let f = File::open(labels_path)?;
        let labels = Numberer::from_cbor_read(f)?;

        Ok(labels)
    }
}

#[serde(rename_all = "lowercase")]
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum LabelerType {
    Parser(EncoderType),
    Sequence(Layer),
}

fn relativize_path(config_path: &Path, filename: &str) -> Result<String, Error> {
    if filename.is_empty() {
        return Ok(filename.to_owned());
    }

    let path = Path::new(&filename);

    // Don't touch absolute paths.
    if path.is_absolute() {
        return Ok(filename.to_owned());
    }

    let abs_config_path = config_path.canonicalize()?;
    Ok(abs_config_path
        .parent()
        .ok_or_else(|| {
            format_err!(
                "Cannot get parent path of the configuration file: {}",
                abs_config_path.to_string_lossy()
            )
        })?
        .join(path)
        .to_str()
        .ok_or_else(|| {
            format_err!(
                "Cannot cannot convert partent path to string: {}",
                abs_config_path.to_string_lossy()
            )
        })?
        .to_owned())
}

#[cfg(test)]
mod tests {
    use lazy_static::lazy_static;
    use std::fs::File;
    use sticker_encoders::layer::Layer;

    use crate::tensorflow::ModelConfig;

    use super::{
        Config, Embedding, EmbeddingAlloc, Embeddings, Input, Labeler, LabelerType, TomlRead,
    };

    lazy_static! {
        static ref BASIC_LABELER_CHECK: Config = Config {
            labeler: Labeler {
                labeler_type: LabelerType::Sequence(Layer::Feature("tf".to_string())),
                labels: "sticker.labels".to_owned(),
                read_ahead: Some(10),
            },
            input: Input {
                embeddings: Embeddings {
                    word: Embedding {
                        filename: "word-vectors.bin".into(),
                        alloc: EmbeddingAlloc::Mmap,
                    },
                    tag: Some(Embedding {
                        filename: "tag-vectors.bin".into(),
                        alloc: EmbeddingAlloc::Read,
                    }),
                },
                subwords: true,
            },
            model: ModelConfig {
                batch_size: Some(128),
                gpu_allow_growth: true,
                graph: "sticker.graph".to_owned(),
                parameters: "sticker.model".to_owned(),
                intra_op_parallelism_threads: Some(4),
                inter_op_parallelism_threads: Some(4),
            }
        };
    }

    #[test]
    fn test_parse_config() {
        let f = File::open("testdata/sticker.conf").unwrap();
        let config = Config::from_toml_read(f).unwrap();
        assert_eq!(*BASIC_LABELER_CHECK, config);
    }
}
