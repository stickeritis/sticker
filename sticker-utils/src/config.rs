use std::fs::File;
use std::hash::Hash;
use std::io::BufReader;
use std::path::Path;

use failure::{format_err, Error};
use finalfusion::embeddings::Embeddings as FiFuEmbeddings;
use finalfusion::prelude::*;
use serde_derive::{Deserialize, Serialize};

use sticker::tensorflow::ModelConfig;
use sticker::{Layer, LayerEmbeddings, Numberer};

use crate::CborRead;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub labeler: Labeler,
    pub embeddings: Embeddings,
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
        self.embeddings.word.filename =
            relativize_path(config_path, &self.embeddings.word.filename)?;
        if let Some(ref mut embeddings) = self.embeddings.tag {
            embeddings.filename = relativize_path(config_path, &embeddings.filename)?;
        }
        self.model.graph = relativize_path(config_path, &self.model.graph)?;
        self.model.parameters = relativize_path(config_path, &self.model.parameters)?;

        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Embeddings {
    pub word: Embedding,
    pub tag: Option<Embedding>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
    ) -> Result<sticker::Embeddings, Error> {
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Labeler {
    pub labels: String,
    pub read_ahead: usize,
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
