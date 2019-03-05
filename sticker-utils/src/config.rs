use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use failure::{format_err, Error};
use finalfrontier::ReadModelBinary;
use ordered_float::NotNan;
use rust2vec::ReadWord2Vec;
use serde_derive::{Deserialize, Serialize};

use sticker::tensorflow::{Model, PlateauLearningRate};
use sticker::LayerEmbeddings;

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Config {
    pub labeler: Labeler,
    pub embeddings: Embeddings,
    pub model: Model,
    pub train: Train,
}

impl Config {
    /// Make configuration paths relative to the configuration file.
    pub fn relativize_paths<P>(&mut self, config_path: P) -> Result<(), Error>
    where
        P: AsRef<Path>,
    {
        let config_path = config_path.as_ref();

        self.labeler.labels = relativize_path(config_path, &self.labeler.labels)?;
        *self.embeddings.word.filename_mut() =
            relativize_path(config_path, &self.embeddings.word.filename_mut())?;
        self.model.graph = relativize_path(config_path, &self.model.graph)?;
        self.model.parameters = relativize_path(config_path, &self.model.parameters)?;

        Ok(())
    }
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Embeddings {
    pub word: Embedding,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Embedding {
    Word2Vec { filename: String, normalize: bool },
    FinalFrontier { filename: String },
}

impl Embedding {
    pub fn filename_mut(&mut self) -> &mut String {
        match self {
            Embedding::Word2Vec {
                ref mut filename, ..
            } => filename,
            Embedding::FinalFrontier { ref mut filename } => filename,
        }
    }
}

impl Embeddings {
    pub fn load_embeddings(&self) -> Result<LayerEmbeddings, Error> {
        let token_embeddings = self.load_layer_embeddings(&self.word)?;
        Ok(LayerEmbeddings::new(token_embeddings))
    }

    pub fn load_layer_embeddings(
        &self,
        embeddings: &Embedding,
    ) -> Result<sticker::Embeddings, Error> {
        match embeddings {
            Embedding::Word2Vec {
                filename,
                normalize,
            } => {
                let f = File::open(filename)?;
                let mut embeds =
                    rust2vec::Embeddings::read_word2vec_binary(&mut BufReader::new(f))?;

                if *normalize {
                    embeds.normalize()
                }

                Ok(embeds.into())
            }
            Embedding::FinalFrontier { filename } => {
                let f = File::open(filename)?;
                let model = finalfrontier::Model::read_model_binary(&mut BufReader::new(f))?;
                Ok(model.into())
            }
        }
    }
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Labeler {
    pub labels: String,
    pub read_ahead: usize,
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

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Train {
    pub initial_lr: NotNan<f32>,
    pub lr_scale: NotNan<f32>,
    pub lr_patience: usize,
    pub patience: usize,
}

impl Train {
    pub fn lr_schedule(&self) -> PlateauLearningRate {
        PlateauLearningRate::new(
            self.initial_lr.into_inner(),
            self.lr_scale.into_inner(),
            self.lr_patience,
        )
    }
}
