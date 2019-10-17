use std::borrow::Borrow;
use std::fs::File;
use std::path::Path;

use conllx::graph::Sentence;
use failure::{Fallible, ResultExt};

use super::{Config, Tagger, TomlRead};

pub struct Pipeline {
    taggers: Vec<Tagger>,
}

impl Pipeline {
    /// Create a pipeline from the given taggers.
    ///
    /// The pipeline will apply the taggers in the given order.
    pub fn new(taggers: Vec<Tagger>) -> Self {
        Pipeline { taggers }
    }

    /// Create a pipeline from tagger configurations.
    ///
    /// The pipeline will apply the taggers in the given order.
    pub fn from_configs(configs: &[impl Borrow<Config>]) -> Fallible<Self> {
        let taggers = configs
            .iter()
            .map(Borrow::borrow)
            .map(Tagger::new)
            .collect::<Fallible<Vec<_>>>()?;
        Ok(Pipeline { taggers })
    }

    /// Create a pipeline from tagger configuration filenames.
    ///
    /// The pipeline will apply the taggers in the given order.
    pub fn from_config_filenames(filenames: &[impl AsRef<Path>]) -> Fallible<Self> {
        let mut configs = Vec::with_capacity(filenames.len());

        for filename in filenames {
            let filename_lossy = filename.as_ref().to_string_lossy();

            let config_file = File::open(filename.as_ref()).context(format!(
                "Cannot open configuration file '{}'",
                filename_lossy
            ))?;

            let mut config = Config::from_toml_read(config_file).context(format!(
                "Cannot parse configuration file '{}'",
                filename_lossy
            ))?;
            config
                .relativize_paths(filename)
                .context(format!("Cannot relativize paths in '{}'", filename_lossy))?;

            configs.push(config);
        }

        Self::from_configs(&configs)
    }

    /// Tag sentences with the pipeline.
    ///
    /// The CoNLL-X graphs are updated in-place.
    pub fn tag_sentences(&self, sentences: &mut [&mut Sentence]) -> Fallible<()> {
        for tagger in &self.taggers {
            tagger.tag_sentences(sentences)?
        }

        Ok(())
    }
}
