use std::fs::File;
use std::hash::Hash;

use clap::{App, Arg, ArgMatches};
use failure::Fallible;
use stdinout::OrExit;
use sticker::encoder::deprel::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::encoder::layer::LayerEncoder;
use sticker::encoder::lemma::EditTreeEncoder;
use sticker::encoder::SentenceEncoder;
use sticker::serialization::CborRead;
use sticker::wrapper::{Config, EncoderType, LabelerType, TomlRead};
use sticker::Numberer;

use crate::traits::StickerApp;

const CONFIG: &str = "CONFIG";

pub struct PrintLabelsApp {
    config: String,
}

impl PrintLabelsApp {
    fn print_labels_with_encoder<E>(&self, config: &Config) -> Fallible<()>
    where
        E: SentenceEncoder,
        E::Encoding: Clone + Eq + Hash + ToString,
        Numberer<E::Encoding>: CborRead,
    {
        let labels: Numberer<E::Encoding> = config.labeler.load_labels().or_exit(
            format!("Cannot load label file '{}'", config.labeler.labels),
            1,
        );

        println!("0\tPADDING");
        for idx in 1..labels.len() {
            println!("{}\t{}", idx, labels.value(idx).unwrap().to_string());
        }

        Ok(())
    }
}

impl StickerApp for PrintLabelsApp {
    fn app() -> App<'static, 'static> {
        App::new("print-labels")
            .about("Print the labels of a model")
            .arg(
                Arg::with_name(CONFIG)
                    .help("Sticker configuration")
                    .index(1)
                    .required(true),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        let config = matches.value_of(CONFIG).unwrap().into();
        PrintLabelsApp { config }
    }

    fn run(&self) {
        let config_file = File::open(&self.config).or_exit(
            format!("Cannot open configuration file '{}'", self.config),
            1,
        );
        let mut config =
            Config::from_toml_read(config_file).or_exit("Cannot parse configuration", 1);
        config
            .relativize_paths(&self.config)
            .or_exit("Cannot relativize paths in configuration", 1);

        match config.labeler.labeler_type {
            LabelerType::Lemma => self.print_labels_with_encoder::<EditTreeEncoder>(&config),
            LabelerType::Sequence(_) => self.print_labels_with_encoder::<LayerEncoder>(&config),
            LabelerType::Parser(EncoderType::RelativePOS) => {
                self.print_labels_with_encoder::<RelativePOSEncoder>(&config)
            }
            LabelerType::Parser(EncoderType::RelativePosition) => {
                self.print_labels_with_encoder::<RelativePositionEncoder>(&config)
            }
        }
        .or_exit("Error while printing labels", 1);
    }
}
