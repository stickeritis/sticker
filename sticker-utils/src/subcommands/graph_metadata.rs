use std::fs::File;
use std::io::BufReader;

use clap::{App, AppSettings, Arg, ArgMatches};
use stdinout::OrExit;
use sticker::tensorflow::TaggerGraph;

use crate::StickerApp;

static GRAPH: &str = "GRAPH";

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub struct GraphMetadataApp {
    graph: String,
}

impl StickerApp for GraphMetadataApp {
    fn app() -> App<'static, 'static> {
        App::new("graph-metadata")
            .about("Query model graph metadata")
            .settings(DEFAULT_CLAP_SETTINGS)
            .arg(
                Arg::with_name(GRAPH)
                    .help("Tensorflow graph")
                    .index(1)
                    .required(true),
            )
    }

    fn parse(matches: &ArgMatches) -> Self {
        let graph = matches.value_of(GRAPH).unwrap().to_owned();

        GraphMetadataApp { graph }
    }

    fn run(&self) {
        let reader =
            BufReader::new(File::open(&self.graph).or_exit("Cannot open graph for reading", 1));
        let model_config = Default::default();
        let graph = TaggerGraph::load_graph(reader, &model_config).expect("Cannot load graph");

        match graph
            .metadata()
            .or_exit("Cannot retrieve graph metadata", 1)
        {
            Some(metadata) => println!("{}", metadata),
            None => eprintln!("Graph does not contain metadata"),
        }
    }
}
