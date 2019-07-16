use std::fs::File;
use std::io::BufReader;

use clap::{App, AppSettings, Arg};
use stdinout::OrExit;
use sticker::tensorflow::TaggerGraph;

static GRAPH: &str = "GRAPH";

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub struct GraphMetadataApp {
    graph: String,
}

impl GraphMetadataApp {
    fn new() -> Self {
        let matches = App::new("sticker-graph-metadata")
            .settings(DEFAULT_CLAP_SETTINGS)
            .arg(
                Arg::with_name(GRAPH)
                    .help("Tensorflow graph")
                    .index(1)
                    .required(true),
            )
            .get_matches();

        let graph = matches.value_of(GRAPH).unwrap().to_owned();

        GraphMetadataApp { graph }
    }
}

impl Default for GraphMetadataApp {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    let app = GraphMetadataApp::new();

    let reader = BufReader::new(File::open(app.graph).or_exit("Cannot open graph for reading", 1));
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
