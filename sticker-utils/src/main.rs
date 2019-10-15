use std::io::stdout;

use clap::{crate_version, App, AppSettings, Arg, Shell, SubCommand};

mod subcommands;

mod traits;
pub use self::traits::{StickerApp, StickerPipelineApp, StickerTrainApp};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
    AppSettings::SubcommandRequiredElseHelp,
];

fn main() {
    // Known subapplications.
    let apps = vec![
        subcommands::Dep2LabelApp::app(),
        subcommands::GraphMetadataApp::app(),
        subcommands::PrepareApp::app(),
        subcommands::PretrainApp::app(),
        subcommands::ServerApp::app(),
        subcommands::TagApp::app(),
        subcommands::TrainApp::app(),
    ];

    let cli = App::new("sticker")
        .settings(DEFAULT_CLAP_SETTINGS)
        .about("A neural sequence labeler")
        .version(crate_version!())
        .subcommands(apps)
        .subcommand(
            SubCommand::with_name("completions")
                .about("Generate completion scripts for your shell")
                .setting(AppSettings::ArgRequiredElseHelp)
                .arg(Arg::with_name("shell").possible_values(&Shell::variants())),
        );
    let matches = cli.clone().get_matches();

    match matches.subcommand_name().unwrap() {
        "completions" => {
            let shell = matches
                .subcommand_matches("completions")
                .unwrap()
                .value_of("shell")
                .unwrap();
            write_completion_script(cli, shell.parse::<Shell>().unwrap());
        }
        "dep2label" => {
            subcommands::Dep2LabelApp::parse(matches.subcommand_matches("dep2label").unwrap()).run()
        }
        "graph-metadata" => subcommands::GraphMetadataApp::parse(
            matches.subcommand_matches("graph-metadata").unwrap(),
        )
        .run(),
        "prepare" => {
            subcommands::PrepareApp::parse(matches.subcommand_matches("prepare").unwrap()).run()
        }
        "pretrain" => {
            subcommands::PretrainApp::parse(matches.subcommand_matches("pretrain").unwrap()).run()
        }
        "server" => {
            subcommands::ServerApp::parse(matches.subcommand_matches("server").unwrap()).run()
        }
        "tag" => subcommands::TagApp::parse(matches.subcommand_matches("tag").unwrap()).run(),
        "train" => subcommands::TrainApp::parse(matches.subcommand_matches("train").unwrap()).run(),
        _unknown => unreachable!(),
    }
}

fn write_completion_script(mut cli: App, shell: Shell) {
    cli.gen_completions_to("sticker", shell, &mut stdout());
}
