use clap::{crate_version, App, AppSettings, Arg, ArgMatches};

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub trait StickerApp {
    fn app() -> App<'static, 'static>;

    fn parse(matches: &ArgMatches) -> Self;

    fn run(&self);
}

pub trait StickerPipelineApp: StickerApp {
    const BATCH_SIZE: &'static str = "BATCH_SIZE";
    const CONFIGS: &'static str = "CONFIGS";
    const READ_AHEAD: &'static str = "READ_AHEAD";

    fn pipeline_app<'a, 'b>(name: &str) -> App<'a, 'b> {
        App::new(name)
            .settings(DEFAULT_CLAP_SETTINGS)
            .version(crate_version!())
            .arg(
                Arg::with_name(Self::CONFIGS)
                    .help("Sticker configuration files")
                    .min_values(1)
                    .required(true),
            )
            .arg(
                Arg::with_name(Self::BATCH_SIZE)
                    .help("Batch size")
                    .long("batchsize")
                    .default_value("256"),
            )
            .arg(
                Arg::with_name(Self::READ_AHEAD)
                    .help("Readahead (number of batches)")
                    .long("readahead")
                    .default_value("10"),
            )
    }
}

pub trait StickerTrainApp: StickerApp {
    const CONFIG: &'static str = "CONFIG";

    fn train_app<'a, 'b>(name: &str) -> App<'a, 'b> {
        App::new(name)
            .settings(DEFAULT_CLAP_SETTINGS)
            .version(crate_version!())
            .arg(
                Arg::with_name(Self::CONFIG)
                    .help("Sticker configuration")
                    .index(1)
                    .required(true),
            )
    }
}
