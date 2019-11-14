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
    const MAX_LEN: &'static str = "MAX_LEN";
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
                Arg::with_name(Self::MAX_LEN)
                    .long("maxlen")
                    .value_name("N")
                    .takes_value(true)
                    .help("Ignore sentences longer than N tokens"),
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
    const BATCH_SIZE: &'static str = "BATCH_SIZE";
    const CONFIG: &'static str = "CONFIG";
    const INTER_OP_THREADS: &'static str = "INTER_OP_THREADS";
    const INTRA_OP_THREADS: &'static str = "INTRA_OP_THREADS";

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
            .arg(
                Arg::with_name(Self::BATCH_SIZE)
                    .help("Batch size")
                    .long("batchsize")
                    .default_value("256"),
            )
            .arg(
                Arg::with_name(Self::INTER_OP_THREADS)
                    .help("Inter op parallelism threads")
                    .long("inter-op-threads")
                    .default_value("4"),
            )
            .arg(
                Arg::with_name(Self::INTRA_OP_THREADS)
                    .help("Intra op parallelism threads")
                    .long("intra-op-threads")
                    .default_value("4"),
            )
    }
}
