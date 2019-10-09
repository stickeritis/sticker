use clap::{App, AppSettings, Arg};

pub static BATCH_SIZE: &str = "BATCH_SIZE";
pub static CONFIG: &str = "CONFIG";
pub static CONFIGS: &str = "CONFIGS";
pub static READ_AHEAD: &str = "READ_AHEAD";

static DEFAULT_CLAP_SETTINGS: &[AppSettings] = &[
    AppSettings::DontCollapseArgsInUsage,
    AppSettings::UnifiedHelpMessage,
];

pub fn sticker_app<'a, 'b>(name: &str) -> App<'a, 'b> {
    App::new(name).settings(DEFAULT_CLAP_SETTINGS).arg(
        Arg::with_name(CONFIG)
            .help("Sticker configuration")
            .index(1)
            .required(true),
    )
}

pub fn sticker_pipeline_app<'a, 'b>(name: &str) -> App<'a, 'b> {
    App::new(name)
        .settings(DEFAULT_CLAP_SETTINGS)
        .arg(
            Arg::with_name(CONFIGS)
                .help("Sticker configuration files")
                .min_values(1)
                .required(true),
        )
        .arg(
            Arg::with_name(BATCH_SIZE)
                .help("Batch size")
                .long("batchsize")
                .default_value("256"),
        )
        .arg(
            Arg::with_name(READ_AHEAD)
                .help("Readahead (number of batches)")
                .long("readahead")
                .default_value("10"),
        )
}
