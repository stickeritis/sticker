use clap::{App, AppSettings, Arg};

static CONFIG: &str = "CONFIG";

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
