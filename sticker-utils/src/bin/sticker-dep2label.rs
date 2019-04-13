use std::env::args;
use std::io::BufWriter;
use std::process;

use conllx::graph::Node;
use conllx::io::{ReadSentence, WriteSentence};
use conllx::token::Features;
use getopts::Options;
use stdinout::{Input, OrExit, Output};

use sticker::depparse::{RelativePOSEncoder, RelativePositionEncoder};
use sticker::SentenceEncoder;

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] [INPUT] [OUTPUT]", program);
    print!("{}", opts.usage(&brief));
    process::exit(1);
}

fn main() {
    let args: Vec<String> = args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.optopt(
        "e",
        "encoder",
        "dependency encoder",
        "rel_pos or rel_position (default: rel_pos)",
    );
    opts.optflag("h", "help", "print this help menu");
    let matches = opts.parse(&args[1..]).or_exit("Cannot parse options", 1);

    if matches.opt_present("h") {
        print_usage(&program, opts);
        return;
    }

    if matches.free.len() > 2 {
        print_usage(&program, opts);
        return;
    }

    let input = Input::from(matches.free.get(0));
    let reader =
        conllx::io::Reader::new(input.buf_read().or_exit("Cannot open input for reading", 1));

    let output = Output::from(matches.free.get(1));
    let writer = conllx::io::Writer::new(BufWriter::new(
        output.write().or_exit("Cannot open output for writing", 1),
    ));

    match matches
        .opt_str("e")
        .unwrap_or_else(|| "rel_pos".to_owned())
        .as_str()
    {
        "rel_pos" => label_with_encoder(RelativePOSEncoder, reader, writer),
        "rel_position" => label_with_encoder(RelativePositionEncoder, reader, writer),
        unknown => {
            eprintln!("Unknown encoder: {}", unknown);
            process::exit(1);
        }
    }
}

fn label_with_encoder<E, R, W>(encoder: E, read: R, mut write: W)
where
    E: SentenceEncoder,
    E::Encoding: ToString,
    R: ReadSentence,
    W: WriteSentence,
{
    for sentence in read.sentences() {
        let mut sentence = sentence.or_exit("Cannot parse sentence", 1);

        let encoded = encoder
            .encode(&sentence)
            .or_exit("Cannot dependency-encode sentence", 1);

        for (token, encoding) in sentence.iter_mut().filter_map(Node::token_mut).zip(encoded) {
            let mut features = token
                .features()
                .map(Features::as_map)
                .cloned()
                .unwrap_or_default();
            features.insert("deplabel".to_owned(), Some(encoding.to_string()));
            token.set_features(Some(Features::from_iter(features)));
        }

        write
            .write_sentence(&sentence)
            .or_exit("Cannot write sentence", 1);
    }
}
