use std::io::BufRead;

use failure::Fallible;

pub fn count_conllx_sentences(buf_read: impl BufRead) -> Fallible<usize> {
    let mut n_sents = 0;

    for line in buf_read.lines() {
        let line = line?;
        if line.starts_with("1\t") {
            n_sents += 1;
        }
    }

    Ok(n_sents)
}
