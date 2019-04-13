use std::io::Write;

use conllx::graph::{Node, Sentence};
use conllx::io::{WriteSentence, Writer};
use failure::Error;
use sticker::{Layer, LayerValue, Tag};

// Wrap the sentence processing in a data type. This has the benefit that
// we can use a destructor to write the last (possibly incomplete) batch.
pub struct SentProcessor<'a, T, W>
where
    T: Tag,
    W: Write,
{
    layer: Layer,
    tagger: &'a T,
    writer: Writer<W>,
    batch_size: usize,
    read_ahead: usize,
    buffer: Vec<Sentence>,
}

impl<'a, T, W> SentProcessor<'a, T, W>
where
    T: Tag,
    W: Write,
{
    pub fn new(
        layer: Layer,
        tagger: &'a T,
        writer: Writer<W>,
        batch_size: usize,
        read_ahead: usize,
    ) -> Self {
        assert!(batch_size > 0, "Batch size should at least be 1.");
        assert!(read_ahead > 0, "Read ahead should at least be 1.");

        SentProcessor {
            layer,
            tagger,
            writer,
            batch_size,
            read_ahead,
            buffer: Vec::new(),
        }
    }

    pub fn process(&mut self, sent: Sentence) -> Result<(), Error> {
        self.buffer.push(sent);

        if self.buffer.len() == self.batch_size * self.read_ahead {
            self.tag_buffered_sentences()?;
        }

        Ok(())
    }

    fn tag_buffered_sentences(&mut self) -> Result<(), Error> {
        // Sort sentences by length.
        let mut sent_refs: Vec<_> = self.buffer.iter_mut().map(|s| s).collect();
        sent_refs.sort_unstable_by_key(|s| s.len());

        // Split in batches, tag, and merge results.
        for batch in sent_refs.chunks_mut(self.batch_size) {
            let labels = labels_to_owned(self.tagger.tag_sentences(batch)?);
            Self::merge_labels(&self.layer, batch, labels)?;
        }

        // Write out sentences.
        let mut sents = Vec::new();
        std::mem::swap(&mut sents, &mut self.buffer);
        for sent in sents {
            self.writer.write_sentence(&sent)?;
        }

        Ok(())
    }

    fn merge_labels(
        layer: &Layer,
        sentences: &mut [&mut Sentence],
        labels: Vec<Vec<String>>,
    ) -> Result<(), Error>
    where
        W: Write,
    {
        for (sentence, sent_labels) in sentences.iter_mut().zip(labels) {
            {
                for (token, label) in sentence
                    .iter_mut()
                    .filter_map(Node::token_mut)
                    .zip(sent_labels)
                {
                    token.set_value(layer, label);
                }
            }
        }

        Ok(())
    }
}

impl<'a, T, W> Drop for SentProcessor<'a, T, W>
where
    T: Tag,
    W: Write,
{
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            if let Err(err) = self.tag_buffered_sentences() {
                eprintln!("Error tagging sentences: {}", err);
            }
        }
    }
}

fn labels_to_owned(labels: Vec<Vec<&str>>) -> Vec<Vec<String>> {
    labels
        .into_iter()
        .map(|sv| sv.into_iter().map(str::to_owned).collect())
        .collect()
}
