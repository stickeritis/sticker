use conllx::graph::Sentence;
use conllx::io::WriteSentence;
use failure::Error;

use sticker::wrapper::Pipeline;

// Wrap the sentence processing in a data type. This has the benefit that
// we can use a destructor to write the last (possibly incomplete) batch.
pub struct SentProcessor<'a, W>
where
    W: WriteSentence,
{
    pipeline: &'a Pipeline,
    writer: W,
    batch_size: usize,
    read_ahead: usize,
    buffer: Vec<Sentence>,
}

impl<'a, W> SentProcessor<'a, W>
where
    W: WriteSentence,
{
    pub fn new(pipeline: &'a Pipeline, writer: W, batch_size: usize, read_ahead: usize) -> Self {
        assert!(batch_size > 0, "Batch size should at least be 1.");
        assert!(read_ahead > 0, "Read ahead should at least be 1.");

        SentProcessor {
            pipeline,
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
            self.pipeline.tag_sentences(batch)?;
        }

        // Write out sentences.
        let mut sents = Vec::new();
        std::mem::swap(&mut sents, &mut self.buffer);
        for sent in sents {
            self.writer.write_sentence(&sent)?;
        }

        Ok(())
    }
}

impl<'a, W> Drop for SentProcessor<'a, W>
where
    W: WriteSentence,
{
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            if let Err(err) = self.tag_buffered_sentences() {
                eprintln!("Error tagging sentences: {}", err);
            }
        }
    }
}
