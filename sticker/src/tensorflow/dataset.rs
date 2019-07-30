use std::hash::Hash;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::usize;

use conllx::io::{ReadSentence, Reader, Sentences};
use failure::Fallible;

use super::tensor::{LabelTensor, TensorBuilder};
use crate::encoder::{CategoricalEncoder, SentenceEncoder};
use crate::SentVectorizer;

/// A set of training/validation data.
///
/// A data set provides an iterator over the batches in that
/// dataset.
pub trait DataSet<'a, E>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    type Iter: Iterator<Item = Fallible<TensorBuilder<LabelTensor>>>;

    /// Get an iterator over the dataset batches.
    ///
    /// The sequence inputs are encoded with the given `vectorizer`,
    /// the sequence labels using the `encoder`.
    ///
    /// Sentences longer than `max_len` are skipped. If you want to
    /// include all sentences, you can use `usize::MAX` as the maximum
    /// length.
    fn batches(
        self,
        encoder: &'a mut CategoricalEncoder<E, E::Encoding>,
        vectorizer: &'a SentVectorizer,
        batch_size: usize,
        max_len: usize,
    ) -> Fallible<Self::Iter>;
}

/// A CoNLL-X data set.
pub struct ConllxDataSet<R>(R);

impl<R> ConllxDataSet<R> where {
    /// Construct a CoNLL-X dataset.
    pub fn new(read: R) -> Self {
        ConllxDataSet(read)
    }
}

impl<'a, 'ds, E, R> DataSet<'a, E> for &'ds mut ConllxDataSet<R>
where
    E: 'a + SentenceEncoder,
    E::Encoding: 'a + Clone + Eq + Hash,
    R: Read + Seek,
{
    type Iter = ConllxIter<'a, E, Reader<BufReader<&'ds mut R>>>;

    fn batches(
        self,
        encoder: &'a mut CategoricalEncoder<E, E::Encoding>,
        vectorizer: &'a SentVectorizer,
        batch_size: usize,
        max_len: usize,
    ) -> Fallible<Self::Iter> {
        // Rewind to the beginning of the data (if necessary).
        self.0.seek(SeekFrom::Start(0))?;

        let reader = Reader::new(BufReader::new(&mut self.0));

        Ok(ConllxIter {
            batch_size,
            encoder,
            max_len,
            sentences: reader.sentences(),
            vectorizer,
        })
    }
}

pub struct ConllxIter<'a, E, R>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    R: ReadSentence,
{
    batch_size: usize,
    max_len: usize,
    encoder: &'a mut CategoricalEncoder<E, E::Encoding>,
    vectorizer: &'a SentVectorizer,
    sentences: Sentences<R>,
}

impl<'a, E, R> Iterator for ConllxIter<'a, E, R>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    R: ReadSentence,
{
    type Item = Fallible<TensorBuilder<LabelTensor>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch_sentences = Vec::with_capacity(self.batch_size);
        while let Some(sentence) = self.sentences.next() {
            let sentence = match sentence {
                Ok(sentence) => sentence,
                Err(err) => return Some(Err(err)),
            };

            if sentence.len() <= self.max_len {
                batch_sentences.push(sentence);
            }

            if batch_sentences.len() == self.batch_size {
                break;
            }
        }

        // Check whether the reader is exhausted.
        if batch_sentences.is_empty() {
            return None;
        }

        let max_seq_len = batch_sentences
            .iter()
            .map(|s| s.len() - 1)
            .max()
            .unwrap_or(0);
        let mut builder = TensorBuilder::new(
            batch_sentences.len(),
            max_seq_len,
            self.vectorizer.input_len(),
            self.vectorizer.has_subwords(),
        );

        for sentence in batch_sentences {
            let inputs = match self.vectorizer.realize(&sentence) {
                Ok(inputs) => inputs,
                Err(err) => return Some(Err(err)),
            };

            let labels = match self.encoder.encode(&sentence) {
                Ok(encoding) => encoding
                    .into_iter()
                    .map(|label| label as i32)
                    .collect::<Vec<_>>(),
                Err(err) => return Some(Err(err)),
            };
            builder.add_with_labels(inputs, &labels);
        }

        Some(Ok(builder))
    }
}
