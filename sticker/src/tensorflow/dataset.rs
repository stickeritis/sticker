use std::hash::Hash;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::usize;

use conllx::graph::Sentence;
use conllx::io::{ReadSentence, Reader};
use failure::{Error, Fallible};

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
        max_len: Option<usize>,
    ) -> Fallible<Self::Iter>;
}

/// A CoNLL-X data set.
pub struct ConllxDataSet<R>(R);

impl<R> ConllxDataSet<R> {
    /// Construct a CoNLL-X dataset.
    pub fn new(read: R) -> Self {
        ConllxDataSet(read)
    }

    /// Returns an `Iterator` over `Result<Sentence, Error>`.
    ///
    /// Depending on the parameters the returned iterator filters
    /// sentences by their lengths or returns the sentences in
    /// sequence without filtering them.
    ///
    /// If `max_len` == `usize::MAX`, no filtering is performed.
    fn get_sentence_iter<'a>(
        reader: R,
        max_len: Option<usize>,
    ) -> Box<dyn Iterator<Item = Result<Sentence, Error>> + 'a>
    where
        R: ReadSentence + 'a,
    {
        match max_len {
            Some(max_len) => Box::new(reader.sentences().filter_by_len(max_len)),
            None => Box::new(reader.sentences()),
        }
    }
}

impl<'a, 'ds, E, R> DataSet<'a, E> for &'ds mut ConllxDataSet<R>
where
    E: 'a + SentenceEncoder,
    E::Encoding: 'a + Clone + Eq + Hash,
    R: Read + Seek,
{
    type Iter = ConllxIter<'a, E, Box<dyn Iterator<Item = Result<Sentence, Error>> + 'ds>>;

    fn batches(
        self,
        encoder: &'a mut CategoricalEncoder<E, E::Encoding>,
        vectorizer: &'a SentVectorizer,
        batch_size: usize,
        max_len: Option<usize>,
    ) -> Fallible<Self::Iter> {
        // Rewind to the beginning of the data (if necessary).
        self.0.seek(SeekFrom::Start(0))?;

        let reader = Reader::new(BufReader::new(&mut self.0));

        Ok(ConllxIter {
            batch_size,
            encoder,
            sentences: ConllxDataSet::get_sentence_iter(reader, max_len),
            vectorizer,
        })
    }
}

pub struct ConllxIter<'a, E, I>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    I: Iterator<Item = Result<Sentence, Error>>,
{
    batch_size: usize,
    encoder: &'a mut CategoricalEncoder<E, E::Encoding>,
    vectorizer: &'a SentVectorizer,
    sentences: I,
}

impl<'a, E, I> Iterator for ConllxIter<'a, E, I>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
    I: Iterator<Item = Result<Sentence, Error>>,
{
    type Item = Fallible<TensorBuilder<LabelTensor>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch_sentences = Vec::with_capacity(self.batch_size);
        while let Some(sentence) = self.sentences.next() {
            let sentence = match sentence {
                Ok(sentence) => sentence,
                Err(err) => return Some(Err(err)),
            };
            batch_sentences.push(sentence);
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

/// Trait providing adapters for `conllx::io::Sentences`.
pub trait SentenceIter: Sized {
    fn filter_by_len(self, max_len: usize) -> LengthFilter<Self>;
}

impl<I> SentenceIter for I
where
    I: Iterator<Item = Result<Sentence, Error>>,
{
    fn filter_by_len(self, max_len: usize) -> LengthFilter<Self> {
        LengthFilter {
            inner: self,
            max_len,
        }
    }
}

/// An Iterator adapter filtering sentences by maximum length.
pub struct LengthFilter<I> {
    inner: I,
    max_len: usize,
}

impl<I> Iterator for LengthFilter<I>
where
    I: Iterator<Item = Result<Sentence, Error>>,
{
    type Item = Result<Sentence, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(sent) = self.inner.next() {
            // Treat Err as length 0 to keep our type as Result<Sentence, Error>. The iterator
            // will properly return the Error at a later point.
            let len = sent.as_ref().map(|s| s.len()).unwrap_or(0);
            if len > self.max_len {
                continue;
            }
            return Some(sent);
        }
        None
    }
}
