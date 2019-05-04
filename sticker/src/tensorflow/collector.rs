use std::hash::Hash;

use conllx::graph::Sentence;
use failure::Error;
use ndarray::{Ix1, Ix2, Ix3};
use ndarray_tensorflow::NdTensor;

use crate::{Collector, Numberer, SentVectorizer, SentenceEncoder};

use super::tensor::{LabelTensor, TensorBuilder};

pub struct CollectedTensors {
    pub sequence_lens: Vec<NdTensor<i32, Ix1>>,
    pub inputs: Vec<NdTensor<f32, Ix3>>,
    pub labels: Vec<NdTensor<i32, Ix2>>,
}

pub struct TensorCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Eq + Hash,
{
    encoder: E,
    numberer: Numberer<E::Encoding>,
    vectorizer: SentVectorizer,
    batch_size: usize,
    sequence_lens: Vec<NdTensor<i32, Ix1>>,
    inputs: Vec<NdTensor<f32, Ix3>>,
    labels: Vec<NdTensor<i32, Ix2>>,
    cur_labels: Vec<Vec<i32>>,
    cur_inputs: Vec<Vec<f32>>,
}

impl<E> TensorCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Eq + Hash,
{
    pub fn new(
        batch_size: usize,
        encoder: E,
        numberer: Numberer<E::Encoding>,
        vectorizer: SentVectorizer,
    ) -> Self {
        TensorCollector {
            batch_size,
            encoder,
            numberer,
            vectorizer,
            labels: Vec::new(),
            inputs: Vec::new(),
            sequence_lens: Vec::new(),
            cur_labels: Vec::new(),
            cur_inputs: Vec::new(),
        }
    }

    fn finalize_batch(&mut self) {
        if self.cur_labels.is_empty() {
            return;
        }

        let batch_size = self.cur_labels.len();
        let max_seq_len = self.cur_labels.iter().map(Vec::len).max().unwrap_or(0);
        let inputs_dims = self.cur_inputs[0].len() / self.cur_labels[0].len();

        let mut builder: TensorBuilder<LabelTensor> =
            TensorBuilder::new(batch_size, max_seq_len, inputs_dims);

        for (inputs, labels) in self.cur_inputs.iter().zip(&self.cur_labels) {
            builder.add_with_labels(inputs, labels);
        }

        let (batch_inputs, batch_seq_lens, batch_labels) = builder.into_parts();

        self.sequence_lens.push(batch_seq_lens);
        self.inputs.push(batch_inputs);
        self.labels.push(batch_labels.0);

        self.cur_inputs.clear();
        self.cur_labels.clear();
    }

    pub fn into_parts(mut self) -> CollectedTensors {
        self.finalize_batch();

        CollectedTensors {
            sequence_lens: self.sequence_lens,
            inputs: self.inputs,
            labels: self.labels,
        }
    }
}

impl<E> Collector for TensorCollector<E>
where
    E: SentenceEncoder,
    E::Encoding: Clone + Eq + Hash,
{
    fn collect(&mut self, sentence: &Sentence) -> Result<(), Error> {
        if self.cur_labels.len() == self.batch_size {
            self.finalize_batch();
        }

        let input = self.vectorizer.realize(sentence)?;
        let mut labels = Vec::with_capacity(sentence.len());

        for encoding in self.encoder.encode(&sentence)? {
            labels.push(self.numberer.add(encoding) as i32);
        }

        self.cur_inputs.push(input);
        self.cur_labels.push(labels);

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
