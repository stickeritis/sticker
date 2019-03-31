use conllx::Token;
use failure::{format_err, Error};
use tensorflow::Tensor;

use crate::{Collector, Layer, LayerValue, Numberer, SentVectorizer};

pub struct CollectedTensors {
    pub sequence_lens: Vec<Tensor<i32>>,
    pub tokens: Vec<Tensor<f32>>,
    pub labels: Vec<Tensor<i32>>,
}

pub struct TensorCollector {
    numberer: Numberer<String>,
    vectorizer: SentVectorizer,
    layer: Layer,
    batch_size: usize,
    sequence_lens: Vec<Tensor<i32>>,
    tokens: Vec<Tensor<f32>>,
    labels: Vec<Tensor<i32>>,
    cur_labels: Vec<Vec<i32>>,
    cur_tokens: Vec<Vec<f32>>,
}

impl TensorCollector {
    pub fn new(
        batch_size: usize,
        layer: Layer,
        numberer: Numberer<String>,
        vectorizer: SentVectorizer,
    ) -> Self {
        TensorCollector {
            batch_size,
            layer,
            numberer,
            vectorizer,
            labels: Vec::new(),
            tokens: Vec::new(),
            sequence_lens: Vec::new(),
            cur_labels: Vec::new(),
            cur_tokens: Vec::new(),
        }
    }

    fn finalize_batch(&mut self) {
        if self.cur_labels.is_empty() {
            return;
        }

        let batch_size = self.cur_labels.len();

        let mut batch_seq_lens = Tensor::new(&[batch_size as u64]);
        self.cur_labels
            .iter()
            .enumerate()
            .for_each(|(idx, labels)| batch_seq_lens[idx] = labels.len() as i32);

        let max_seq_len = self.cur_labels.iter().map(Vec::len).max().unwrap_or(0);
        let token_dims = self.cur_tokens[0].len() / self.cur_labels[0].len();

        let mut batch_tokens =
            Tensor::new(&[batch_size as u64, max_seq_len as u64, token_dims as u64]);
        let mut batch_labels = Tensor::new(&[batch_size as u64, max_seq_len as u64]);

        for i in 0..batch_size {
            let offset = i * max_seq_len;
            let token_offset = offset * token_dims;
            let seq_len = self.cur_labels[i].len();

            batch_tokens[token_offset..token_offset + token_dims * seq_len]
                .copy_from_slice(&self.cur_tokens[i]);
            batch_labels[offset..offset + seq_len].copy_from_slice(&self.cur_labels[i]);
        }

        self.sequence_lens.push(batch_seq_lens);
        self.tokens.push(batch_tokens);
        self.labels.push(batch_labels);

        self.cur_tokens.clear();
        self.cur_labels.clear();
    }

    pub fn into_parts(mut self) -> CollectedTensors {
        self.finalize_batch();

        CollectedTensors {
            sequence_lens: self.sequence_lens,
            tokens: self.tokens,
            labels: self.labels,
        }
    }
}

impl Collector for TensorCollector {
    fn collect(&mut self, sentence: &[Token]) -> Result<(), Error> {
        if self.cur_labels.len() == self.batch_size {
            self.finalize_batch();
        }

        let input = self.vectorizer.realize(sentence)?;
        let mut labels = Vec::with_capacity(sentence.len());
        for token in sentence {
            let label = token
                .value(&self.layer)
                .ok_or_else(|| format_err!("Token without a tag: {}", token))?;
            labels.push(self.numberer.add(label.to_owned()) as i32);
        }

        let tokens = input.into_inner();
        self.cur_tokens.push(tokens);
        self.cur_labels.push(labels);

        Ok(())
    }

    fn vectorizer(&self) -> &SentVectorizer {
        &self.vectorizer
    }
}
