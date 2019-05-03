use std::cmp::min;

use tensorflow::Tensor;

mod labels {
    pub trait Labels {
        fn from_shape(batch_size: u64, time_steps: u64) -> Self;
    }
}

/// No labels.
#[derive(Default)]
pub struct NoLabels;

impl labels::Labels for NoLabels {
    fn from_shape(_batch_size: u64, _time_steps: u64) -> Self {
        NoLabels
    }
}

/// Labels stored in a `Tensor<i32>`.
pub struct LabelTensor(pub Tensor<i32>);

impl labels::Labels for LabelTensor {
    fn from_shape(batch_size: u64, time_steps: u64) -> Self {
        LabelTensor(Tensor::new(&[batch_size, time_steps]))
    }
}

/// Build `Tensor`s from Rust slices.
pub struct TensorBuilder<L> {
    sequence: usize,
    sequence_lens: Tensor<i32>,
    inputs: Tensor<f32>,
    labels: L,
}

impl<L> TensorBuilder<L>
where
    L: labels::Labels,
{
    /// Create a new `TensorBuilder`.
    ///
    /// Creates a new builder with the given batch size, number of time steps,
    /// and input size.
    pub fn new(batch_size: usize, time_steps: usize, inputs_size: usize) -> Self {
        TensorBuilder {
            sequence: 0,
            sequence_lens: Tensor::new(&[batch_size as u64]),
            inputs: Tensor::new(&[batch_size as u64, time_steps as u64, inputs_size as u64]),
            labels: L::from_shape(batch_size as u64, time_steps as u64),
        }
    }
}

impl<L> TensorBuilder<L> {
    /// Add an input.
    fn add_input(&mut self, input: &[f32]) {
        assert!((self.sequence as u64) < self.inputs.dims()[0]);

        let max_seq_len = self.inputs.dims()[1] as usize;
        let token_repr_size = self.inputs.dims()[2] as usize;

        // Number of time steps to copy.
        let timesteps = min(max_seq_len, input.len() / token_repr_size);
        self.sequence_lens[self.sequence] = timesteps as i32;

        let token_offset = self.sequence * max_seq_len * token_repr_size;
        let token_seq =
            &mut self.inputs[token_offset..token_offset + (token_repr_size * timesteps)];
        token_seq.copy_from_slice(&input[..token_repr_size * timesteps]);
    }

    /// Get the constructed tensors.
    ///
    /// Returns a triple of:
    ///
    /// * The input tensor.
    /// * The sequence lengths tensor.
    /// * The labels.
    pub fn into_parts(self) -> (Tensor<f32>, Tensor<i32>, L) {
        (self.inputs, self.sequence_lens, self.labels)
    }

    /// Get the inputs lengths tensor.
    pub fn inputs(&self) -> &Tensor<f32> {
        &self.inputs
    }

    /// Get the labels.
    #[allow(dead_code)]
    pub fn labels(&self) -> &L {
        &self.labels
    }

    /// Get the sequence lengths tensor.
    pub fn seq_lens(&self) -> &Tensor<i32> {
        &self.sequence_lens
    }
}

impl TensorBuilder<LabelTensor> {
    /// Add an instance with labels.
    pub fn add_with_labels(&mut self, input: &[f32], labels: &[i32]) {
        self.add_input(input);

        let max_seq_len = self.inputs.dims()[1] as usize;
        let token_repr_size = self.inputs.dims()[2] as usize;

        // Number of time steps to copy
        let timesteps = min(max_seq_len, input.len() / token_repr_size);

        let label_offset = self.sequence * max_seq_len;
        let label_seq = &mut self.labels.0[label_offset..label_offset + timesteps];
        label_seq.copy_from_slice(labels);

        self.sequence += 1;
    }
}

impl TensorBuilder<NoLabels> {
    /// Add an instance without labels.
    pub fn add_without_labels(&mut self, input: &[f32]) {
        self.add_input(input);
        self.sequence += 1;
    }
}
