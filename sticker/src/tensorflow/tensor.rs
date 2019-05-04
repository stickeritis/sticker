use std::cmp::min;

use ndarray::{s, ArrayView2, Ix1, Ix2, Ix3};
use ndarray_tensorflow::NdTensor;

mod labels {
    pub trait Labels {
        fn from_shape(batch_size: usize, time_steps: usize) -> Self;
    }
}

/// No labels.
#[derive(Default)]
pub struct NoLabels;

impl labels::Labels for NoLabels {
    fn from_shape(_batch_size: usize, _time_steps: usize) -> Self {
        NoLabels
    }
}

/// Labels stored in a `Tensor<i32>`.
pub struct LabelTensor(pub NdTensor<i32, Ix2>);

impl labels::Labels for LabelTensor {
    fn from_shape(batch_size: usize, time_steps: usize) -> Self {
        LabelTensor(NdTensor::zeros([batch_size, time_steps]))
    }
}

/// Build `Tensor`s from Rust slices.
pub struct TensorBuilder<L> {
    sequence: usize,
    sequence_lens: NdTensor<i32, Ix1>,
    inputs: NdTensor<f32, Ix3>,
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
            sequence_lens: NdTensor::zeros([batch_size]),
            inputs: NdTensor::zeros([batch_size, time_steps, inputs_size]),
            labels: L::from_shape(batch_size, time_steps),
        }
    }
}

impl<L> TensorBuilder<L> {
    /// Add an input.
    fn add_input(&mut self, input: &[f32]) {
        assert!(self.sequence < self.inputs.view().shape()[0]);

        let token_repr_size = self.inputs.view().shape()[2];

        let input = ArrayView2::from_shape([input.len() / token_repr_size, token_repr_size], input)
            .unwrap();

        let timesteps = min(self.inputs.view().shape()[1], input.shape()[0]);

        self.sequence_lens.view_mut()[self.sequence] = timesteps as i32;

        #[allow(clippy::deref_addrof)]
        self.inputs
            .view_mut()
            .slice_mut(s![self.sequence, 0..timesteps, ..])
            .assign(&input.slice(s![0..timesteps, ..]));
    }

    /// Get the constructed tensors.
    ///
    /// Returns a triple of:
    ///
    /// * The input tensor.
    /// * The sequence lengths tensor.
    /// * The labels.
    pub fn into_parts(self) -> (NdTensor<f32, Ix3>, NdTensor<i32, Ix1>, L) {
        (self.inputs, self.sequence_lens, self.labels)
    }

    /// Get the inputs lengths tensor.
    pub fn inputs(&self) -> &NdTensor<f32, Ix3> {
        &self.inputs
    }

    /// Get the labels.
    #[allow(dead_code)]
    pub fn labels(&self) -> &L {
        &self.labels
    }

    /// Get the sequence lengths tensor.
    pub fn seq_lens(&self) -> &NdTensor<i32, Ix1> {
        &self.sequence_lens
    }
}

impl TensorBuilder<LabelTensor> {
    /// Add an instance with labels.
    pub fn add_with_labels(&mut self, input: &[f32], labels: &[i32]) {
        self.add_input(input);

        let token_repr_size = self.inputs.view().shape()[2] as usize;

        // Number of time steps to copy
        let timesteps = min(self.inputs.view().shape()[1], input.len() / token_repr_size);
        #[allow(clippy::deref_addrof)]
        self.labels
            .0
            .view_mut()
            .slice_mut(s![self.sequence, 0..timesteps])
            .assign(&labels.into());

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
