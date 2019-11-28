use std::cmp::min;

use ndarray::{s, Array1, Array2, Ix1, Ix2, Ix3};
use ndarray_tensorflow::NdTensor;

use crate::InputVector;

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

pub type LabelTensor = NdTensor<i32, Ix2>;

impl labels::Labels for LabelTensor {
    fn from_shape(batch_size: usize, time_steps: usize) -> Self {
        NdTensor::zeros([batch_size, time_steps])
    }
}

/// Build `Tensor`s from Rust slices.
pub struct TensorBuilder<L> {
    sequence: usize,
    seq_lens: NdTensor<i32, Ix1>,
    inputs: NdTensor<f32, Ix3>,
    subwords: Option<NdTensor<String, Ix2>>,
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
    pub fn new(
        batch_size: usize,
        time_steps: usize,
        inputs_size: usize,
        use_subwords: bool,
    ) -> Self {
        let subwords = if use_subwords {
            Some(NdTensor::zeros([batch_size, time_steps]))
        } else {
            None
        };

        TensorBuilder {
            sequence: 0,
            seq_lens: NdTensor::zeros([batch_size]),
            inputs: NdTensor::zeros([batch_size, time_steps, inputs_size]),
            subwords,
            labels: L::from_shape(batch_size, time_steps),
        }
    }
}

impl<L> TensorBuilder<L> {
    /// Add an input.
    fn add_input(&mut self, input_vector: InputVector) {
        assert!(self.sequence < self.inputs.view().shape()[0]);

        let token_repr_size = self.inputs.view().shape()[2];
        let input_timesteps = input_vector.sequence.len() / token_repr_size;

        let input =
            Array2::from_shape_vec([input_timesteps, token_repr_size], input_vector.sequence)
                .unwrap();

        let subwords = input_vector.subwords.map(Array1::from);

        let timesteps = min(self.inputs.view().shape()[1], input.shape()[0]);

        self.seq_lens.view_mut()[self.sequence] = timesteps as i32;

        #[allow(clippy::deref_addrof)]
        self.inputs
            .view_mut()
            .slice_mut(s![self.sequence, 0..timesteps, ..])
            .assign(&input.slice(s![0..timesteps, ..]));

        if let Some(subwords) = subwords {
            #[allow(clippy::deref_addrof)]
            self.subwords
                .as_mut()
                .unwrap()
                .view_mut()
                .slice_mut(s![self.sequence, 0..timesteps])
                .assign(&subwords.slice(s![..timesteps]));
        }
    }

    /// Get the constructed tensors.
    ///
    /// Returns a triple of:
    ///
    /// * The input tensor.
    /// * The sequence lengths tensor.
    /// * The labels.
    pub fn into_parts(self) -> Tensors<L> {
        Tensors {
            inputs: self.inputs,
            subwords: self.subwords,
            seq_lens: self.seq_lens,
            labels: self.labels,
        }
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
        &self.seq_lens
    }

    /// Get subwords tensor.
    pub fn subwords(&self) -> Option<&NdTensor<String, Ix2>> {
        self.subwords.as_ref()
    }
}

impl TensorBuilder<LabelTensor> {
    /// Add an instance with labels.
    pub fn add_with_labels(&mut self, input_vector: InputVector, labels: &[i32]) {
        // Number of sequence time steps.
        let token_repr_size = self.inputs.view().shape()[2] as usize;
        let timesteps = min(
            self.inputs.view().shape()[1],
            input_vector.sequence.len() / token_repr_size,
        );

        self.add_input(input_vector);

        #[allow(clippy::deref_addrof)]
        self.labels
            .view_mut()
            .slice_mut(s![self.sequence, 0..timesteps])
            .assign(&labels.into());

        self.sequence += 1;
    }
}

impl TensorBuilder<NoLabels> {
    /// Add an instance without labels.
    pub fn add_without_labels(&mut self, input_vector: InputVector) {
        self.add_input(input_vector);
        self.sequence += 1;
    }
}

/// Tensors constructed by `TensorBuilder`.
pub struct Tensors<L> {
    pub inputs: NdTensor<f32, Ix3>,
    pub subwords: Option<NdTensor<String, Ix2>>,
    pub seq_lens: NdTensor<i32, Ix1>,
    pub labels: L,
}
