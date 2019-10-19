//! Learning rate functions.

use std::f32;

/// Trait for learning rate schedules.
///
/// A learning rate schedule determines the learning rate
/// at a given epoch.
pub trait LearningRateSchedule {
    /// Compute the learning rate for an epoch.
    fn compute_epoch_learning_rate(&mut self, epoch: usize, last_score: f32) -> f32;
    /// Compute the learning rate for the current batch.
    fn compute_step_learning_rate(&mut self, global_step: usize) -> f32;
}

/// Constant learning rate schedule.
///
/// This schedule uses the same learning rate for every epoch.
pub struct ConstantLearningRate(f32);

impl ConstantLearningRate {
    /// Construct a constant learning reate.
    pub fn new(lr: f32) -> Self {
        assert!(lr > 0.0, "Learning rate must be a positive value");

        ConstantLearningRate(lr)
    }
}

impl LearningRateSchedule for ConstantLearningRate {
    fn compute_epoch_learning_rate(&mut self, _epoch: usize, _last_score: f32) -> f32 {
        self.0
    }
    fn compute_step_learning_rate(&mut self, _epoch: usize) -> f32 {
        self.0
    }
}

/// Exponential decay learning rate schedule.
///
/// This schedule starts at an initial learning rate, which decays
/// exponentionally over time. To be specific, the learning rate is
/// calculated as follows:
///
/// *lr = initial_lr * decay_rate ^ (epoch / decay_steps)*
pub struct ExponentialDecay {
    initial_lr: f32,
    lr: f32,
    decay_rate: f32,
    decay_epochs: usize,
    warmup_steps: usize,
    staircase: bool,
}

impl ExponentialDecay {
    /// Construct an exponential decay schedule.
    ///
    /// If `staircase` is true, the exponent of the decay is
    /// computed using integer division. This has the effect that
    /// the learning rate only changes every `decay_epochs` epochs.
    /// If `warmup_steps` > 0, the learning rate is linearly scaled
    /// for `warmup_steps` from 0 -> `initial_lr`.
    pub fn new(
        initial_lr: f32,
        decay_rate: f32,
        decay_epochs: usize,
        staircase: bool,
        warmup_steps: usize,
    ) -> Self {
        assert!(
            initial_lr > 0.0,
            "The initial learning rate must be a positive value."
        );
        assert!(
            decay_rate > 0.0 && decay_rate < 1.0,
            "The decay rate must be in (0, 1)."
        );
        assert!(
            decay_epochs > 0,
            "The number decay steps should be non-zero."
        );

        ExponentialDecay {
            lr: initial_lr,
            initial_lr,
            decay_rate,
            decay_epochs,
            staircase,
            warmup_steps,
        }
    }
}

impl LearningRateSchedule for ExponentialDecay {
    fn compute_step_learning_rate(&mut self, global_step: usize) -> f32 {
        if global_step < self.warmup_steps {
            (self.initial_lr / (self.warmup_steps as f32)) * global_step as f32
        } else {
            self.lr
        }
    }

    fn compute_epoch_learning_rate(&mut self, epoch: usize, _last_score: f32) -> f32 {
        let exponent = if self.staircase {
            (epoch / self.decay_epochs) as f32
        } else {
            epoch as f32 / self.decay_epochs as f32
        };

        self.lr = self.initial_lr * self.decay_rate.powf(exponent);
        self.lr
    }
}

/// Plateau learning rate schedule.
///
/// This schedule scales the learning rate by some factor when a
/// a plateau is reached in model scores.
pub struct PlateauLearningRate {
    lr: f32,
    scale: f32,
    best_score: f32,
    patience: usize,
    max_patience: usize,
    warmup_steps: usize,
}

impl PlateauLearningRate {
    /// Construct a PlateauLearningrate.
    ///
    /// `initial_lr` specifies the initial learning rate. The learning rate
    /// is scaled using `scale` when the model score does not improve for
    /// `max_patience` steps.
    /// If `warmup_steps` > 0, the learning rate is linearly scaled
    /// for `warmup_steps` from 0 -> `initial_lr`.
    pub fn new(initial_lr: f32, scale: f32, max_patience: usize, warmup_steps: usize) -> Self {
        PlateauLearningRate {
            lr: initial_lr,
            scale,
            best_score: -f32::INFINITY,
            patience: 0,
            max_patience,
            warmup_steps,
        }
    }
}

impl LearningRateSchedule for PlateauLearningRate {
    fn compute_epoch_learning_rate(&mut self, _epoch: usize, last_score: f32) -> f32 {
        if last_score > self.best_score {
            self.best_score = last_score;
            self.patience = 0;
        } else {
            self.patience += 1;

            if self.patience == self.max_patience {
                self.lr *= self.scale;
                self.patience = 0;
            }
        }

        self.lr
    }

    fn compute_step_learning_rate(&mut self, global_step: usize) -> f32 {
        if global_step < self.warmup_steps {
            (self.lr / (self.warmup_steps as f32)) * global_step as f32
        } else {
            self.lr
        }
    }
}
#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::{
        ConstantLearningRate, ExponentialDecay, LearningRateSchedule, PlateauLearningRate,
    };

    #[test]
    pub fn constant_lr() {
        let mut constant = ConstantLearningRate(0.1);
        assert_relative_eq!(constant.compute_epoch_learning_rate(0, 0.), 0.1);
        assert_relative_eq!(constant.compute_epoch_learning_rate(1, 0.), 0.1);
        assert_relative_eq!(constant.compute_epoch_learning_rate(5, 0.), 0.1);
        assert_relative_eq!(constant.compute_epoch_learning_rate(15, 0.), 0.1);
        assert_relative_eq!(constant.compute_epoch_learning_rate(25, 0.), 0.1);
    }

    #[test]
    pub fn exponential_decay_lr() {
        let mut decay1 = ExponentialDecay::new(0.1, 0.2, 10, true, 0);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(0, 0.), 0.1);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(1, 0.), 0.1);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(5, 0.), 0.1);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(15, 0.), 0.02);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(25, 0.), 0.004);

        let mut decay2 = ExponentialDecay::new(0.1, 0.2, 10, false, 0);
        assert_relative_eq!(decay2.compute_epoch_learning_rate(0, 0.), 0.1);
        assert_relative_eq!(decay2.compute_epoch_learning_rate(1, 0.), 0.085133992);
        assert_relative_eq!(decay2.compute_epoch_learning_rate(5, 0.), 0.044721359);
        assert_relative_eq!(decay2.compute_epoch_learning_rate(15, 0.), 0.008944271);
        assert_relative_eq!(decay2.compute_epoch_learning_rate(25, 0.), 0.001788854);
    }
    #[test]

    pub fn exponential_decay_lr_warmup() {
        let mut decay1 = ExponentialDecay::new(0.1, 0.2, 10, true, 5);
        assert_relative_eq!(decay1.compute_step_learning_rate(0), 0.0);
        assert_relative_eq!(decay1.compute_step_learning_rate(1), 0.02);
        assert_relative_eq!(decay1.compute_step_learning_rate(2), 0.04);
        assert_relative_eq!(decay1.compute_step_learning_rate(3), 0.06);
        assert_relative_eq!(decay1.compute_step_learning_rate(4), 0.08);
        assert_relative_eq!(decay1.compute_step_learning_rate(5), 0.1);
        assert_relative_eq!(decay1.compute_step_learning_rate(6), 0.1);

        assert_relative_eq!(decay1.compute_epoch_learning_rate(0, 0.), 0.1);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(1, 0.), 0.1);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(5, 0.), 0.1);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(15, 0.), 0.02);
        assert_relative_eq!(decay1.compute_epoch_learning_rate(25, 0.), 0.004);
    }

    #[test]
    fn plateau_lr() {
        let mut plateau = PlateauLearningRate::new(0.1, 0.5, 2, 0);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(0, 1.0), 0.1);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(1, 2.0), 0.1);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(2, 2.0), 0.1);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(3, 2.0), 0.05);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(4, 2.0), 0.05);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(5, 2.0), 0.025);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(6, 3.0), 0.025);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(6, 4.0), 0.025);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(6, 5.0), 0.025);
    }
    #[test]
    fn plateau_lr_warmup() {
        let mut plateau = PlateauLearningRate::new(0.1, 0.5, 2, 2);
        assert_relative_eq!(plateau.compute_step_learning_rate(0), 0.0,);
        assert_relative_eq!(plateau.compute_step_learning_rate(1), 0.05);
        assert_relative_eq!(plateau.compute_step_learning_rate(2), 0.1);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(0, 1.0), 0.1);
        assert_relative_eq!(plateau.compute_step_learning_rate(3), 0.1);
        assert_relative_eq!(plateau.compute_step_learning_rate(4), 0.1);
        assert_relative_eq!(plateau.compute_step_learning_rate(5), 0.1);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(1, 2.0), 0.1);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(2, 2.0), 0.1);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(3, 2.0), 0.05);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(4, 2.0), 0.05);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(5, 2.0), 0.025);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(6, 3.0), 0.025);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(6, 4.0), 0.025);
        assert_relative_eq!(plateau.compute_epoch_learning_rate(6, 5.0), 0.025);
    }
}
