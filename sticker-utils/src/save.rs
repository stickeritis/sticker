use failure::Fallible;

use sticker::tensorflow::TaggerTrainer;

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum CompletedUnit {
    /// A batch is completed.
    Batch,

    /// An epoch is completed.
    Epoch,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum SaveSchedule {
    /// Save after every N batches.
    Batches(usize),

    /// Save after every epoch.
    Epoch,

    /// Save every epoch and after every N batches.
    EpochAndBatches(usize),
}

impl SaveSchedule {
    /// Create a scheduler from the schedule.
    pub fn to_save_scheduler(self, prefix: impl Into<String>) -> SaveScheduler {
        SaveScheduler {
            prefix: prefix.into(),
            batch: 0,
            epoch: 0,
            epoch_batch: 0,
            schedule: self,
        }
    }
}

/// Scheduler that saves at points dictated by the schedule.
pub struct SaveScheduler {
    prefix: String,
    epoch_batch: usize,
    epoch: usize,
    batch: usize,
    schedule: SaveSchedule,
}

impl SaveScheduler {
    /// Current batch.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Current epoch.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Save the model parameters when a save point has been reached.
    pub fn save(&mut self, trainer: &TaggerTrainer, completed: CompletedUnit) -> Fallible<()> {
        match completed {
            CompletedUnit::Epoch => {
                match self.schedule {
                    SaveSchedule::Epoch | SaveSchedule::EpochAndBatches(_) => {
                        trainer.save(format!("{}epoch-{}", self.prefix, self.epoch))?
                    }
                    SaveSchedule::Batches(_) => (),
                }

                self.epoch += 1;
                self.epoch_batch = 0;
            }
            CompletedUnit::Batch => {
                match self.schedule {
                    SaveSchedule::Batches(batches) | SaveSchedule::EpochAndBatches(batches) => {
                        if (self.batch + 1) % batches == 0 {
                            trainer.save(format!(
                                "{}epoch-{}-batch-{}",
                                self.prefix, self.epoch, self.epoch_batch
                            ))?
                        }
                    }
                    SaveSchedule::Epoch => (),
                }

                self.batch += 1;
                self.epoch_batch += 1;
            }
        }

        Ok(())
    }
}
