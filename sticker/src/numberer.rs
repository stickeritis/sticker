use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;

use serde_derive::{Deserialize, Serialize};

/// Numberer for categorical values, such as features or class labels.
#[derive(Eq, PartialEq, Serialize, Deserialize)]
pub struct Numberer<T>
where
    T: Eq + Hash,
{
    values: Vec<T>,
    numbers: HashMap<T, usize>,
    start_at: usize,
}

impl<T> Numberer<T>
where
    T: Clone + Eq + Hash,
{
    pub fn new(start_at: usize) -> Self {
        Numberer {
            values: Vec::new(),
            numbers: HashMap::new(),
            start_at,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn len(&self) -> usize {
        self.values.len() + self.start_at
    }

    /// Add an value. If the value has already been encountered before,
    /// the corresponding number is returned.
    pub fn add(&mut self, value: T) -> usize {
        match self.numbers.entry(value.clone()) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let number = self.values.len() + self.start_at;
                self.values.push(value);
                e.insert(number);
                number
            }
        }
    }

    /// Return the number for a value.
    pub fn number(&self, item: &T) -> Option<usize> {
        self.numbers.get(item).cloned()
    }

    /// Return the value for a number.
    pub fn value(&self, number: usize) -> Option<&T> {
        self.values.get(number - self.start_at)
    }
}
