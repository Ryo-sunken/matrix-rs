use rayon::prelude::*;

use std::cmp::PartialEq;

#[derive(Clone, Debug)]
pub struct RowVector<T> {
    pub(crate) dim: usize,
    pub(crate) array: Vec<T>,
}

#[derive(Clone, Debug)]
pub struct ColVector<T> {
    pub(crate) dim: usize,
    pub(crate) array: Vec<T>,
}

impl<T> RowVector<T> {
    pub fn new<const C: usize>(data: [T; C]) -> Self {
        assert!(C != 0, "Columns cannot be set to zero.");
        Self {
            dim: C,
            array: data.into_iter().collect(),
        }
    }

    pub fn from_vec(array: Vec<T>) -> Self {
        assert!(!array.is_empty(), "Array cannot be empty.");
        Self {
            dim: array.len(),
            array,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_ref(&self, idx: usize) -> Option<&T> {
        if self.dim <= idx {
            return None;
        }
        self.array.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        if self.dim <= idx {
            return None;
        }
        self.array.get_mut(idx)
    }
}

impl<T> ColVector<T> {
    pub fn new<const R: usize>(data: [T; R]) -> Self {
        assert!(R != 0, "Rows cannot be set to zero.");
        Self {
            dim: R,
            array: data.into_iter().collect(),
        }
    }

    pub fn from_vec(array: Vec<T>) -> Self {
        assert!(!array.is_empty(), "Array cannot be empty.");
        Self {
            dim: array.len(),
            array,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn get_ref(&self, idx: usize) -> Option<&T> {
        if self.dim <= idx {
            return None;
        }
        self.array.get(idx)
    }

    pub fn get_mut(&mut self, idx: usize) -> Option<&mut T> {
        if self.dim <= idx {
            return None;
        }
        self.array.get_mut(idx)
    }
}
