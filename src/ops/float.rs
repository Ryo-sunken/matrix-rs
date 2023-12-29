use crate::matrix::Matrix;
use num_traits::Float;
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Float + Send + Sync,
{
    pub fn exp(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.exp()).collect(),
        }
    }

    pub fn sin(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.sin()).collect(),
        }
    }

    pub fn cos(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.cos()).collect(),
        }
    }

    pub fn sqrt(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.sqrt()).collect(),
        }
    }

    pub fn powf(&self, n: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.powf(n)).collect(),
        }
    }
}
