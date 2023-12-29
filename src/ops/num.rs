use crate::matrix::Matrix;
use num_traits::identities::{One, Zero};

impl<T> Matrix<T>
where
    T: Zero + Clone,
{
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            array: vec![T::zero(); rows * cols],
        }
    }

    pub fn zero_like<U>(mat: &Matrix<U>) -> Self {
        Self::zero(mat.rows, mat.cols)
    }

    pub fn diag<const D: usize>(data: [T; D]) -> Self {
        let mut array = vec![T::zero(); D * D];
        for d in 0..D {
            array[d * D + d] = data[d].clone();
        }

        Self {
            rows: D,
            cols: D,
            array,
        }
    }
}

impl<T> Matrix<T>
where
    T: One + Clone,
{
    pub fn one(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            array: vec![T::one(); rows * cols],
        }
    }

    pub fn one_like<U>(mat: &Matrix<U>) -> Self {
        Self::one(mat.rows, mat.cols)
    }
}

impl<T> Matrix<T>
where
    T: One + Zero + Clone,
{
    pub fn eye(dim: usize) -> Self {
        let mut array = vec![T::zero(); dim * dim];
        for i in 0..dim {
            array[i * dim + i] = T::one();
        }

        Self {
            rows: dim,
            cols: dim,
            array,
        }
    }
}
