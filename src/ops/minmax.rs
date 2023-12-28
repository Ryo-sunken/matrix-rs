use crate::matrix::{Matrix, Axis};
use num_traits::Float;
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Float + Copy + Send + Sync,
{
    fn max_array(matrix: &Matrix<T>) -> Vec<T>
    {
        matrix.array.par_chunks(matrix.cols)
            .map(|s| s.iter().fold(T::neg_infinity(), |x, &y| x.max(y)))
            .collect::<Vec<_>>()
    }

    fn min_array(matrix: &Matrix<T>) -> Vec<T>
    {
        matrix.array.par_chunks(matrix.cols)
            .map(|s| s.iter().fold(T::infinity(), |x, &y| x.min(y)))
            .collect::<Vec<_>>()
    }

    pub fn max(&self, ax: Axis) -> Self
    {
        match ax {
            Axis::ROW => Self::from_vec_col(Self::max_array(self)),
            Axis::COLUMN => Self::from_vec_row(Self::max_array(&self.transpose())),
            Axis::BOTH => Self::new([[self.array.iter().fold(T::neg_infinity(), |x, &y| x.max(y))]]),
        }
    }

    pub fn min(&self, ax: Axis) -> Self
    {
        match ax {
            Axis::ROW => Self::from_vec_col(Self::min_array(self)),
            Axis::COLUMN => Self::from_vec_row(Self::min_array(&self.transpose())),
            Axis::BOTH => Self::new([[self.array.iter().fold(T::infinity(), |x, &y| x.min(y))]]),
        }
    }
}

