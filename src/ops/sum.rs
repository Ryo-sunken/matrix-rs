use crate::matrix::{Axis, Matrix};
use rayon::prelude::*;
use std::iter::Sum;

impl<T> Matrix<T>
where
    T: Sum + for<'a> Sum<&'a T> + Copy + Send + Sync,
{
    fn sum_array(matrix: &Matrix<T>) -> Vec<T> {
        matrix
            .array
            .par_chunks(matrix.cols)
            .map(|s| s.iter().sum())
            .collect::<Vec<_>>()
    }

    pub fn sum(&self, ax: Option<Axis>) -> Self {
        match ax {
            Some(Axis::ROW) => Self::from_vec_col(Self::sum_array(self)),
            Some(Axis::COLUMN) => Self::from_vec_row(Self::sum_array(&self.transpose())),
            None => Self::new([[self.array.par_iter().sum()]]),
        }
    }
}
