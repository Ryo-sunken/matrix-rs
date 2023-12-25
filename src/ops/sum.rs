use crate::matrix::{Matrix, Axis};
use std::iter::Sum;
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Sum + for<'a> Sum<&'a T> + Copy + Send + Sync,
{
    pub fn sum(&self, ax: Axis) -> Self
    {
        match ax {
            Axis::ROW => {
                let sum_array = self.array.par_chunks(self.cols)
                    .map(|s| s.iter().sum())
                    .collect::<Vec<T>>();
                Self::from_vec_col(sum_array)
            }
            Axis::COLUMN => {
                let sum_array = self.transpose().array.par_chunks(self.rows)
                    .map(|s| s.iter().sum())
                    .collect::<Vec<T>>();
                Self::from_vec_row(sum_array)
            }
            Axis::BOTH => Self::new([[self.array.par_iter().sum()]])
        }
    }
}