use crate::matrix::{Matrix, Axis};
use std::iter::Sum;
use rayon::prelude::*;

impl<T> Matrix<T>
where
    T: Sum + Copy + Send + Sync,
{
    pub fn sum(&self, ax: Axis) -> Self
    {
        match ax {
            Axis::ROW => {
                let split_array = self.array.chunks(self.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<T>>>();
                let sum_array = split_array.into_par_iter()
                    .map(|s| s.into_iter().sum())
                    .collect::<Vec<T>>();
                Self::from_vec_col(sum_array)
            }
            Axis::COLUMN => {
                let split_array = self.transpose().array.chunks(self.rows)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<T>>>();
                let sum_array = split_array.into_par_iter()
                    .map(|s| s.into_iter().sum())
                    .collect::<Vec<T>>();
                Self::from_vec_row(sum_array)
            }
            Axis::BOTH => Self::new([[self.array.clone().into_par_iter().sum()]])
        }
    }
}