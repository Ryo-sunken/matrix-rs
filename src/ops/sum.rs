use crate::{Axis, Matrix};
use std::iter::Sum;

impl<T> Matrix<T>
where
    T: Sum + Copy,
{
    fn sum_array(matrix: &Matrix<T>) -> Vec<T> {
        matrix
            .array
            .chunks(matrix.cols)
            .map(|s| s.iter().copied().sum())
            .collect::<Vec<_>>()
    }

    pub fn sum(&self, ax: Option<Axis>) -> Self {
        match ax {
            Some(Axis::ROW) => Self::from_vec_col(Self::sum_array(self)),
            Some(Axis::COLUMN) => Self::from_vec_row(Self::sum_array(&self.transpose())),
            None => Self::new([[self.array.iter().copied().sum()]]),
        }
    }
}
