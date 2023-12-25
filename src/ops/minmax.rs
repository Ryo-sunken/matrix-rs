use crate::matrix::{Matrix, Axis};
use rayon::prelude::*;

impl Matrix<f64>
{
    pub fn max(&self, ax: Axis) -> Self
    {
        let max = |x, y| if y > x { y } else { x };
        match ax {
            Axis::ROW => {
                let split_array = self.array.chunks(self.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let max_array = split_array.into_par_iter()
                    .map(|s| s.into_par_iter().reduce(|| f64::NEG_INFINITY, max))
                    .collect::<Vec<_>>();
                Self::from_vec_col(max_array)
            },
            Axis::COLUMN => {
                let split_array = self.transpose().array.chunks(self.rows)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let max_array = split_array.into_par_iter()
                    .map(|s| s.into_par_iter().reduce(|| f64::NEG_INFINITY, max))
                    .collect::<Vec<_>>();
                Self::from_vec_row(max_array)
            },
            Axis::BOTH => Self::new([[self.array.clone().into_par_iter().reduce(|| f64::NEG_INFINITY, max)]]),
        }
    }

    pub fn min(&self, ax: Axis) -> Self
    {
        let min = |x, &y| if y < x { y } else { x };
        match ax {
            Axis::ROW => {
                let split_array = self.array
                    .chunks(self.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let min_array = split_array.par_iter()
                    .map(|s| s.iter().fold(f64::INFINITY, min))
                    .collect::<Vec<_>>();
                Self::from_vec_col(min_array)
            },
            Axis::COLUMN => {
                let trans = self.transpose();
                let split_array = trans.array
                    .chunks(trans.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let min_array = split_array.par_iter()
                    .map(|s| s.iter().fold(f64::INFINITY, min))
                    .collect::<Vec<_>>();
                Self::from_vec_row(min_array)
            },
            Axis::BOTH => Self::new([[self.array.iter().fold(f64::INFINITY, min)]]),
        }
    }
}