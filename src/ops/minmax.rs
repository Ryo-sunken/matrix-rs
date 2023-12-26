use crate::matrix::{Matrix, Axis};
use rayon::prelude::*;

impl Matrix<f64>
{
    pub fn max(&self, ax: Axis) -> Self
    {
        let max = |x, &y| if y > x { y } else { x };
        match ax {
            Axis::ROW => {
                let max_array = self.array.par_chunks(self.cols)
                    .map(|s| s.iter().fold(f64::NEG_INFINITY, max))
                    .collect::<Vec<_>>();
                Self::from_vec_col(max_array)
            },
            Axis::COLUMN => {
                let max_array = self.transpose().array.par_chunks(self.rows)
                    .map(|s| s.iter().fold(f64::NEG_INFINITY, max))
                    .collect::<Vec<_>>();
                Self::from_vec_row(max_array)
            },
            Axis::BOTH => Self::new([[self.array.iter().fold(f64::NEG_INFINITY, max)]]),
        }
    }

    pub fn min(&self, ax: Axis) -> Self
    {
        let min = |x, &y| if y < x { y } else { x };
        match ax {
            Axis::ROW => {
                let min_array = self.array.par_chunks(self.cols)
                    .map(|s| s.iter().fold(f64::INFINITY, min))
                    .collect::<Vec<_>>();
                Self::from_vec_col(min_array)
            },
            Axis::COLUMN => {
                let min_array = self.transpose().array.par_chunks(self.rows)
                    .map(|s| s.iter().fold(f64::INFINITY, min))
                    .collect::<Vec<_>>();
                Self::from_vec_row(min_array)
            },
            Axis::BOTH => Self::new([[self.array.iter().fold(f64::INFINITY, min)]]),
        }
    }
}