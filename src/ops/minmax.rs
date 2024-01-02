use crate::matrix::{Axis, Matrix};
use num_traits::Zero;

impl<T> Matrix<T>
where
    T: PartialOrd + Copy,
{
    fn max_array(matrix: &Matrix<T>) -> Vec<T> {
        matrix
            .array
            .chunks(matrix.cols)
            .map(|s| {
                s.iter()
                    .reduce(|x, y| if x > y { x } else { y })
                    .copied()
                    .unwrap()
            })
            .collect::<Vec<_>>()
    }

    fn min_array(matrix: &Matrix<T>) -> Vec<T> {
        matrix
            .array
            .chunks(matrix.cols)
            .map(|s| {
                s.iter()
                    .reduce(|x, y| if x < y { x } else { y })
                    .copied()
                    .unwrap()
            })
            .collect::<Vec<_>>()
    }

    pub fn max(&self, ax: Option<Axis>) -> Self {
        match ax {
            Some(Axis::ROW) => Self::from_vec_col(Self::max_array(self)),
            Some(Axis::COLUMN) => Self::from_vec_row(Self::max_array(&self.transpose())),
            None => Self::new([[Self::max_array(self)
                .iter()
                .reduce(|x, y| if x > y { x } else { y })
                .copied()
                .unwrap()]]),
        }
    }

    pub fn min(&self, ax: Option<Axis>) -> Self {
        match ax {
            Some(Axis::ROW) => Self::from_vec_col(Self::min_array(self)),
            Some(Axis::COLUMN) => Self::from_vec_row(Self::min_array(&self.transpose())),
            None => Self::new([[Self::min_array(self)
                .iter()
                .reduce(|x, y| if x < y { x } else { y })
                .copied()
                .unwrap()]]),
        }
    }

    pub fn cwise_max(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .zip(rhs.array.iter())
                .map(|(&x, &y)| if x > y { x } else { y })
                .collect(),
        }
    }

    pub fn cwise_min(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .zip(rhs.array.iter())
                .map(|(&x, &y)| if x < y { x } else { y })
                .collect(),
        }
    }
}

impl<T> Matrix<T>
where
    T: PartialOrd + Copy + Zero,
{
    pub fn relu(&self) -> Self {
        self.cwise_max(&Self::zero_like(self))
    }
}
