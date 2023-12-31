use crate::matrix::{Axis, Matrix};

impl<T> Matrix<T>
where
    T: PartialOrd + Clone,
{
    fn max_array(matrix: &Matrix<T>) -> Vec<T> {
        matrix
            .array
            .chunks(matrix.cols)
            .map(|s| {
                s.iter()
                    .reduce(|x, y| if x > y { x } else { y })
                    .unwrap()
                    .clone()
            })
            .collect::<Vec<_>>()
    }

    fn min_array(matrix: &Matrix<T>) -> Vec<T> {
        matrix
            .array
            .chunks(matrix.cols)
            .map(|s| {
                s.iter()
                    .reduce(|x, y| if x > y { y } else { x })
                    .unwrap()
                    .clone()
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
                .unwrap()
                .clone()]]),
        }
    }

    pub fn min(&self, ax: Option<Axis>) -> Self {
        match ax {
            Some(Axis::ROW) => Self::from_vec_col(Self::min_array(self)),
            Some(Axis::COLUMN) => Self::from_vec_row(Self::min_array(&self.transpose())),
            None => Self::new([[Self::min_array(self)
                .iter()
                .reduce(|x, y| if x > y { y } else { x })
                .unwrap()
                .clone()]]),
        }
    }
}
