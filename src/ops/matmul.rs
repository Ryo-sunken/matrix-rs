use crate::matrix::{Axis, Matrix};
use rayon::prelude::*;
use std::{
    iter::Sum,
    ops::{Add, Div, Mul},
};

impl<T> Matrix<T>
where
    T: Mul + Copy + Send + Sync,
    <T as Mul>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Mul>::Output>,
{
    pub fn cwise_mul(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .zip(rhs.array.par_iter())
                .map(|(&x, &y)| x * y)
                .collect(),
        }
    }
}

impl<T> Matrix<T>
where
    T: Div + Copy + Send + Sync,
    <T as Div>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Div>::Output>,
{
    pub fn cwise_div(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .zip(rhs.array.par_iter())
                .map(|(&x, &y)| x / y)
                .collect(),
        }
    }
}

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Sum + Mul<Output = T> + Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.cols, rhs.rows);

        // row_vector * col_vector
        if self.rows == 1 && rhs.cols == 1 {
            return Self::Output {
                rows: 1,
                cols: 1,
                array: vec![self
                    .array
                    .par_iter()
                    .zip(rhs.array.par_iter())
                    .map(|(&x, &y)| x * y)
                    .sum()],
            };
        // row_vector * matrix
        } else if self.rows == 1 {
            return Self::Output {
                rows: 1,
                cols: rhs.cols,
                array: self
                    .transpose()
                    .array
                    .par_chunks(self.cols)
                    .map(|s| s.iter().zip(rhs.array.iter()).map(|(&x, &y)| x * y).sum())
                    .collect::<Vec<_>>(),
            };
        // matrix * col_vector
        } else if rhs.cols == 1 {
            return Self::Output {
                rows: self.rows,
                cols: 1,
                array: self
                    .array
                    .par_chunks(self.rows)
                    .map(|s| s.iter().zip(rhs.array.iter()).map(|(&x, &y)| x * y).sum())
                    .collect::<Vec<_>>(),
            };
        }

        // col_vector * row_vector
        if self.cols == 1 {
            return Self::Output {
                rows: self.rows,
                cols: rhs.cols,
                array: rhs
                    .array
                    .par_iter()
                    .map(|&x| self.array.par_iter().map(move |&y| x * y))
                    .flatten()
                    .collect::<Vec<_>>(),
            };
        }

        let array = self
            .transpose()
            .array
            .par_chunks(self.rows)
            .zip(rhs.array.par_chunks(rhs.cols))
            .map(|(s, t)| {
                s.par_iter()
                    .map(|&x| t.par_iter().map(move |&y| x * y))
                    .flatten()
            })
            .flatten()
            .collect::<Vec<_>>();

        let mut ret = Self::Output {
            rows: self.cols,
            cols: self.rows * rhs.cols,
            array,
        }
        .sum(Some(Axis::COLUMN));
        ret.reshape(self.rows, rhs.cols);
        ret
    }
}
impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Sum + Mul<Output = T> + Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        &self * rhs
    }
}
impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Sum + Mul<Output = T> + Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}
impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Sum + Mul<Output = T> + Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        &self * &rhs
    }
}
