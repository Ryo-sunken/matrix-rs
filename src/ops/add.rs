use crate::matrix::Matrix;
use rayon::prelude::*;
use std::ops::{Add, AddAssign};

impl<T> Add<&Matrix<T>> for &Matrix<T>
where
    T: Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .zip(rhs.array.iter())
                .map(|(&x, &y)| x + y)
                .collect(),
        }
    }
}
impl<T> Add<&Matrix<T>> for Matrix<T>
where
    T: Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        &self + rhs
    }
}
impl<T> Add<Matrix<T>> for &Matrix<T>
where
    T: Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        self + &rhs
    }
}
impl<T> Add<Matrix<T>> for Matrix<T>
where
    T: Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<T> AddAssign<&Matrix<T>> for Matrix<T>
where
    T: Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        self.array = self
            .array
            .par_iter()
            .zip(rhs.array.par_iter())
            .map(|(&x, &y)| x + y)
            .collect();
    }
}
impl<T> AddAssign<Matrix<T>> for Matrix<T>
where
    T: Add<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn add_assign(&mut self, rhs: Matrix<T>) {
        *self += &rhs;
    }
}
