use crate::matrix::Matrix;
use rayon::prelude::*;
use std::ops::{Sub, SubAssign};

impl<T> Sub<&Matrix<T>> for &Matrix<T>
where
    T: Sub + Copy + Send + Sync,
    <T as Sub>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Sub>::Output>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .zip(rhs.array.par_iter())
                .map(|(&x, &y)| x - y)
                .collect(),
        }
    }
}
impl<T> Sub<&Matrix<T>> for Matrix<T>
where
    T: Sub + Copy + Send + Sync,
    <T as Sub>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Sub>::Output>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        &self - rhs
    }
}
impl<T> Sub<Matrix<T>> for &Matrix<T>
where
    T: Sub + Copy + Send + Sync,
    <T as Sub>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Sub>::Output>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        self - &rhs
    }
}
impl<T> Sub<Matrix<T>> for Matrix<T>
where
    T: Sub + Copy + Send + Sync,
    <T as Sub>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Sub>::Output>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        &self - &rhs
    }
}

impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where
    T: Sub + Copy + Send + Sync,
    <T as Sub>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Sub>::Output>,
{
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        self.array = self
            .array
            .par_iter()
            .zip(rhs.array.par_iter())
            .map(|(&x, &y)| x - y)
            .collect();
    }
}
impl<T> SubAssign<Matrix<T>> for Matrix<T>
where
    T: Sub + Copy + Send + Sync,
    <T as Sub>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Sub>::Output>,
{
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        *self -= &rhs;
    }
}
