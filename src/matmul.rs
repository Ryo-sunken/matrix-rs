use crate::matrix::Matrix;
use std::ops::{Add, Mul};
use num_traits::identities::Zero;
use rayon::prelude::*;

impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Mul + Add<<T as Mul>::Output, Output = T> + Zero + Copy,
    Vec<T>: FromIterator<<T as Mul>::Output>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.cols, rhs.rows);

        let mut array = vec![T::zero(); self.rows * rhs.cols];
        for r in 0..self.rows {
            for c in 0..rhs.cols {
                array[r * rhs.cols + c] = self.array[r * self.cols..(r + 1) * self.cols]
                    .iter()
                    .zip(0..self.cols)
                    .map(|(&s, oi)| s * rhs.array[oi * rhs.cols + c])
                    .fold(T::zero(), |acc, cur| acc + cur);
            }
        }

        Self::Output
        {
            rows: self.rows,
            cols: rhs.cols,
            array,
        }
    }
}
impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Mul + Add<<T as Mul>::Output, Output = T> + Zero + Copy,
    Vec<T>: FromIterator<<T as Mul>::Output>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        &self * rhs
    }
}
impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Mul + Add<<T as Mul>::Output, Output = T> + Zero + Copy,
    Vec<T>: FromIterator<<T as Mul>::Output>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}
impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Mul + Add<<T as Mul>::Output, Output = T> + Zero + Copy,
    Vec<T>: FromIterator<<T as Mul>::Output>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        &self * &rhs
    }
}