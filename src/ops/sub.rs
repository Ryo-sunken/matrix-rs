use crate::matrix::Matrix;
use std::ops::{Sub, SubAssign};

impl<T> Sub<&Matrix<T>> for &Matrix<T>
where
    T: Sub<Output = T> + Copy,
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
                .iter()
                .zip(rhs.array.iter())
                .map(|(&x, &y)| x - y)
                .collect(),
        }
    }
}
impl<T> Sub<&Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        &self - rhs
    }
}
impl<T> Sub<Matrix<T>> for &Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        self - &rhs
    }
}
impl<T> Sub<Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        &self - &rhs
    }
}

impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        self.array = self
            .array
            .iter()
            .zip(rhs.array.iter())
            .map(|(&x, &y)| x - y)
            .collect();
    }
}
impl<T> SubAssign<Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        *self -= &rhs;
    }
}
