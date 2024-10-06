use crate::Matrix;
use std::ops::{Sub, SubAssign};

#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "rayon")]
use rayon::{iter::FromParallelIterator, prelude::*};

#[cfg(feature = "rayon")]
impl<T> Sub<&Matrix<T>> for &Matrix<T>
where
    T: Sub<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let threads = num_cpus::get();
        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .with_min_len(self.rows * self.cols / threads)
                .zip(rhs.array.par_iter())
                .map(|(&x, &y)| x - y)
                .collect(),
        }
    }
}

#[cfg(feature = "rayon")]
impl<T> Sub<&Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        &self - rhs
    }
}

#[cfg(feature = "rayon")]
impl<T> Sub<Matrix<T>> for &Matrix<T>
where
    T: Sub<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        self - &rhs
    }
}

#[cfg(feature = "rayon")]
impl<T> Sub<Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        &self - &rhs
    }
}

#[cfg(not(feature = "rayon"))]
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

#[cfg(not(feature = "rayon"))]
impl<T> Sub<&Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: &Matrix<T>) -> Self::Output {
        &self - rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Sub<Matrix<T>> for &Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        self - &rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Sub<Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn sub(self, rhs: Matrix<T>) -> Self::Output {
        &self - &rhs
    }
}

#[cfg(feature = "rayon")]
impl<T> SubAssign<&Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let threads = num_cpus::get();
        self.array = self
            .array
            .par_iter()
            .with_min_len(self.rows * self.cols / threads)
            .zip(rhs.array.par_iter())
            .map(|(&x, &y)| x - y)
            .collect();
    }
}

#[cfg(feature = "rayon")]
impl<T> SubAssign<Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        *self -= &rhs
    }
}

#[cfg(not(feature = "rayon"))]
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

#[cfg(not(feature = "rayon"))]
impl<T> SubAssign<Matrix<T>> for Matrix<T>
where
    T: Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        *self -= &rhs;
    }
}
