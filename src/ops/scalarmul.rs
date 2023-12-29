use crate::matrix::Matrix;
use rayon::prelude::*;
use std::ops::{Div, DivAssign, Mul, MulAssign, Neg};

macro_rules! defscalarmul {
    ( $( $t: ty ),+ ) => {
        $(
            impl Mul<&Matrix<$t>> for $t
            {
                type Output = Matrix<$t>;

                fn mul(self, rhs: &Matrix<$t>) -> Self::Output
                {
                    Self::Output
                    {
                        rows: rhs.rows,
                        cols: rhs.cols,
                        array: rhs.array.par_iter().map(|&x| x * self).collect(),
                    }
                }
            }
            impl Mul<Matrix<$t>> for $t
            {
                type Output = Matrix<$t>;

                fn mul(self, rhs: Matrix<$t>) -> Self::Output
                {
                    self * &rhs
                }
            }
        )+
    };
}

impl<T> Neg for &Matrix<T>
where
    T: Neg<Outout = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| -x).collect(),
        }
    }
}
impl<T> Neg for Matrix<T>
where
    T: Neg<Outout = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

defscalarmul![i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64];

impl<T> Mul<T> for &Matrix<T>
where
    T: Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x * rhs).collect(),
        }
    }
}
impl<T> Mul<T> for Matrix<T>
where
    T: Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}

impl<T> MulAssign<T> for Matrix<T>
where
    T: Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.array = self.array.par_iter().map(|&x| x * rhs).collect();
    }
}

impl<T> Div<T> for &Matrix<T>
where
    T: Div<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x / rhs).collect(),
        }
    }
}
impl<T> Div<T> for Matrix<T>
where
    T: Div<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

impl<T> DivAssign<T> for Matrix<T>
where
    T: Div<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn div_assign(&mut self, rhs: T) {
        self.array = self.array.par_iter().map(|&x| x / rhs).collect()
    }
}
