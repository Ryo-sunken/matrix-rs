use crate::Matrix;
use std::ops::{Div, DivAssign, Mul, MulAssign, Neg};

#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "rayon")]
use rayon::{iter::FromParallelIterator, prelude::*};

#[cfg(feature="rayon")]
macro_rules! defscalarmul_rayon {
    ( $( $t: ty ),+ ) => {
        $(
            impl Mul<&Matrix<$t>> for $t {
                type Output = Matrix<$t>;

                fn mul(self, rhs: &Matrix<$t>) -> Self::Output {
                    Self::Output {
                        rows: rhs.rows,
                        cols: rhs.cols,
                        array: rhs.array.par_iter().map(|&x| x * self).collect(),
                    }
                }
            }
            impl Mul<Matrix<$t>> for $t {
                type Output = Matrix<$t>;
                fn mul(self, rhs: Matrix<$t>) -> Self::Output {
                    self * &rhs
                }
            }
        )+
    };
}

#[cfg(not(feature="rayon"))]
macro_rules! defscalarmul {
    ( $( $t: ty ),+ ) => {
        $(
            impl Mul<&Matrix<$t>> for $t {
                type Output = Matrix<$t>;

                fn mul(self, rhs: &Matrix<$t>) -> Self::Output {
                    Self::Output {
                        rows: rhs.rows,
                        cols: rhs.cols,
                        array: rhs.array.iter().map(|&x| x * self).collect(),
                    }
                }
            }
            impl Mul<Matrix<$t>> for $t {
                type Output = Matrix<$t>;

                fn mul(self, rhs: Matrix<$t>) -> Self::Output {
                    self * &rhs
                }
            }
        )+
    };
}

#[cfg(feature="rayon")]
impl<T> Neg for &Matrix<T>
where 
    T: Neg<Output = T> + Copy + Send + Sync,
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

#[cfg(feature="rayon")]
impl<T> Neg for Matrix<T>
where
    T: Neg<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

#[cfg(not(feature="rayon"))]
impl<T> Neg for &Matrix<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| -x).collect(),
        }
    }
}

#[cfg(not(feature="rayon"))]
impl<T> Neg for Matrix<T>
where
    T: Neg<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

#[cfg(feature="rayon")]
defscalarmul_rayon![i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64];

#[cfg(not(feature="rayon"))]
defscalarmul![i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64];

#[cfg(feature="rayon")]
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

#[cfg(feature="rayon")]
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

#[cfg(not(feature="rayon"))]
impl<T> Mul<T> for &Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x * rhs).collect(),
        }
    }
}

#[cfg(not(feature="rayon"))]
impl<T> Mul<T> for Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}

#[cfg(feature="rayon")]
impl<T> MulAssign<T> for Matrix<T>
where
    T: Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.array = self.array.par_iter().map(|&x| x * rhs).collect();
    }
}

#[cfg(not(feature="rayon"))]
impl<T> MulAssign<T> for Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        self.array = self.array.iter().map(|&x| x * rhs).collect();
    }
}

#[cfg(feature="rayon")]
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

#[cfg(feature="rayon")]
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

#[cfg(not(feature="rayon"))]
impl<T> Div<T> for &Matrix<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        Self::Output {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x / rhs).collect(),
        }
    }
}

#[cfg(not(feature="rayon"))]
impl<T> Div<T> for Matrix<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

#[cfg(feature="rayon")]
impl<T> DivAssign<T> for Matrix<T>
where 
    T: Div<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    fn div_assign(&mut self, rhs: T) {
        self.array = self.array.par_iter().map(|&x| x / rhs).collect()
    }
}

#[cfg(not(feature="rayon"))]
impl<T> DivAssign<T> for Matrix<T>
where
    T: Div<Output = T> + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        self.array = self.array.iter().map(|&x| x / rhs).collect()
    }
}
