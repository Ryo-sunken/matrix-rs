use crate::{sparse::SparseMatrix, Matrix};
use std::{
    iter::Sum,
    ops::{Div, Mul},
};

#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "rayon")]
use rayon::{iter::FromParallelIterator, prelude::*};

#[cfg(feature = "rayon")]
impl<T> Matrix<T>
where
    T: Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    pub fn cwise_mul(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let threads = num_cpus::get();
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .with_min_len(self.rows * self.cols / threads)
                .zip(rhs.array.par_iter())
                .map(|(&x, &y)| x * y)
                .collect(),
        }
    }
}

#[cfg(feature = "rayon")]
impl<T> Matrix<T>
where
    T: Div<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    pub fn cwise_div(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        let threads = num_cpus::get();
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .with_min_len(self.rows * self.cols / threads)
                .zip(rhs.array.par_iter())
                .map(|(&x, &y)| x / y)
                .collect(),
        }
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Matrix<T>
where
    T: Mul<Output = T> + Copy,
{
    pub fn cwise_mul(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .zip(rhs.array.iter())
                .map(|(&x, &y)| x * y)
                .collect(),
        }
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Matrix<T>
where
    T: Div<Output = T> + Copy,
{
    pub fn cwise_div(&self, rhs: &Matrix<T>) -> Self {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .zip(rhs.array.iter())
                .map(|(&x, &y)| x / y)
                .collect(),
        }
    }
}

#[cfg(feature = "rayon")]
impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        let threads = num_cpus::get();
        // scalar * scalar
        if self.is_scalar() && rhs.is_scalar() {
            return Self::Output {
                rows: 1,
                cols: 1,
                array: vec![self.array[0] * rhs.array[0]],
            };
        // scalar * matrix
        } else if self.is_scalar() {
            return Self::Output {
                rows: rhs.rows,
                cols: rhs.cols,
                array: rhs
                    .array
                    .par_iter()
                    .with_min_len(rhs.rows * rhs.cols / threads)
                    .map(|&x| self.array[0] * x)
                    .collect(),
            };
        // matrix * scalar
        } else if rhs.is_scalar() {
            return Self::Output {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .par_iter()
                    .with_min_len(self.rows * self.cols / threads)
                    .map(|&x| x * rhs.array[0])
                    .collect(),
            };
        }

        assert_eq!(self.cols, rhs.rows);

        // row_vector * col_vector
        if self.rows == 1 && rhs.cols == 1 {
            return Self::Output {
                rows: 1,
                cols: 1,
                array: vec![self
                    .array
                    .par_iter()
                    .with_min_len(self.cols / threads)
                    .zip(rhs.array.par_iter().with_min_len(rhs.rows / threads))
                    .map(|(&x, &y)| x * y)
                    .sum()],
            };
        // row_vector * matrix
        } else if self.rows == 1 {
            return Self::Output {
                rows: 1,
                cols: rhs.cols,
                array: rhs
                    .transpose()
                    .array
                    .par_chunks(rhs.rows)
                    .map(|s| s.iter().zip(self.array.iter()).map(|(&x, &y)| x * y).sum())
                    .collect::<Vec<_>>(),
            };
        // matrix * col_vector
        } else if rhs.cols == 1 {
            return Self::Output {
                rows: self.rows,
                cols: 1,
                array: self
                    .array
                    .par_chunks(self.cols)
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
                    .with_min_len(self.rows * rhs.cols / threads)
                    .flat_map(|&x| {
                        self.array
                            .par_iter()
                            .with_min_len(self.rows * rhs.cols / threads)
                            .map(move |&y| x * y)
                    })
                    .collect(),
            };
        }

        let array = (0..self.rows)
            .into_par_iter()
            .with_min_len(self.rows / threads)
            .map(|r| {
                (0..rhs.cols)
                    .into_par_iter()
                    .with_min_len(rhs.cols / threads)
                    .map(move |c| {
                        self.array[(r * self.cols)..((r + 1) * self.cols)]
                            .iter()
                            .zip(0..self.cols)
                            .map(|(&s, oi)| s * rhs.array[oi * rhs.cols + c])
                            .sum()
                    })
            })
            .flatten()
            .collect();

        Self::Output {
            rows: self.rows,
            cols: rhs.cols,
            array,
        }
    }
}

#[cfg(feature = "rayon")]
impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        &self * rhs
    }
}

#[cfg(feature = "rayon")]
impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}

#[cfg(feature = "rayon")]
impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        &self * &rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        // scalar * scalar
        if self.is_scalar() && rhs.is_scalar() {
            return Self::Output {
                rows: 1,
                cols: 1,
                array: vec![self.array[0] * rhs.array[0]],
            };
        // scalar * matrix
        } else if self.is_scalar() {
            return Self::Output {
                rows: rhs.rows,
                cols: rhs.cols,
                array: rhs.array.iter().map(|&x| self.array[0] * x).collect(),
            };
        // matrix * scalar
        } else if rhs.is_scalar() {
            return Self::Output {
                rows: self.rows,
                cols: self.cols,
                array: self.array.iter().map(|&x| x * rhs.array[0]).collect(),
            };
        }

        assert_eq!(self.cols, rhs.rows);

        // row_vector * col_vector
        if self.rows == 1 && rhs.cols == 1 {
            return Self::Output {
                rows: 1,
                cols: 1,
                array: vec![self
                    .array
                    .iter()
                    .zip(rhs.array.iter())
                    .map(|(&x, &y)| x * y)
                    .sum()],
            };
        // row_vector * matrix
        } else if self.rows == 1 {
            return Self::Output {
                rows: 1,
                cols: rhs.cols,
                array: rhs
                    .transpose()
                    .array
                    .chunks(rhs.rows)
                    .map(|s| s.iter().zip(self.array.iter()).map(|(&x, &y)| x * y).sum())
                    .collect::<Vec<_>>(),
            };
        // matrix * col_vector
        } else if rhs.cols == 1 {
            return Self::Output {
                rows: self.rows,
                cols: 1,
                array: self
                    .array
                    .chunks(self.cols)
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
                    .iter()
                    .flat_map(|&x| self.array.iter().map(move |&y| x * y))
                    .collect(),
            };
        }

        let mut array = Vec::with_capacity(self.rows * rhs.cols);
        for r in 0..self.rows {
            for c in 0..rhs.cols {
                array.push(
                    self.array[(r * self.cols)..((r + 1) * self.cols)]
                        .iter()
                        .zip(0..self.cols)
                        .map(|(&s, oi)| s * rhs.array[oi * rhs.cols + c])
                        .sum(),
                )
            }
        }

        Self::Output {
            rows: self.rows,
            cols: rhs.cols,
            array,
        }
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<&Matrix<T>> for Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        &self * rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<Matrix<T>> for &Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<Matrix<T>> for Matrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        &self * &rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<&Matrix<T>> for &SparseMatrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(rhs.cols, 1);
        assert_eq!(self.cols, rhs.rows);

        let array = (0..self.rows)
            .into_iter()
            .map(|i| {
                (self.row_ptr[i]..self.row_ptr[i + 1])
                    .map(|j| self.val[j] * rhs.array[self.col_idx[j]])
                    .sum()
            })
            .collect();

        Self::Output {
            rows: self.rows,
            cols: 1,
            array,
        }
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<&Matrix<T>> for SparseMatrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: &Matrix<T>) -> Self::Output {
        &self * rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<Matrix<T>> for &SparseMatrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        self * &rhs
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Mul<Matrix<T>> for SparseMatrix<T>
where
    T: Sum + Mul<Output = T> + Copy,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Self::Output {
        &self * &rhs
    }
}
