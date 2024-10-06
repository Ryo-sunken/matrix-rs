use crate::{sparse::SparseMatrix, Matrix};
use num_traits::identities::{One, Zero};

impl<T> Matrix<T>
where
    T: Zero + Clone,
{
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            array: vec![T::zero(); rows * cols],
        }
    }

    pub fn zero_like<U>(mat: &Matrix<U>) -> Self {
        Self::zero(mat.rows, mat.cols)
    }

    pub fn diag(&self) -> Self {
        assert!(self.rows == 1 || self.cols == 1, "Only the vector.");
        let size = if self.rows != 1 { self.rows } else { self.cols };
        let mut ret = Matrix::<T>::zero(size, size);
        for d in 0..size {
            ret.array[d * size + d] = self.array[d].clone();
        }
        ret
    }

    pub fn diag_row(&self) -> Self {
        let mut ret = Matrix::zero(self.rows, self.rows * self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                ret[i][j + i * self.cols] = self[i][j].clone();
            }
        }
        ret
    }

    pub fn diag_col(&self) -> Self {
        self.transpose().diag_row().transpose()
    }

    pub fn to_sparse(&self) -> SparseMatrix<T> {
        let idx_val: Vec<(usize, _)> = self
            .array
            .chunks(self.cols)
            .flat_map(|row| row.iter().enumerate().filter(|(_, x)| !x.is_zero()))
            .collect();

        let mut row_ptr: Vec<_> = self
            .array
            .chunks(self.cols)
            .map(|row| row.iter().filter(|x| !x.is_zero()).count())
            .scan(0, |cum, x| {
                *cum += x;
                Some(*cum)
            })
            .collect();
        row_ptr.insert(0, 0);

        SparseMatrix::<T> {
            rows: self.rows,
            cols: self.cols,
            val: idx_val.clone().into_iter().map(|(_, x)| x.clone()).collect(),
            col_idx: idx_val.clone().into_iter().map(|(i, _)| i).collect(),
            row_ptr,
        }
    }
}

impl<T> Matrix<T>
where
    T: One + Copy,
{
    pub fn one(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            array: vec![T::one(); rows * cols],
        }
    }

    pub fn one_like<U>(mat: &Matrix<U>) -> Self {
        Self::one(mat.rows, mat.cols)
    }
}

impl<T> Matrix<T>
where
    T: One + Zero + Copy,
{
    pub fn eye(dim: usize) -> Self {
        let mut array = vec![T::zero(); dim * dim];
        for i in 0..dim {
            array[i * dim + i] = T::one();
        }

        Self {
            rows: dim,
            cols: dim,
            array,
        }
    }
}
