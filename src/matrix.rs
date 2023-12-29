use rayon::prelude::*;

use std::cmp::PartialEq;

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) array: Vec<T>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum Axis {
    ROW,
    COLUMN,
}

impl<T> Matrix<T> {
    pub fn new<const R: usize, const C: usize>(data: [[T; C]; R]) -> Self {
        assert!(R != 0 && C != 0, "Rows and columns cannot be set to zero.");
        Self {
            rows: R,
            cols: C,
            array: data.into_iter().flatten().collect(),
        }
    }

    pub fn new_row_vector<const C: usize>(data: [T; C]) -> Self {
        assert!(C != 0, "Columns cannot be set to zero.");
        Self {
            rows: 1,
            cols: C,
            array: data.into_iter().collect(),
        }
    }

    pub fn new_col_vector<const R: usize>(data: [T; R]) -> Self {
        assert!(R != 0, "Rows cannot be set to zero.");
        Self {
            rows: R,
            cols: 1,
            array: data.into_iter().collect(),
        }
    }

    pub fn from_vec(array: Vec<T>, rows: usize, cols: usize) -> Self {
        assert!(
            rows != 0 && cols != 0,
            "Rows and columns cannot be set to zero."
        );
        assert_eq!(
            array.len(),
            rows * cols,
            "The array length must be equal the matrix size."
        );
        Self { rows, cols, array }
    }

    pub fn from_vec_row(array: Vec<T>) -> Self {
        let cols = array.len();
        Self::from_vec(array, 1, cols)
    }

    pub fn from_vec_col(array: Vec<T>) -> Self {
        let rows = array.len();
        Self::from_vec(array, rows, 1)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get_ref(&self, row: usize, col: usize) -> Option<&T> {
        if self.rows <= row || self.cols <= col {
            return None;
        }
        self.array.get(row * self.cols + col)
    }

    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if self.rows <= row || self.cols <= col {
            return None;
        }
        self.array.get_mut(row * self.cols + col)
    }

    pub fn is_scalar(&self) -> bool {
        self.rows == 1 && self.cols == 1
    }

    pub fn reshape(&mut self, rows: usize, cols: usize) {
        assert_eq!(
            self.rows * self.cols,
            rows * cols,
            "({}, {}) cannot reshape to ({}, {})",
            self.rows,
            self.cols,
            rows,
            cols
        );
        self.rows = rows;
        self.cols = cols;
    }
}

#[allow(dead_code)]
impl<T> Matrix<T>
where
    T: Clone,
{
    pub fn as_shape(&self, rows: usize, cols: usize) -> Self {
        assert_eq!(self.rows * self.cols, rows * cols);

        Self {
            rows,
            cols,
            array: self.array.clone(),
        }
    }

    pub fn from_slice<const C: usize>(data: &[[T; C]]) -> Self {
        assert!(!data.is_empty(), "Rows cannot be set to zero.");
        assert!(C != 0, "Columns cannot be set to zero.");
        Self {
            rows: data.len(),
            cols: C,
            array: data.iter().flatten().cloned().collect(),
        }
    }

    pub fn get(&self, r: usize, c: usize) -> Option<T> {
        if self.rows <= r || self.cols <= c {
            return None;
        }
        self.array.get(r * self.cols + c).cloned()
    }

    pub fn as_scalar(&self) -> Option<T> {
        assert!(self.is_scalar());
        self.array.get(0).cloned()
    }

    pub fn transpose(&self) -> Self {
        // scalar
        if self.rows == 1 && self.cols == 1 {
            return self.clone();
        // row_vector
        } else if self.rows == 1 {
            return Self {
                rows: self.cols,
                cols: 1,
                array: self.array.clone(),
            };
        // col_vector
        } else if self.cols == 1 {
            return Self {
                rows: 1,
                cols: self.rows,
                array: self.array.clone(),
            };
        }

        // matrix
        let mut array = Vec::with_capacity(self.rows * self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                array.push(self.array[r * self.cols + c].clone());
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            array,
        }
    }
}

// ParticalEq, Eq implementation
impl<T> PartialEq for Matrix<T>
where
    T: PartialEq + Send + Sync,
{
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.array
            .par_iter()
            .zip(other.array.par_iter())
            .all(|(x, y)| x == y)
    }
}
impl<T> Eq for Matrix<T> where T: Eq + Send + Sync {}

// Display implementation
impl<T> std::fmt::Display for Matrix<T>
where
    T: std::fmt::Display + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[")?;
        for row in self.array.chunks(self.cols) {
            writeln!(
                f,
                " [ {} ]",
                row.iter()
                    .map(|&x| format!("{:.4}", x))
                    .collect::<Vec<_>>()
                    .join(" ")
            )?;
        }
        write!(f, "]")
    }
}
