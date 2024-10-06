pub mod ops;
pub mod rand;
pub mod sparse;
pub mod tensor;

use std::cmp::PartialEq;

#[derive(Clone, Debug)]
pub struct Matrix<T> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) array: Vec<T>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
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

    pub fn from_vec2d(array: Vec<Vec<T>>, rows: usize, cols: usize) -> Self {
        assert!(
            rows != 0 && cols != 0,
            "Rows and columns cannot be set to zero."
        );
        assert_eq!(
            array.iter().flatten().collect::<Vec<_>>().len(),
            rows * cols,
            "The array length must be equal the matrix size."
        );
        Self {
            rows,
            cols,
            array: array.into_iter().flatten().collect(),
        }
    }

    pub fn from_vec_row(array: Vec<T>) -> Self {
        let cols = array.len();
        Self::from_vec(array, 1, cols)
    }

    pub fn from_vec_col(array: Vec<T>) -> Self {
        let rows = array.len();
        Self::from_vec(array, rows, 1)
    }

    pub fn to_slice(&self) -> &[T] {
        &self.array
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

    pub fn concat(&self, rhs: &Matrix<T>, ax: Axis) -> Self {
        match ax {
            Axis::ROW => {
                assert_eq!(self.cols, rhs.cols);
                Self {
                    rows: self.rows + rhs.rows,
                    cols: self.cols,
                    array: [&self.array[..], &rhs.array[..]].concat(),
                }
            }
            Axis::COLUMN => {
                assert_eq!(self.rows, rhs.rows);
                Self {
                    rows: self.rows,
                    cols: self.cols + rhs.cols,
                    array: self
                        .array
                        .chunks(self.rows)
                        .zip(rhs.array.chunks(rhs.rows))
                        .map(|(s, t)| [s, t].concat())
                        .flatten()
                        .collect(),
                }
            }
        }
    }
}

// ParticalEq, Eq implementation
impl<T> PartialEq for Matrix<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.array
            .iter()
            .zip(other.array.iter())
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

#[cfg(test)]
mod tests {
    use crate::{sparse::SparseMatrix, Axis, Matrix};

    #[test]
    fn to_slice() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(x.to_slice(), &[1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    }

    #[test]
    fn reshape() {
        let mut x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        x.reshape(1, 9);
        assert_eq!(
            x,
            Matrix::new_row_vector([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        );
    }

    #[test]
    fn transpose() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            x.transpose(),
            Matrix::new([[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]])
        );
    }

    #[test]
    fn neg() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            -x,
            Matrix::new([[-1., -2., -3.], [-4., -5., -6.], [-7., -8., -9.]])
        );
    }

    #[test]
    fn add() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);
        assert_eq!(
            &x + &y,
            Matrix::new([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
        );
        assert_eq!(
            x.clone() + &y,
            Matrix::new([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
        );
        assert_eq!(
            &x + y.clone(),
            Matrix::new([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
        );
        assert_eq!(
            x + y,
            Matrix::new([[10., 10., 10.], [10., 10., 10.], [10., 10., 10.]])
        );
    }

    #[test]
    fn sub() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);
        assert_eq!(
            &x - &y,
            Matrix::new([[-8., -6., -4.], [-2., 0., 2.], [4., 6., 8.]])
        );
        assert_eq!(
            x.clone() - &y,
            Matrix::new([[-8., -6., -4.], [-2., 0., 2.], [4., 6., 8.]])
        );
        assert_eq!(
            &x - y.clone(),
            Matrix::new([[-8., -6., -4.], [-2., 0., 2.], [4., 6., 8.]])
        );
        assert_eq!(
            x - y,
            Matrix::new([[-8., -6., -4.], [-2., 0., 2.], [4., 6., 8.]])
        );
    }

    #[test]
    fn cwise_mul() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);
        assert_eq!(
            x.cwise_mul(&y),
            Matrix::new([[9., 16., 21.], [24., 25., 24.], [21., 16., 9.]])
        );
    }

    #[test]
    fn cwise_div() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            x.cwise_div(&y),
            Matrix::new([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
        );
    }

    #[test]
    fn cwise_max() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);
        assert_eq!(
            x.cwise_max(&y),
            Matrix::new([[9., 8., 7.], [6., 5., 6.], [7., 8., 9.]])
        );
    }

    #[test]
    fn cwise_min() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);
        assert_eq!(
            x.cwise_min(&y),
            Matrix::new([[1., 2., 3.], [4., 5., 4.], [3., 2., 1.]])
        );
    }

    #[test]
    fn matmul() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.]]);
        let y = Matrix::new([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]);

        assert_eq!(
            &x * &y,
            Matrix::new([[38., 44., 50., 56.], [83., 98., 113., 128.]])
        );
        assert_eq!(
            x.clone() * &y,
            Matrix::new([[38., 44., 50., 56.], [83., 98., 113., 128.]])
        );
        assert_eq!(
            &x * y.clone(),
            Matrix::new([[38., 44., 50., 56.], [83., 98., 113., 128.]])
        );
        assert_eq!(
            x * y,
            Matrix::new([[38., 44., 50., 56.], [83., 98., 113., 128.]])
        );

        let x = Matrix::new([[1., 2., 3., 4., 5.]]);
        let y = Matrix::new([[5., 4., 3., 2., 1.]]).transpose();
        assert_eq!(&x * &y, Matrix::new([[35.]]));
        assert_eq!(
            &y * &x,
            Matrix::new([
                [5., 4., 3., 2., 1.],
                [10., 8., 6., 4., 2.],
                [15., 12., 9., 6., 3.],
                [20., 16., 12., 8., 4.],
                [25., 20., 15., 10., 5.]
            ])
        );

        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[1., 2., 3.]]).transpose();
        assert_eq!(&x * &y, Matrix::new([[14., 32., 50.]]).transpose());
        assert_eq!(y.transpose() * &x, Matrix::new([[30., 36., 42.]]));

        let x = Matrix::new([
            [1., -1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., -1., 0., 0., 0., 0., 0.],
            [0., 1., -1., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., -1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., -1., 0., 0., 0.],
            [0., 0., 0., 1., -1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., -1., 0., 0.],
            [0., 0., 0., 0., 1., -1., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., -1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., -1.],
            [0., 0., 0., 0., 0., 0., 1., -1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., -1.],
        ]);
        let y = Matrix::<f64>::one(9, 1);
        assert_eq!(&x * &y, Matrix::<f64>::zero(12, 1));

        let x = Matrix::new([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
        let y = Matrix::new([[1.0, 0.5]]);
        assert_eq!(&y * &x, Matrix::new([[0.2, 0.5, 0.8]]));
    }

    #[test]
    fn sum() {
        let x = Matrix::new([[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.]]);
        assert_eq!(
            x.sum(Some(Axis::ROW)),
            Matrix::new_col_vector([10., 26., 42.])
        );
        assert_eq!(
            x.sum(Some(Axis::COLUMN)),
            Matrix::new_row_vector([15., 18., 21., 24.])
        );
        assert_eq!(x.sum(None), Matrix::new([[78.]]));
    }

    #[test]
    fn max() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(x.max(Some(Axis::ROW)), Matrix::new_col_vector([3., 6., 9.]));
        assert_eq!(
            x.max(Some(Axis::COLUMN)),
            Matrix::new_row_vector([7., 8., 9.])
        );
        assert_eq!(x.max(None), Matrix::new([[9.]]));
    }

    #[test]
    fn min() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(x.min(Some(Axis::ROW)), Matrix::new_col_vector([1., 4., 7.]));
        assert_eq!(
            x.min(Some(Axis::COLUMN)),
            Matrix::new_row_vector([1., 2., 3.])
        );
        assert_eq!(x.min(None), Matrix::new([[1.]]));
    }

    #[test]
    fn relu() {
        let x = Matrix::new([[-1., 2., -3.], [4., -5., 6.], [-7., 8., 9.]]);
        assert_eq!(
            x.relu(),
            Matrix::new([[0., 2., 0.], [4., 0., 6.], [0., 8., 9.]])
        );
    }

    #[test]
    fn triangle_func() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            x.sin(),
            Matrix::new([
                [(1_f64).sin(), (2_f64).sin(), (3_f64).sin()],
                [(4_f64).sin(), (5_f64).sin(), (6_f64).sin()],
                [(7_f64).sin(), (8_f64).sin(), (9_f64).sin()]
            ])
        );
        assert_eq!(
            x.cos(),
            Matrix::new([
                [(1_f64).cos(), (2_f64).cos(), (3_f64).cos()],
                [(4_f64).cos(), (5_f64).cos(), (6_f64).cos()],
                [(7_f64).cos(), (8_f64).cos(), (9_f64).cos()]
            ])
        );
    }

    #[test]
    fn index() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(x[0][0], 1.);
        assert_eq!(x[2][2], 9.);

        let mut y = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(y[0][0], 1.);
        y[0][0] = -2.;
        assert_eq!(y[0][0], -2.);
    }

    #[test]
    fn concat() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        let y = Matrix::new([[9., 8., 7.], [6., 5., 4.], [3., 2., 1.]]);
        assert_eq!(
            x.concat(&y, Axis::ROW),
            Matrix::new([
                [1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.],
                [9., 8., 7.],
                [6., 5., 4.],
                [3., 2., 1.]
            ])
        );
        assert_eq!(
            x.concat(&y, Axis::COLUMN),
            Matrix::new([
                [1., 2., 3., 9., 8., 7.],
                [4., 5., 6., 6., 5., 4.],
                [7., 8., 9., 3., 2., 1.],
            ])
        );
    }

    #[test]
    fn step() {
        let x = Matrix::new([[-2., -1., 0., 1., 2.]]);
        assert_eq!(x.step(), Matrix::new([[0., 0., 0., 1., 1.]]));
    }

    #[test]
    fn clamp() {
        let x = Matrix::new([[-2., -1., -0.5, 0., 0.5, 1., 2.]]);
        assert_eq!(
            x.clamp(-1., 1.),
            Matrix::new([[-1., -1., -0.5, 0., 0.5, 1., 1.]])
        );
    }

    #[test]
    fn repeat() {
        let x = Matrix::new([[-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2.]]);
        assert_eq!(
            x.repeat(-1., 1.),
            Matrix::new([[0., 0.5, -1., -0.5, 0., 0.5, 1., -0.5, 0.]])
        );
    }

    #[test]
    fn normalize1() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            x.normalize1(None),
            Matrix::new([
                [1. / 45., 2. / 45., 3. / 45.],
                [4. / 45., 5. / 45., 6. / 45.],
                [7. / 45., 8. / 45., 9. / 45.]
            ])
        );
        assert_eq!(
            x.normalize1(Some(Axis::ROW)),
            Matrix::new([
                [1. / 6., 2. / 6., 3. / 6.],
                [4. / 15., 5. / 15., 6. / 15.],
                [7. / 24., 8. / 24., 9. / 24.]
            ])
        );
        assert_eq!(
            x.normalize1(Some(Axis::COLUMN)),
            Matrix::new([
                [1. / 12., 2. / 15., 3. / 18.],
                [4. / 12., 5. / 15., 6. / 18.],
                [7. / 12., 8. / 15., 9. / 18.]
            ])
        );
    }

    #[test]
    fn normalize2() {
        let x = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            x.normalize2(None),
            Matrix::new([
                [
                    1. / (285_f64).sqrt(),
                    2. / (285_f64).sqrt(),
                    3. / (285_f64).sqrt()
                ],
                [
                    4. / (285_f64).sqrt(),
                    5. / (285_f64).sqrt(),
                    6. / (285_f64).sqrt()
                ],
                [
                    7. / (285_f64).sqrt(),
                    8. / (285_f64).sqrt(),
                    9. / (285_f64).sqrt()
                ]
            ])
        );
        assert_eq!(
            x.normalize2(Some(Axis::ROW)),
            Matrix::new([
                [
                    1. / (14_f64).sqrt(),
                    2. / (14_f64).sqrt(),
                    3. / (14_f64).sqrt()
                ],
                [
                    4. / (77_f64).sqrt(),
                    5. / (77_f64).sqrt(),
                    6. / (77_f64).sqrt()
                ],
                [
                    7. / (194_f64).sqrt(),
                    8. / (194_f64).sqrt(),
                    9. / (194_f64).sqrt()
                ]
            ])
        );
        assert_eq!(
            x.normalize2(Some(Axis::COLUMN)),
            Matrix::new([
                [
                    1. / (66_f64).sqrt(),
                    2. / (93_f64).sqrt(),
                    3. / (126_f64).sqrt()
                ],
                [
                    4. / (66_f64).sqrt(),
                    5. / (93_f64).sqrt(),
                    6. / (126_f64).sqrt()
                ],
                [
                    7. / (66_f64).sqrt(),
                    8. / (93_f64).sqrt(),
                    9. / (126_f64).sqrt()
                ]
            ])
        );
    }

    #[test]
    fn floor() {
        let x = Matrix::new_col_vector([1., 2., 3.]);
        assert_eq!(
            x.floor(),
            Matrix::new_col_vector([1_f64.floor(), 2_f64.floor(), 3_f64.floor()])
        );
    }

    #[test]
    fn diag() {
        let x = Matrix::new_col_vector([1., 2., 3.]);
        assert_eq!(
            x.diag(),
            Matrix::new([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        );
        let y = Matrix::new([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);
        assert_eq!(
            y.diag_row(),
            Matrix::new([
                [1., 2., 3., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 4., 5., 6., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 7., 8., 9.]
            ])
        );
        assert_eq!(
            y.diag_col(),
            Matrix::new([
                [1., 0., 0.],
                [4., 0., 0.],
                [7., 0., 0.],
                [0., 2., 0.],
                [0., 5., 0.],
                [0., 8., 0.],
                [0., 0., 3.],
                [0., 0., 6.],
                [0., 0., 9.]
            ])
        )
    }

    #[test]
    fn sparse() {
        let x = SparseMatrix::new([
            [1., 0., 0., 0.],
            [0., 2., 1., 0.],
            [3., 0., 0., 2.],
            [0., 0., 1., 0.],
        ]);
        assert_eq!(
            x,
            SparseMatrix::<f64> {
                rows: 4,
                cols: 4,
                val: vec![1., 2., 1., 3., 2., 1.],
                col_idx: vec![0, 1, 2, 0, 3, 2],
                row_ptr: vec![0, 1, 3, 5, 6],
            }
        );
        assert_eq!(
            x.to_dence(),
            Matrix::<f64>::new([
                [1., 0., 0., 0.],
                [0., 2., 1., 0.],
                [3., 0., 0., 2.],
                [0., 0., 1., 0.],
            ])
        )
    }
}
