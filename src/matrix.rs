use rand::Rng;
use rand::distributions::{Standard, Distribution};
use rand::rngs::ThreadRng;

use rand_distr::{StandardNormal, Normal};

use rayon::prelude::*;

use num_traits::identities::{One, Zero};
use num_traits::Float;

use std::marker::{Send, Sync};
use std::ops::{Neg, Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use std::cmp::PartialEq;

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


#[derive(Clone, Debug)]
pub struct Matrix<T>
{
    rows: usize,
    cols: usize,
    array: Vec<T>,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum Axis {
    ROW,
    COLUMN,
    BOTH,
}

impl<T> Matrix<T>
{
    pub fn new<const R: usize, const C: usize>(data: [[T; C]; R]) -> Self
    {
        Self 
        {
            rows: R,
            cols: C,
            array: data.into_iter().flatten().collect(),
        }
    }

    pub fn new_row_vector<const C: usize>(data: [T; C]) -> Self
    {
        Self
        {
            rows: 1,
            cols: C,
            array: data.into_iter().collect(),
        }
    }

    pub fn new_col_vector<const R: usize>(data: [T; R]) -> Self
    {
        Self
        {
            rows: R,
            cols: 1,
            array: data.into_iter().collect(),
        }
    }

    pub fn from_vec(array: Vec<T>, rows: usize, cols: usize) -> Self
    {
        Self { rows, cols, array }
    }

    pub fn from_vec_row(array: Vec<T>) -> Self
    {
        let cols = array.len();
        Self::from_vec(array, 1, cols)
    }

    pub fn from_vec_col(array: Vec<T>) -> Self
    {
        let rows = array.len();
        Self::from_vec(array, rows, 1)
    }

    pub fn rows(&self) -> usize
    {
        self.rows
    }

    pub fn cols(&self) -> usize
    {
        self.cols
    }

    pub fn get_ref(&self, r: usize, c: usize) -> Option<&T>
    {
        if self.rows <= r || self.cols <= c { return None; }
        self.array.get(r * self.cols + c)
    }

    pub fn reshape(&mut self, rows: usize, cols: usize)
    {
        assert_eq!(self.rows * self.cols, rows * cols);
        self.rows = rows;
        self.cols = cols;
    }
}

#[allow(dead_code)]
impl<T> Matrix<T>
where T: Clone,
{
    pub fn as_shape(&self, rows: usize, cols: usize) -> Self
    {
        assert_eq!(self.rows * self.cols, rows * cols);

        Self
        {
            rows, cols,
            array: self.array.clone()
        }
    }

    pub fn from_slice<const C: usize>(data: &[[T;C]]) -> Self
    {
        Self
        {
            rows: data.len(),
            cols: C,
            array: data.iter().flatten().cloned().collect(),
        }
    }

    pub fn get(&self, r: usize, c: usize) -> Option<T>
    {
        if self.rows <= r || self.cols <= c { return None; }
        self.array.get(r * self.cols + c).cloned()
    }

    pub fn transpose(&self) -> Self
    {
        let mut array = Vec::with_capacity(self.rows * self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                array.push(self.array[r * self.cols + c].clone());
            }
        }
        Self 
        {
            rows: self.cols,
            cols: self.rows,
            array,
        }
    }
}

impl<T> Matrix<T>
where T: Zero + Clone
{
    pub fn zero(rows: usize, cols: usize) -> Self
    {
        Self
        {
            rows, cols,
            array: vec![T::zero(); rows * cols],
        }
    }

    pub fn zero_like<U>(mat: &Matrix<U>) -> Self
    {
        Self::zero(mat.rows, mat.cols)
    }

    pub fn diag<const D: usize>(data: [T; D]) -> Self
    {
        let mut array = vec![T::zero(); D * D];
        for d in 0..D {
            array[d * D + d] = data[d].clone();
        }

        Self
        {
            rows: D, cols: D,
            array
        }
    }
}

impl<T> Matrix<T>
where T: One + Clone
{
    pub fn one(rows: usize, cols: usize) -> Self
    {
        Self
        {
            rows, cols,
            array: vec![T::one(); rows * cols],
        }
    }

    pub fn one_like<U>(mat: &Matrix<U>) -> Self
    {
        Self::one(mat.rows, mat.cols)
    }
}

impl<T> Matrix<T>
where T: One + Zero + Clone,
{
    pub fn eye(dim: usize) -> Self
    {
        let mut array = vec![T::zero(); dim * dim];
        for i in 0..dim {
            array[i * dim + i] = T::one();
        }

        Self
        {
            rows: dim,
            cols: dim,
            array,
        }
    }
}

impl<T> Matrix<T>
where
    T: Mul + Copy + Send + Sync,
    <T as Mul>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Mul>::Output>,
{
    pub fn cwise_mul(&self, rhs: &Matrix<T>) -> Self
    {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().zip(rhs.array.par_iter())
                .map(|(&x, &y)| x * y)
                .collect()
        }
    }
}

impl<T> Matrix<T>
where
    T: Div + Copy + Send + Sync,
    <T as Div>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Div>::Output>,
{
    pub fn cwise_div(&self, rhs: &Matrix<T>) -> Self
    {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().zip(rhs.array.par_iter())
                .map(|(&x, &y)| x / y)
                .collect()
        }
    }
}

impl<T> Matrix<T>
where
    T: Float + Send + Sync
{
    pub fn exp(&self) -> Self
    {
        Self
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter()
                .map(|&x| x.exp())
                .collect(),
        }
    }

    pub fn sin(&self) -> Self
    {
        Self
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter()
                .map(|&x| x.sin())
                .collect(),
        }
    }

    pub fn cos(&self) -> Self
    {
        Self
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter()
                .map(|&x| x.cos())
                .collect(),
        }
    }

    pub fn powf(&self, n: T) -> Self
    {
        Self
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter()
                .map(|&x| x.powf(n))
                .collect(),
        }
    }
}

impl<T> Matrix<T>
where
    Standard: Distribution<T>,
{
    pub fn rand(rows: usize, cols: usize, engine: &mut ThreadRng) -> Self
    {
        Self
        {
            rows, cols,
            array: (0..(rows * cols)).map(|_| engine.gen::<T>()).collect(),
        }
    }
}


impl<T> Matrix<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    pub fn randn(rows: usize, cols: usize, engine: &mut ThreadRng) -> Self
    {
        let dist = Normal::<T>::new(T::zero(), T::one()).unwrap();
        Self
        {
            rows, cols,
            array: (0..(rows * cols)).map(|_| dist.sample(engine)).collect(),
        }
    }
}

impl Matrix<f64>
{
    pub fn max(&self, ax: Axis) -> Self
    {
        let max = |x, y| if y > x { y } else { x };
        match ax {
            Axis::ROW => {
                let split_array = self.array.chunks(self.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let max_array = split_array.into_par_iter()
                    .map(|s| s.into_par_iter().reduce(|| f64::NEG_INFINITY, max))
                    .collect::<Vec<_>>();
                Self::from_vec_col(max_array)
            },
            Axis::COLUMN => {
                let split_array = self.transpose().array.chunks(self.rows)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let max_array = split_array.into_par_iter()
                    .map(|s| s.into_par_iter().reduce(|| f64::NEG_INFINITY, max))
                    .collect::<Vec<_>>();
                Self::from_vec_row(max_array)
            },
            Axis::BOTH => Self::new([[self.array.clone().into_par_iter().reduce(|| f64::NEG_INFINITY, max)]]),
        }
    }

    pub fn min(&self, ax: Axis) -> Self
    {
        let min = |x, &y| if y < x { y } else { x };
        match ax {
            Axis::ROW => {
                let split_array = self.array
                    .chunks(self.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let min_array = split_array.par_iter()
                    .map(|s| s.iter().fold(f64::INFINITY, min))
                    .collect::<Vec<_>>();
                Self::from_vec_col(min_array)
            },
            Axis::COLUMN => {
                let trans = self.transpose();
                let split_array = trans.array
                    .chunks(trans.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let min_array = split_array.par_iter()
                    .map(|s| s.iter().fold(f64::INFINITY, min))
                    .collect::<Vec<_>>();
                Self::from_vec_row(min_array)
            },
            Axis::BOTH => Self::new([[self.array.iter().fold(f64::INFINITY, min)]]),
        }
    }

    pub fn sum(&self, ax: Axis) -> Self
    {
        match ax {
            Axis::ROW => {
                let split_array = self.array.chunks(self.cols)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let sum_array = split_array.into_par_iter()
                    .map(|s| s.into_par_iter().reduce(|| 0_f64, |sum, x| sum + x))
                    .collect::<Vec<_>>();
                Self::from_vec_col(sum_array)
            },
            Axis::COLUMN => {
                let split_array = self.transpose().array.chunks(self.rows)
                    .map(|s| s.into())
                    .collect::<Vec<Vec<_>>>();
                let sum_array = split_array.into_par_iter()
                    .map(|s| s.into_par_iter().reduce(|| 0_f64, |sum, x| sum + x))
                    .collect::<Vec<_>>();
                Self::from_vec_row(sum_array)
            },
            Axis::BOTH => Self::new([[self.array.clone().into_par_iter().reduce(|| 0_f64, |sum, x| sum + x)]])
        }
    }

}

// ParticalEq implementation
impl<T> PartialEq<Matrix<T>> for Matrix<T>
where
    T: PartialEq + Send + Sync,
{
    fn eq(&self, other: &Self) -> bool {
        self.array.par_iter().zip(other.array.par_iter())
            .all(|(x, y)| x == y)
    }
}

// Neg implementation
impl<T> Neg for &Matrix<T>
where
    T: Neg + Copy + Send + Sync,
    <T as Neg>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Neg>::Output>,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        Self::Output
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter()
                .map(|&x| -x)
                .collect(),
        }
    }
}
impl<T> Neg for Matrix<T>
where
    T: Neg + Copy + Send + Sync,
    <T as Neg>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Neg>::Output>,
{
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

// Add implementation
impl<T> Add<&Matrix<T>> for &Matrix<T>
where
    T: Add + Copy + Send + Sync,
    <T as Add>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Add>::Output>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        Self::Output
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().zip(rhs.array.par_iter())
                .map(|(&x, &y)| x + y)
                .collect(),
        }
    }
}
impl<T> Add<&Matrix<T>> for Matrix<T>
where
    T: Add + Copy + Send + Sync,
    <T as Add>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Add>::Output>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: &Matrix<T>) -> Self::Output {
        &self + rhs
    }
}
impl<T> Add<Matrix<T>> for &Matrix<T>
where
    T: Add + Copy + Send + Sync,
    <T as Add>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Add>::Output>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        self + &rhs
    }
}
impl<T> Add<Matrix<T>> for Matrix<T>
where
    T: Add + Copy + Send + Sync,
    <T as Add>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Add>::Output>,
{
    type Output = Matrix<T>;

    fn add(self, rhs: Matrix<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<T> AddAssign<&Matrix<T>> for Matrix<T>
where
    T: Add + Copy + Send + Sync,
    <T as Add>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Add>::Output>,
{
    fn add_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);

        self.array = self.array.par_iter().zip(rhs.array.par_iter())
            .map(|(&x, &y)| x + y)
            .collect();
    }
}
impl<T> AddAssign<Matrix<T>> for Matrix<T>
where
    T: Add + Copy + Send + Sync,
    <T as Add>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Add>::Output>,
{
    fn add_assign(&mut self, rhs: Matrix<T>) {
        *self += &rhs;
    }
}

// Sub implementation
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

        Self::Output
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().zip(rhs.array.par_iter())
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

        self.array = self.array.par_iter().zip(rhs.array.par_iter())
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

// MatMul implementation
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

// ScalarMul implementation
defscalarmul![i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64];

impl<T> Mul<T> for &Matrix<T>
where
    T: Mul + Copy + Send + Sync,
    <T as Mul>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Mul>::Output>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Self::Output
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x * rhs).collect(),
        }
    }
}
impl<T> Mul<T> for Matrix<T>
where
    T: Mul + Copy + Send + Sync,
    <T as Mul>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Mul>::Output>,
{
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}

impl<T> MulAssign<T> for Matrix<T>
where
    T: Mul + Copy + Send + Sync,
    <T as Mul>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Mul>::Output>,
{
    fn mul_assign(&mut self, rhs: T) {
        self.array = self.array.par_iter().map(|&x| x * rhs).collect();
    }
}

// ScalarDiv implementation
impl<T> Div<T> for &Matrix<T>
where
    T: Div + Copy + Send + Sync,
    <T as Div>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Div>::Output>,
{
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        Self::Output
        {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x / rhs).collect(),
        }
    }
}
impl<T> Div<T> for Matrix<T>
where
    T: Div + Copy + Send + Sync,
    <T as Div>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Div>::Output>,
{
    type Output = Matrix<T>;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

impl<T> DivAssign<T> for Matrix<T>
where
    T: Div + Copy + Send + Sync,
    <T as Div>::Output: Send + Sync,
    Vec<T>: FromParallelIterator<<T as Div>::Output>,
{
    fn div_assign(&mut self, rhs: T) {
        self.array = self.array.par_iter().map(|&x| x / rhs).collect()
    }
}

impl<T> std::fmt::Display for Matrix<T>
where T: std::fmt::Display + Copy,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[\n")?;
        for row in self.array.chunks(self.cols) {
            write!(f, " [ {} ]", row.iter().map(|&x| format!("{:.4}", x)).collect::<Vec<_>>().join(" "))?;
            write!(f, "\n")?;
        }
        write!(f, "]")
    }
}