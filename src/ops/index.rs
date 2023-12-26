use crate::matrix::Matrix;
use std::ops::{Index, IndexMut};

impl<T> Index<usize> for Matrix<T>
{
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.array[(index*self.cols)..((index + 1)*self.cols)]
    }
}

impl<T> IndexMut<usize> for Matrix<T>
{

    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.array[(index*self.cols)..((index + 1)*self.cols)]
    }
}