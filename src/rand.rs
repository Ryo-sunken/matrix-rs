use crate::matrix::Matrix;

use rand::Rng;
use rand::distributions::{Standard, Distribution};
use rand::rngs::ThreadRng;

use rand_distr::{StandardNormal, Normal, Uniform};
use rand_distr::uniform::SampleUniform;

use num_traits::Float;

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

impl<T> Matrix<T>
where
    T: Float + SampleUniform,
{
    pub fn randu(rows: usize, cols: usize, engine: &mut ThreadRng) -> Self
    {
        let dist = Uniform::<T>::new(T::zero(), T::one());
        Self
        {
            rows, cols,
            array: (0..(rows * cols)).map(|_| dist.sample(engine)).collect(),
        }
    }
}