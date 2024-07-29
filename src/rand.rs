use crate::Matrix;

use rand::distributions::{Distribution, Standard};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

use rand_distr::uniform::SampleUniform;
use rand_distr::{Normal, StandardNormal, Uniform};

use num_traits::Float;

impl<T> Matrix<T>
where
    Standard: Distribution<T>,
{
    pub fn rand(rows: usize, cols: usize, engine: &mut ChaCha8Rng) -> Self {
        Self {
            rows,
            cols,
            array: (0..(rows * cols)).map(|_| engine.gen::<T>()).collect(),
        }
    }
}

impl<T> Matrix<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    pub fn randn(rows: usize, cols: usize, engine: &mut ChaCha8Rng) -> Self {
        let dist = Normal::<T>::new(T::zero(), T::one()).unwrap();
        Self {
            rows,
            cols,
            array: (0..(rows * cols)).map(|_| dist.sample(engine)).collect(),
        }
    }
}

impl<T> Matrix<T>
where
    T: Float + SampleUniform,
{
    pub fn randu(rows: usize, cols: usize, engine: &mut ChaCha8Rng) -> Self {
        let dist = Uniform::<T>::new(T::zero(), T::one());
        Self {
            rows,
            cols,
            array: (0..(rows * cols)).map(|_| dist.sample(engine)).collect(),
        }
    }
}
