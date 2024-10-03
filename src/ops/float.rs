use std::iter::Sum;

use crate::{Axis, Matrix};
use num_traits::Float;

#[cfg(feature = "mpi")]
use mpi::traits::*;
#[cfg(feature = "rayon")]
use rayon::{iter::FromParallelIterator, prelude::*};

#[cfg(feature = "rayon")]
macro_rules! deffloatfunc_rayon {
    ( $( $f: ident ),+ ) => {
        impl<T> Matrix<T>
        where
            T: Float + Send + Sync,
            Vec<T>: FromParallelIterator<T>,
        {
            $(
                pub fn $f(&self) -> Self {
                    Self {
                        rows: self.rows,
                        cols: self.cols,
                        array: self.array.par_iter().map(|&x| x.$f()).collect(),
                    }
                }
            )+
        }
    };
}

#[cfg(not(feature = "rayon"))]
macro_rules! deffloatfunc {
    ( $( $f: ident ),+ ) => {
        impl<T> Matrix<T>
        where
            T: Float,
        {
            $(
                pub fn $f(&self) -> Self {
                    Self {
                        rows: self.rows,
                        cols: self.cols,
                        array: self.array.iter().map(|&x| x.$f()).collect(),
                    }
                }
            )+
        }
    };
}

#[cfg(feature = "rayon")]
deffloatfunc_rayon![
    floor, ceil, round, trunc, fract, abs, signum, recip, sqrt, exp, exp2, ln, log2, log10, cbrt,
    sin, cos, tan, sinh, cosh, tanh
];

#[cfg(feature = "rayon")]
impl<T> Matrix<T>
where
    T: Float + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    pub fn powi(&self, n: i32) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.powi(n)).collect(),
        }
    }

    pub fn powf(&self, n: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.powf(n)).collect(),
        }
    }

    pub fn log(&self, base: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.par_iter().map(|&x| x.log(base)).collect(),
        }
    }

    pub fn sigmoid(&self) -> Self {
        (Self::one_like(self) + (-self).exp()).recip()
    }

    pub fn step(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .map(|&x| if x > T::zero() { T::one() } else { T::zero() })
                .collect(),
        }
    }

    pub fn clamp(&self, min: T, max: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .map(|&x| {
                    if x < min {
                        min
                    } else if x > max {
                        max
                    } else {
                        x
                    }
                })
                .collect(),
        }
    }

    pub fn repeat(&self, min: T, max: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .par_iter()
                .map(|&x| {
                    if x < min {
                        x + max - min
                    } else if x > max {
                        x - (max - min)
                    } else {
                        x
                    }
                })
                .collect(),
        }
    }

    pub fn normalize1(&self, axis: Option<Axis>) -> Self {
        match axis {
            Some(Axis::ROW) => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .chunks(self.cols)
                    .map(|arr| {
                        arr.iter()
                            .map(|&x| x / arr.iter().fold(T::zero(), |first, x| first + *x))
                    })
                    .flatten()
                    .collect(),
            },
            Some(Axis::COLUMN) => self.transpose().normalize1(Some(Axis::ROW)).transpose(),
            None => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .iter()
                    .map(|&x| x / self.array.iter().fold(T::zero(), |first, x| first + *x))
                    .collect(),
            },
        }
    }

    pub fn normalize2(&self, axis: Option<Axis>) -> Self {
        match axis {
            Some(Axis::ROW) => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .chunks(self.cols)
                    .map(|arr| {
                        arr.iter().map(|&x| {
                            x / arr
                                .iter()
                                .fold(T::zero(), |first, x| first + *x * *x)
                                .sqrt()
                        })
                    })
                    .flatten()
                    .collect(),
            },
            Some(Axis::COLUMN) => self.transpose().normalize2(Some(Axis::ROW)).transpose(),
            None => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .iter()
                    .map(|&x| {
                        x / self
                            .array
                            .iter()
                            .fold(T::zero(), |first, x| first + *x * *x)
                            .sqrt()
                    })
                    .collect(),
            },
        }
    }
}

#[cfg(not(feature = "rayon"))]
deffloatfunc![
    floor, ceil, round, trunc, fract, abs, signum, recip, sqrt, exp, exp2, ln, log2, log10, cbrt,
    sin, cos, tan, sinh, cosh, tanh
];

#[cfg(not(feature = "rayon"))]
impl<T> Matrix<T>
where
    T: Float,
{
    pub fn powi(&self, n: i32) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.powi(n)).collect(),
        }
    }

    pub fn powf(&self, n: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.powf(n)).collect(),
        }
    }

    pub fn log(&self, base: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.log(base)).collect(),
        }
    }

    pub fn sigmoid(&self) -> Self {
        (Self::one_like(self) + (-self).exp()).recip()
    }

    pub fn step(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .map(|&x| if x > T::zero() { T::one() } else { T::zero() })
                .collect(),
        }
    }

    pub fn clamp(&self, min: T, max: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .map(|&x| {
                    if x < min {
                        min
                    } else if x > max {
                        max
                    } else {
                        x
                    }
                })
                .collect(),
        }
    }

    pub fn repeat(&self, min: T, max: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self
                .array
                .iter()
                .map(|&x| {
                    if x < min {
                        x + max - min
                    } else if x > max {
                        x - (max - min)
                    } else {
                        x
                    }
                })
                .collect(),
        }
    }

    pub fn normalize1(&self, axis: Option<Axis>) -> Self {
        match axis {
            Some(Axis::ROW) => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .chunks(self.cols)
                    .map(|arr| {
                        arr.iter()
                            .map(|&x| x / arr.iter().fold(T::zero(), |first, x| first + *x))
                    })
                    .flatten()
                    .collect(),
            },
            Some(Axis::COLUMN) => self.transpose().normalize1(Some(Axis::ROW)).transpose(),
            None => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .iter()
                    .map(|&x| x / self.array.iter().fold(T::zero(), |first, x| first + *x))
                    .collect(),
            },
        }
    }

    pub fn normalize2(&self, axis: Option<Axis>) -> Self {
        match axis {
            Some(Axis::ROW) => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .chunks(self.cols)
                    .map(|arr| {
                        arr.iter().map(|&x| {
                            x / arr
                                .iter()
                                .fold(T::zero(), |first, x| first + *x * *x)
                                .sqrt()
                        })
                    })
                    .flatten()
                    .collect(),
            },
            Some(Axis::COLUMN) => self.transpose().normalize2(Some(Axis::ROW)).transpose(),
            None => Self {
                rows: self.rows,
                cols: self.cols,
                array: self
                    .array
                    .iter()
                    .map(|&x| {
                        x / self
                            .array
                            .iter()
                            .fold(T::zero(), |first, x| first + *x * *x)
                            .sqrt()
                    })
                    .collect(),
            },
        }
    }
}

#[cfg(feature = "rayon")]
impl<T> Matrix<T>
where
    T: Sum + Float + Send + Sync,
    Vec<T>: FromParallelIterator<T>,
{
    pub fn softmax(&self) -> Self {
        let max_coef = self.max(None).array[0];
        let tmp = self - Matrix::one_like(self) * max_coef;
        tmp.exp() / tmp.exp().sum(None).array[0]
    }
}

#[cfg(not(feature = "rayon"))]
impl<T> Matrix<T>
where
    T: Sum + Float,
{
    pub fn softmax(&self) -> Self {
        let max_coef = self.max(None).array[0];
        let tmp = self - Matrix::one_like(self) * max_coef;
        tmp.exp() / tmp.exp().sum(None).array[0]
    }
}
