use std::iter::Sum;

use crate::matrix::Matrix;
use num_traits::Float;

impl<T: Float> Matrix<T> {
    pub fn floor(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.floor()).collect(),
        }
    }

    pub fn ceil(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.ceil()).collect(),
        }
    }

    pub fn round(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.round()).collect(),
        }
    }

    pub fn trunc(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.trunc()).collect(),
        }
    }

    pub fn fract(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.fract()).collect(),
        }
    }

    pub fn abs(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.abs()).collect(),
        }
    }

    pub fn signum(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.signum()).collect(),
        }
    }

    pub fn recip(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.recip()).collect(),
        }
    }

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

    pub fn sqrt(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.sqrt()).collect(),
        }
    }

    pub fn exp(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.exp()).collect(),
        }
    }

    pub fn exp2(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.exp2()).collect(),
        }
    }

    pub fn ln(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.ln()).collect(),
        }
    }

    pub fn log(&self, base: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.log(base)).collect(),
        }
    }

    pub fn log2(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.log2()).collect(),
        }
    }

    pub fn log10(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.log10()).collect(),
        }
    }

    pub fn cbrt(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.cbrt()).collect(),
        }
    }

    pub fn sin(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.sin()).collect(),
        }
    }

    pub fn cos(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.cos()).collect(),
        }
    }

    pub fn tan(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.tan()).collect(),
        }
    }

    pub fn sinh(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.sinh()).collect(),
        }
    }

    pub fn cosh(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.cosh()).collect(),
        }
    }

    pub fn tanh(&self) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            array: self.array.iter().map(|&x| x.tanh()).collect(),
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
}

impl<T> Matrix<T>
where T: Sum + Float
{
    pub fn softmax(&self) -> Self {
        let max_coef = self.max(None).array[0];
        let tmp = self - Matrix::one_like(self) * max_coef;
        tmp.exp() / tmp.exp().sum(None).array[0]
    }
}
