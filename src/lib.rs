pub mod matrix;
pub mod ops;
pub mod rand;

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::matrix::{Matrix, Axis};

    #[test]
    fn reshape() {
        let mut x = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        x.reshape(1, 9);
        assert_eq!(x, Matrix::new_row_vector([1.,2.,3.,4.,5.,6.,7.,8.,9.]));
    }

    #[test]
    fn transpose() {
        let x = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.transpose(), Matrix::new([[1.,4.,7.],[2.,5.,8.],[3.,6.,9.]]));
    }

    #[test]
    fn neg() {
        let x = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(-x, Matrix::new([[-1.,-2.,-3.],[-4.,-5.,-6.],[-7.,-8.,-9.]]));
    }

    #[test]
    fn add() {
        let x = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        let y = Matrix::new([[9.,8.,7.],[6.,5.,4.],[3.,2.,1.]]);
        assert_eq!(&x        + &y,        Matrix::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
        assert_eq!(x.clone() + &y,        Matrix::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
        assert_eq!(&x        + y.clone(), Matrix::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
        assert_eq!(x         + y,         Matrix::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
    }

    #[test]
    fn sub() {
        let x = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        let y = Matrix::new([[9.,8.,7.],[6.,5.,4.],[3.,2.,1.]]);
        assert_eq!(&x        - &y,        Matrix::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
        assert_eq!(x.clone() - &y,        Matrix::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
        assert_eq!(&x        - y.clone(), Matrix::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
        assert_eq!(x         - y,         Matrix::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
    }

    #[test]
    fn cwise_mul() {
        let x = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        let y = Matrix::new([[9.,8.,7.],[6.,5.,4.],[3.,2.,1.]]);
        assert_eq!(x.cwise_mul(&y), Matrix::new([[9.,16.,21.],[24.,25.,24.],[21.,16.,9.]]));
    }

    #[test]
    fn cwise_div() {
        let x = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        let y = Matrix::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.cwise_div(&y), Matrix::new([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]]));
    }

    #[test]
    fn sum() {
        let x = Matrix::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.sum(Axis::ROW),    Matrix::new_col_vector([6.,15.,24.]));
        assert_eq!(x.sum(Axis::COLUMN), Matrix::new_row_vector([12.,15.,18.]));
        assert_eq!(x.sum(Axis::BOTH),   Matrix::new([[45.]]));
    }

    #[test]
    fn max() {
        let x = Matrix::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.max(Axis::ROW),    Matrix::new_col_vector([3.,6.,9.]));
        assert_eq!(x.max(Axis::COLUMN), Matrix::new_row_vector([7.,8.,9.]));
        assert_eq!(x.max(Axis::BOTH),   Matrix::new([[9.]]));
    }

    #[test]
    fn min() {
        let x = Matrix::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.min(Axis::ROW),    Matrix::new_col_vector([1.,4.,7.]));
        assert_eq!(x.min(Axis::COLUMN), Matrix::new_row_vector([1.,2.,3.]));
        assert_eq!(x.min(Axis::BOTH),   Matrix::new([[1.]]));
    }

    #[test]
    fn index() {
        let x = Matrix::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x[0][0], 1.);
        assert_eq!(x[2][2], 9.);

        let mut y = Matrix::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(y[0][0], 1.);
        y[0][0] = -2.;
        assert_eq!(y[0][0], -2.);
    }
}
