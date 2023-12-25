pub mod matrix;
pub mod add;
pub mod sub;
pub mod matmul;
pub mod scalarmul;

#[cfg(test)]
mod tests 
{
    #[allow(unused_imports)]
    use crate::matrix::{Matrix, Axis};

    #[test]
    fn transpose()
    {
        let x = Matrix::<f64>::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.transpose(), Matrix::<f64>::new([[1.,4.,7.],[2.,5.,8.],[3.,6.,9.]]));
    }

    #[test]
    fn neg()
    {
        let x = Matrix::<f64>::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(-x, Matrix::<f64>::new([[-1.,-2.,-3.],[-4.,-5.,-6.],[-7.,-8.,-9.]]));
    }

    #[test]
    fn add() 
    {
        let x = Matrix::<f64>::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        let y = Matrix::<f64>::new([[9.,8.,7.],[6.,5.,4.],[3.,2.,1.]]);
        assert_eq!(&x        + &y,        Matrix::<f64>::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
        assert_eq!(x.clone() + &y,        Matrix::<f64>::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
        assert_eq!(&x        + y.clone(), Matrix::<f64>::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
        assert_eq!(x         + y,         Matrix::<f64>::new([[10.,10.,10.],[10.,10.,10.],[10.,10.,10.]]));
    }

    #[test]
    fn sub()
    {
        let x = Matrix::<f64>::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        let y = Matrix::<f64>::new([[9.,8.,7.],[6.,5.,4.],[3.,2.,1.]]);
        assert_eq!(&x        - &y,        Matrix::<f64>::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
        assert_eq!(x.clone() - &y,        Matrix::<f64>::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
        assert_eq!(&x        - y.clone(), Matrix::<f64>::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
        assert_eq!(x         - y,         Matrix::<f64>::new([[-8.,-6.,-4.],[-2.,0.,2.],[4.,6.,8.]]));
    }

    #[test]
    fn cwise_mul()
    {
        let x = Matrix::<f64>::new([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]);
        let y = Matrix::<f64>::new([[9.,8.,7.],[6.,5.,4.],[3.,2.,1.]]);
        assert_eq!(x.cwise_mul(&y), Matrix::<f64>::new([[9.,16.,21.],[24.,25.,24.],[21.,16.,9.]]));
    }

    #[test]
    fn sum()
    {
        let x = Matrix::<f64>::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.sum(Axis::ROW),    Matrix::<f64>::new_col_vector([6.,15.,24.]));
        assert_eq!(x.sum(Axis::COLUMN), Matrix::<f64>::new_row_vector([12.,15.,18.]));
        assert_eq!(x.sum(Axis::BOTH),   Matrix::<f64>::new([[45.]]));
    }

    #[test]
    fn max()
    {
        let x = Matrix::<f64>::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.max(Axis::ROW),    Matrix::<f64>::new_col_vector([3.,6.,9.]));
        assert_eq!(x.max(Axis::COLUMN), Matrix::<f64>::new_row_vector([7.,8.,9.]));
        assert_eq!(x.max(Axis::BOTH),   Matrix::<f64>::new([[9.]]));
    }

    #[test]
    fn min()
    {
        let x = Matrix::<f64>::new([[1.,2.,3.,],[4.,5.,6.],[7.,8.,9.]]);
        assert_eq!(x.min(Axis::ROW),    Matrix::<f64>::new_col_vector([1.,4.,7.]));
        assert_eq!(x.min(Axis::COLUMN), Matrix::<f64>::new_row_vector([1.,2.,3.]));
        assert_eq!(x.min(Axis::BOTH),   Matrix::<f64>::new([[1.]]));
    }
}
