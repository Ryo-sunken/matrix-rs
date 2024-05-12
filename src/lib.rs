pub mod matrix;
pub mod ops;
pub mod rand;

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use crate::matrix::{Axis, Matrix};

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
            Matrix::new([
                [0., 2., 0.], 
                [4., 0., 6.], 
                [0., 8., 9.]
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
        assert_eq!(
            x.step(),
            Matrix::new([[
                0., 0., 0., 1., 1.
            ]])
        );
    }
}
