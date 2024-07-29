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
}
