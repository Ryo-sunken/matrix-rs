use crate::Matrix;
use num_traits::identities::Zero;

#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix<T> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) val: Vec<T>,
    pub(crate) col_idx: Vec<usize>,
    pub(crate) row_ptr: Vec<usize>,
}

impl<T> SparseMatrix<T>
where
    T: Zero + Clone,
{
    pub fn new<const R: usize, const C: usize>(data: [[T; C]; R]) -> Self {
        let idx_val: Vec<(usize, _)> = data
            .clone()
            .into_iter()
            .flat_map(|row| row.into_iter().enumerate().filter(|(_, x)| !x.is_zero()))
            .collect();

        let mut row_ptr: Vec<_> = data
            .clone()
            .into_iter()
            .map(|row| row.into_iter().filter(|x| !x.is_zero()).count())
            .scan(0, |cum, x| {
                *cum += x;
                Some(*cum)
            })
            .collect();
        row_ptr.insert(0, 0);

        Self {
            rows: R,
            cols: C,
            val: idx_val.clone().into_iter().map(|(_, x)| x).collect(),
            col_idx: idx_val.clone().into_iter().map(|(i, _)| i).collect(),
            row_ptr,
        }
    }

    pub fn to_dence(&self) -> Matrix<T> {
        let mut mat = Matrix::zero(self.rows, self.cols);

        let mut i = 0;
        for n in 0..self.val.len() {
            if n >= self.row_ptr[i + 1] {
                i += 1;
            }
            mat.array[i * self.rows + self.col_idx[n]] = self.val[n].clone();
        }
        mat
    }
}
