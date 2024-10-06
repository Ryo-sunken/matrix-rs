use crate::Matrix;
use num_traits::identities::Zero;

#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix<T> {
    pub(crate) val: Vec<T>,
    pub(crate) col_idx: Vec<usize>,
    pub(crate) row_ptr: Vec<usize>,
}

impl<T> SparseMatrix<T>
where
    T: Zero + Copy,
{
    pub fn new<const R: usize, const C: usize>(data: [[T; C]; R]) -> Self {
        let idx_val: Vec<(usize, T)> = data
            .into_iter()
            .map(|arr| arr.into_iter().enumerate().filter(|(_, x)| !x.is_zero()))
            .flatten()
            .collect();

        let row_ptr = data
            .iter()
            .copied()
            .map(|arr| arr.iter().filter(|&x| !x.is_zero()).count())
            .scan(0, |cum, x| {
                *cum += x;
                Some(*cum)
            })
            .collect();

        Self {
            val: idx_val.iter().copied().map(|(_, x)| x).collect(),
            col_idx: idx_val.iter().copied().map(|(i, _)| i).collect(),
            row_ptr,
        }
    }
}
