#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Tensor<T, const D: usize> {
    pub(crate) dims: [usize; D],
    pub(crate) array: Vec<T>,
}
