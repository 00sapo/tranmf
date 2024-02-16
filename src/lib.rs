use ndarray::{Array2, ArrayBase, Dimension, Ix2, RawData};
use pyo3::prelude::*;

type Float = f64;

struct Loss<D: Dimension, P> {
    Precision: Float,
    components: Vec<Box<dyn LossComponent<D, Precision = P>>>,
}

impl Loss<Ix2, f64> {
    fn compute(&self, v: &Array2<f64>) -> f64 {
        self.components.iter().map(|c| c.compute(v)).sum()
    }

    fn gradient(&self, v: &Array2<f64>) -> Array2<f64> {
        self.components
            .iter()
            .map(|c| c.gradient(v))
            .fold(Array2::zeros(v.dim()), |acc, x| acc + x)
    }
}

trait LossComponent<D: Dimension>
where
    Self::Precision: RawData,
{
    type Precision;
    fn compute(&self, v: Vec<ArrayBase<Self::Precision, D>>) -> Self::Precision;
    fn gradient(&self, v: Vec<ArrayBase<Self::Precision, D>>) -> Self::Precision;
}

trait NMF<D: Dimension>
where
    Self::Precision: RawData,
{
    type Precision;
    fn run(&self, max_iter: usize);
    fn random_initialize(&mut self);
    fn set_w(&mut self, w: &ArrayBase<Self::Precision, D>);
    fn get_w(&self) -> &ArrayBase<Self::Precision, D>;
    fn set_h(&mut self, h: ArrayBase<Self::Precision, D>);
    fn get_h(&self) -> &ArrayBase<Self::Precision, D>;
    fn set_loss_fn(&mut self, loss_fn: Loss<D, Self::Precision>);
    fn get_loss_fn(&self) -> Loss<D, Self::Precision>;
}

// TODO: modify
/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

// TODO: modify
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _rustnmf(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
