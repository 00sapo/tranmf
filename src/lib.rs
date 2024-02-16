use ndarray::{Array, ArrayBase, Data, DataOwned, Dimension, Ix2, OwnedRepr, RawData};
use num::Zero;
use pyo3::prelude::*;
use std::ops::{AddAssign, MulAssign};

struct Loss<R: RawData, D: Dimension> {
    components: Vec<Box<dyn LossComponent<R, D>>>,
}

impl<R, D> Loss<R, D>
where
    R: Data,
    D: Dimension,
    R::Elem: Zero + Clone,
    ArrayBase<R, D>: AddAssign,
{
    fn compute(&self, v: &[ArrayBase<R, D>]) -> ArrayBase<R, D> {
        let shape = v.get(0).unwrap().raw_dim();
        let mut computed = self.components.iter().map(|c| c.compute(v));
        let mut sum = computed.next().unwrap();
        computed.for_each(|c| sum += c);
        sum
    }

    fn gradient(&self, v: &[ArrayBase<R, D>]) -> ArrayBase<R, D> {
        let shape = v.get(0).unwrap().raw_dim();
        let mut computed = self.components.iter().map(|c| c.gradient(v));
        let mut sum = computed.next().unwrap();
        computed.for_each(|c| sum += c);
        sum
    }
}

trait LossComponent<R: RawData, D: Dimension> {
    fn compute(&self, v: &[ArrayBase<R, D>]) -> ArrayBase<R, D>;
    fn gradient(&self, v: &[ArrayBase<R, D>]) -> ArrayBase<R, D>;
}

trait UpdateComponent<R: RawData, D: Dimension> {
    fn update_h(
        &self,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        gradient: &ArrayBase<R, D>,
    ) -> ArrayBase<R, D>;

    fn update_w(
        &self,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        gradient: &ArrayBase<R, D>,
    ) -> ArrayBase<R, D>;
}

enum UpdateType {
    Multiplicative,
    Additive,
}

struct Updater<R: RawData, D: Dimension> {
    update_type: UpdateType,
    components: Vec<Box<dyn UpdateComponent<R, D>>>,
}

impl<R, D> Updater<R, D>
where
    R: Data,
    D: Dimension,
    R::Elem: Zero + Clone,
    ArrayBase<R, D>: MulAssign + AddAssign,
{
    fn update(&self, w: &mut ArrayBase<R, D>, h: &mut ArrayBase<R, D>, gradient: &ArrayBase<R, D>) {
        for component in &self.components {
            let update_w = component.update_w(w, h, gradient);
            let update_h = component.update_h(w, h, gradient);
            match self.update_type {
                UpdateType::Multiplicative => {
                    *w *= update_w;
                    *h *= update_h;
                }
                UpdateType::Additive => {
                    *w += update_w;
                    *h += update_h;
                }
            }
        }
    }
}

struct NmfBase<R: RawData, D: Dimension> {
    w: ArrayBase<R, D>,
    h: ArrayBase<R, D>,
    loss: Loss<R, D>,
    updater: Updater<R, D>,
}

impl<R: RawData, D: Dimension> NmfBase<R, D> {
    fn random_initialize(&mut self) {}
    fn add_update_component(&mut self, component: Box<dyn UpdateComponent<R, D>>) {
        self.updater.components.push(component);
    }
    fn add_loss_component(&mut self, component: Box<dyn LossComponent<R, D>>) {
        self.loss.components.push(component);
    }
}

trait NmfComputer<R: RawData, D: Dimension> {
    fn compute(&self, nmf: NmfBase<R, D>) -> ArrayBase<R, D>;
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
