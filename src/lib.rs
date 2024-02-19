use ndarray::{linalg::Dot, ArrayBase, Data, Dimension, Ix2, RawData};
use num::{Signed, Zero};
use pyo3::prelude::*;
use std::iter::Sum;
use std::ops::{AddAssign, Mul, MulAssign, Sub};

struct Loss<R: RawData, D: Dimension> {
    components: Vec<Box<dyn LossComponent<R, D>>>,
}

impl<R, D> Loss<R, D>
where
    R: RawData,
    D: Dimension,
    R::Elem: Zero + Clone,
    ArrayBase<R, D>: AddAssign,
{
    fn compute(
        &self,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v_guessed: &ArrayBase<R, D>,
        v_target: &ArrayBase<R, D>,
    ) -> f64 {
        let mut computed = self
            .components
            .iter()
            .map(|c| c.compute(w, h, v_guessed, v_target));
        let mut sum = computed.next().unwrap();
        computed.for_each(|c| sum += c);
        sum
    }
}

/// a Component of the loss; components will be summed; only used for checking tolerance.
trait LossComponent<R: RawData, D: Dimension> {
    fn compute(
        &self,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v_guessed: &ArrayBase<R, D>,
        v_target: &ArrayBase<R, D>,
    ) -> f64;
}

struct LossEuclidean;

impl<R, D> LossComponent<R, D> for LossEuclidean
where
    D: Dimension,
    R: Data,
    R::Elem: Sub<Output = R::Elem> + Mul<Output = R::Elem> + Signed + Copy,
    f64: Sum<R::Elem>,
{
    fn compute(
        &self,
        _w: &ArrayBase<R, D>,
        _h: &ArrayBase<R, D>,
        v_guessed: &ArrayBase<R, D>,
        v_target: &ArrayBase<R, D>,
    ) -> f64 {
        v_guessed
            .iter()
            .zip(v_target.iter())
            .map(|(&x, &y)| (x - y).abs())
            .sum::<f64>()
            / (v_guessed.len() as f64)
    }
}
trait UpdateComponent<R: RawData, D: Dimension> {
    fn update_h(&self, w: &ArrayBase<R, D>, h: &mut ArrayBase<R, D>, v: &ArrayBase<R, D>);

    fn update_w(&self, w: &mut ArrayBase<R, D>, h: &ArrayBase<R, D>, v: &ArrayBase<R, D>);
}

struct MultiplicativeEuclideanUpdates {}

impl<R, D> UpdateComponent<R, D> for MultiplicativeEuclideanUpdates
where
    R: Data,
    D: Dimension,
    R::Elem: Zero + Clone,
    ArrayBase<R, D>: Dot<ArrayBase<R, D>, Output = ArrayBase<R, D>>,
{
    fn update_h(&self, w: &ArrayBase<R, D>, h: &mut ArrayBase<R, D>, v: &ArrayBase<R, D>) {}
    fn update_w(&self, w: &mut ArrayBase<R, D>, h: &ArrayBase<R, D>, v: &ArrayBase<R, D>) {
        let h_t = h.t();
        for i in 0..w.shape()[0] {
            for j in 0..w.shape()[1] {
                let mut numerator = R::Elem::zero();
                let mut denominator = R::Elem::zero();
                for k in 0..h.shape()[1] {
                    numerator += h_t[[k, j]] * v[[i, k]];
                    denominator += h_t[[k, j]] * w[[i, k]].clone();
                }
                w[[i, j]] *= numerator / denominator;
            }
        }
    }
}

struct Updater<R: RawData, D: Dimension> {
    components: Vec<Box<dyn UpdateComponent<R, D>>>,
}

impl<R, D> Updater<R, D>
where
    R: RawData,
    D: Dimension,
    R::Elem: Zero + Clone,
    ArrayBase<R, D>: MulAssign + AddAssign,
{
    fn update(&self, w: &mut ArrayBase<R, D>, h: &mut ArrayBase<R, D>, v: &ArrayBase<R, D>) {
        for component in &self.components {
            component.update_w(w, h, v);
            component.update_h(w, h, v);
        }
    }
}

struct Nmf<R: RawData, D: Dimension> {
    w: ArrayBase<R, D>,
    h: ArrayBase<R, D>,
    v_target: ArrayBase<R, D>,
    loss: Loss<R, D>,
    updater: Updater<R, D>,
}

impl<R, D> Nmf<R, D>
where
    R: RawData,
    D: Dimension,
    R::Elem: Zero + Clone,
    ArrayBase<R, D>: MulAssign + AddAssign + Dot<ArrayBase<R, D>, Output = ArrayBase<R, D>>,
{
    fn set_w(&mut self, w: ArrayBase<R, D>) {
        self.w = w;
    }
    fn set_h(&mut self, h: ArrayBase<R, D>) {
        self.h = h;
    }
    fn add_update_component(&mut self, component: Box<dyn UpdateComponent<R, D>>) {
        self.updater.components.push(component);
    }
    fn step(&mut self) -> ArrayBase<R, D> {
        let v = self.w.dot(&self.h);
        self.updater.update(&mut self.w, &mut self.h, &v);
        v
    }
    fn fit(&mut self, max_iter: usize, tol: f64) {
        let mut loss = f64::INFINITY;
        let mut v_guessed = self.step();
        for _ in 1..max_iter {
            loss = self
                .loss
                .compute(&self.w, &self.h, &v_guessed, &self.v_target);
            if loss < tol {
                break;
            }
            v_guessed = self.step();
        }
    }
}

struct Nmf2D<R: RawData> {
    nmf: Nmf<R, Ix2>,
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
