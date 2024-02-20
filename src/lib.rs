use ndarray::{
    linalg::Dot, ArrayBase, Data, DataMut, Dim, Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn,
    IxDynImpl, NdIndex, RawData,
};
use num::{Signed, Zero};
use pyo3::prelude::*;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, IndexMut, Mul, MulAssign, Sub, SubAssign};

enum UpdateType {
    Additive,
    Multiplicative,
}

struct Loss<R: RawData, D: Dimension> {
    components: Vec<Box<dyn LossComponent<R, D>>>,
}

/// convert an index referred to the flattened array into a vector of indices
/// only row-major order is supported (C)
fn index_to_indices<D>(index: usize, shape: &[usize]) -> &[usize] {
    let mut indices = vec![0; shape.len()];
    let mut index = index;
    for (i, &s) in shape.iter().enumerate().rev() {
        indices[i] = index % s;
        index /= s;
    }
    &indices
}

impl<R, D> Loss<R, D>
where
    R: Data + DataMut,
    D: Dimension,
    R::Elem: Zero + Clone + AddAssign + SubAssign + MulAssign + DivAssign,
    ArrayBase<R, D>: AddAssign,
{
    /// sums the value of each component of the loss or of the updates
    pub(self) fn _sum_of_components<F, T>(&self, compute_component_value: F) -> T
    where
        F: Fn(&Box<dyn LossComponent<R, D>>) -> T,
        T: AddAssign,
    {
        let mut computed = self.components.iter().map(compute_component_value);
        let mut sum = computed.next().unwrap();
        computed.for_each(|c| sum += c);
        sum
    }

    /// updates the matrix W or H using additive or multiplicative update rules
    pub(self) fn _update_matrix<F, T>(
        &self,
        array: &mut ArrayBase<R, D>,
        update_type: &UpdateType,
        component_update_fn: F,
        majorize: bool,
    ) where
        F: Fn(&D, &ArrayBase<R, D>) -> T,
        T: Fn(&Box<dyn LossComponent<R, D>>) -> R::Elem,
    {
        for index in 0..array.len() {
            let coordinates = index_to_indices(index, array.shape());
            let update_value = self._sum_of_components(component_update_fn(&coordinates, &array));
            match update_type {
                UpdateType::Additive => {
                    if majorize {
                        array.get_mut(coordinates).map(|x| *x += update_value);
                    } else {
                        array.get_mut(coordinates).map(|x| *x -= update_value);
                    }
                }
                UpdateType::Multiplicative => {
                    if majorize {
                        array.get_mut(coordinates).map(|x| *x *= update_value);
                    } else {
                        array.get_mut(coordinates).map(|x| *x /= update_value);
                    }
                }
            }
        }
    }

    /// computes the loss
    fn compute(
        &self,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v_guessed: &ArrayBase<R, D>,
        v_target: &ArrayBase<R, D>,
    ) -> f64 {
        self._sum_of_components(|c| c.compute(w, h, v_guessed, v_target))
    }

    fn majorize_w(
        &self,
        w: &mut ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            w,
            update_type,
            |coord, w_| |c| c.majorize_w(coord, w_, h, v),
            true,
        );
    }

    fn majorize_h(
        &self,
        w: &ArrayBase<R, D>,
        h: &mut ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            h,
            update_type,
            |coord, h_| |c| c.majorize_h(coord, w, h_, v),
            true,
        );
    }

    fn minorize_w(
        &self,
        w: &mut ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            w,
            update_type,
            |coord, w_| |c| c.minorize_w(coord, w_, h, v),
            false,
        );
    }

    fn minorize_h(
        &self,
        w: &ArrayBase<R, D>,
        h: &mut ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            h,
            update_type,
            |coord, h_| |c| c.minorize_h(coord, w, h_, v),
            false,
        );
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

    /// This function will be used to increase each element of W during the update; usually it is
    /// the sum of all the negative terms of the gradient in respect to an element of W.
    /// the returned value is the sum of these terms, without the negative sign.
    fn majorize_w(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;

    fn majorize_h(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;

    /// This function will be used to decrease each element of W during the update; usually it is
    /// the sum of all the positive terms of the gradient in respect to an element of W.
    /// the returned value is the sum of these terms, without the negative sign.
    fn minorize_w(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;

    fn minorize_h(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;
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

    fn majorize_w(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> <R as RawData>::Elem {
        todo!()
    }

    fn majorize_h(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> <R as RawData>::Elem {
        todo!()
    }

    fn minorize_w(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> <R as RawData>::Elem {
        todo!()
    }

    fn minorize_h(
        &self,
        coordinates: &Dim<IxDynImpl>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> <R as RawData>::Elem {
        todo!()
    }
}

struct Nmf<R: RawData, D: Dimension> {
    w: ArrayBase<R, D>,
    h: ArrayBase<R, D>,
    v_target: ArrayBase<R, D>,
    loss: Loss<R, D>,
    update_type: UpdateType,
}

impl<R, D> Nmf<R, D>
where
    R: DataMut,
    D: Dimension,
    R::Elem: Zero + Clone + AddAssign + SubAssign + MulAssign + DivAssign,
    ArrayBase<R, D>:
        MulAssign + AddAssign + SubAssign + Dot<ArrayBase<R, D>, Output = ArrayBase<R, D>>,
{
    fn set_w(&mut self, w: ArrayBase<R, D>) {
        self.w = w;
    }
    fn set_h(&mut self, h: ArrayBase<R, D>) {
        self.h = h;
    }
    fn step(&mut self) -> ArrayBase<R, D> {
        let v = self.w.dot(&self.h);
        self.loss
            .majorize_w(&mut self.w, &self.h, &v, &self.update_type);
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
