use ndarray::{
    indices_of, linalg::Dot, ArrayBase, Data, DataMut, Dimension, IntoDimension, Ix2, NdProducer,
    RawData, RawDataClone, Zip,
};
use num::{Num, Signed, Zero};
use pyo3::prelude::*;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

trait Elem<R: RawData<Elem = Self>>: // Make sure R::Elem is Self
    Zero
    + Clone
    + Signed
    + Add<Self, Output = Self>
    + AddAssign
    + Sub<Self, Output = Self>
    + SubAssign
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Num
    + Sync
    + Send
    + Sized
{
}

enum UpdateType {
    Additive,
    Multiplicative,
}

struct Loss<R: RawData, D: Dimension> {
    components: Vec<Box<dyn LossComponent<R, D>>>,
}

impl<R, D> Loss<R, D>
where
    R: Data + DataMut + RawDataClone + Sync,
    D: Dimension + Copy,
    D::Pattern: Send,
    R::Elem: Elem<R>,
    ArrayBase<R, D>: AddAssign + Sync + Send + NdProducer,
    for<'a> &'a R::Elem: Add<&'a R::Elem, Output = R::Elem>
        + Sub<&'a R::Elem, Output = R::Elem>
        + Mul<&'a R::Elem, Output = R::Elem>
        + Div<&'a R::Elem, Output = R::Elem>,
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
    pub(self) fn _update_matrix<F>(
        &self,
        output: &mut ArrayBase<R, D>,
        array: &ArrayBase<R, D>,
        update_type: &UpdateType,
        component_update_fn: F,
        majorize: bool,
    ) where
        F: Fn(&D, &ArrayBase<R, D>, &Box<dyn LossComponent<R, D>>) -> R::Elem + Sync + Send,
    {
        // use a Zip because it supports rayon's par_map_assign_into
        Zip::from(array.view())
            .and(indices_of(array))
            .par_map_assign_into(output, move |element, coordinates| {
                // let coordinates = index_to_indices::<D>(index, array.shape());
                let cc = coordinates.into_dimension();
                let update_value =
                    &self._sum_of_components(|c| component_update_fn(&cc, &array, c));
                match update_type {
                    UpdateType::Additive => {
                        if majorize {
                            element + update_value
                        } else {
                            element - update_value
                        }
                    }
                    UpdateType::Multiplicative => {
                        if majorize {
                            element * update_value
                        } else {
                            element / update_value
                        }
                    }
                }
            });
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
        output: &mut ArrayBase<R, D>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            output,
            w,
            update_type,
            |coord, w_, c| c.majorize_w(coord, w_, h, v),
            true,
        )
    }

    fn majorize_h(
        &self,
        output: &mut ArrayBase<R, D>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            output,
            h,
            update_type,
            |coord, h_, c| c.majorize_h(coord, w, h_, v),
            true,
        );
    }

    fn minorize_w(
        &self,
        output: &mut ArrayBase<R, D>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            output,
            w,
            update_type,
            |coord, w_, c| c.minorize_w(coord, w_, h, v),
            false,
        );
    }

    fn minorize_h(
        &self,
        output: &mut ArrayBase<R, D>,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
        update_type: &UpdateType,
    ) {
        self._update_matrix(
            output,
            h,
            update_type,
            |coord, h_, c| c.minorize_h(coord, w, h_, v),
            false,
        );
    }
}

/// a Component of the loss; components will be summed; only used for checking tolerance.
trait LossComponent<R, D>: Sync
where
    R: RawData,
    D: Dimension,
{
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
        coordinates: &D,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;

    fn majorize_h(
        &self,
        coordinates: &D,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;

    /// This function will be used to decrease each element of W during the update; usually it is
    /// the sum of all the positive terms of the gradient in respect to an element of W.
    /// the returned value is the sum of these terms, without the negative sign.
    fn minorize_w(
        &self,
        coordinates: &D,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;

    fn minorize_h(
        &self,
        coordinates: &D,
        w: &ArrayBase<R, D>,
        h: &ArrayBase<R, D>,
        v: &ArrayBase<R, D>,
    ) -> R::Elem;
}

struct EuclideanLoss2D;

impl<R> LossComponent<R, Ix2> for EuclideanLoss2D
where
    R: Data,
    R::Elem: Elem<R>,
    f64: std::iter::Sum<<R as ndarray::RawData>::Elem>,
    for<'a> &'a R::Elem: Add<&'a R::Elem, Output = R::Elem>
        + Sub<&'a R::Elem, Output = R::Elem>
        + Mul<&'a R::Elem, Output = R::Elem>
        + Div<&'a R::Elem, Output = R::Elem>,
{
    fn compute(
        &self,
        _w: &ArrayBase<R, Ix2>,
        _h: &ArrayBase<R, Ix2>,
        v_guessed: &ArrayBase<R, Ix2>,
        v_target: &ArrayBase<R, Ix2>,
    ) -> f64 {
        v_guessed
            .iter()
            .zip(v_target.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f64>()
            / (v_guessed.len() as f64)
    }

    fn majorize_w(
        &self,
        coordinates: &Ix2,
        _w: &ArrayBase<R, Ix2>,
        h: &ArrayBase<R, Ix2>,
        v: &ArrayBase<R, Ix2>,
    ) -> <R as RawData>::Elem {
        // (VH)_ij = (\sum_k(V_ik*H_kj))_ij
        // (VH^T)_ij =
        let mut sum = R::Elem::zero();
        let i = coordinates[0];
        let j = coordinates[1];
        for k in 0..h.shape()[1] {
            sum += v.get([i, k]).unwrap() * h.get([j, k]).unwrap();
        }
        sum
    }

    fn majorize_h(
        &self,
        coordinates: &Ix2,
        w: &ArrayBase<R, Ix2>,
        _h: &ArrayBase<R, Ix2>,
        v: &ArrayBase<R, Ix2>,
    ) -> <R as RawData>::Elem {
        // (W^TV)_ij
        let mut sum = R::Elem::zero();
        let i = coordinates[0];
        let j = coordinates[1];
        for k in 0..w.shape()[0] {
            sum += w.get([k, i]).unwrap() * v.get([k, j]).unwrap();
        }
        sum
    }

    fn minorize_w(
        &self,
        coordinates: &Ix2,
        w: &ArrayBase<R, Ix2>,
        h: &ArrayBase<R, Ix2>,
        _v: &ArrayBase<R, Ix2>,
    ) -> <R as RawData>::Elem {
        // (W^TWH)_ij = ((W^TW)H)_ij = (\sum_k((W^TW)_ik*H_kj))_ij =
        // = (\sum_k(\sum_l(W_li*W_lk)*H_kj))_ij
        let mut sum = R::Elem::zero();
        let i = coordinates[0];
        let j = coordinates[1];
        for k in 0..w.shape()[0] {
            let mut w_sum = R::Elem::zero();
            for l in 0..w.shape()[1] {
                w_sum += w.get([l, i]).unwrap() * w.get([l, k]).unwrap();
            }
            sum += &w_sum * h.get([k, j]).unwrap();
        }
        sum
    }

    fn minorize_h(
        &self,
        coordinates: &Ix2,
        w: &ArrayBase<R, Ix2>,
        h: &ArrayBase<R, Ix2>,
        _v: &ArrayBase<R, Ix2>,
    ) -> <R as RawData>::Elem {
        // WHH^T)_ij = (W(HH^T))_ij = (\sum_k(W_ik*(HH^T)_kj))_ij =
        // = (\sum_k(W_ik*\sum_l(H_kl*H_jl)))_ij
        let mut sum = R::Elem::zero();
        let i = coordinates[0];
        let j = coordinates[1];
        for k in 0..h.shape()[1] {
            let mut h_sum = R::Elem::zero();
            for l in 0..h.shape()[0] {
                h_sum += h.get([k, l]).unwrap() * h.get([j, l]).unwrap();
            }
            sum += w.get([i, k]).unwrap() * &h_sum;
        }
        sum
    }
}

struct Nmf<R: RawData, D: Dimension> {
    w: ArrayBase<R, D>,
    h: ArrayBase<R, D>,
    v_target: ArrayBase<R, D>,
    loss: Loss<R, D>,
    update_type: UpdateType,
    fix_w: bool,
    fix_h: bool,
}

impl<R, D> Nmf<R, D>
where
    R: RawDataClone + DataMut + Sync,
    R::Elem: Elem<R>,
    D: Dimension + Copy,
    D::Pattern: Send,
    ArrayBase<R, D>: MulAssign
        + AddAssign
        + SubAssign
        + Dot<ArrayBase<R, D>, Output = ArrayBase<R, D>>
        + NdProducer
        + Sync
        + Send,
    for<'a> &'a R::Elem: Add<&'a R::Elem, Output = R::Elem>
        + Sub<&'a R::Elem, Output = R::Elem>
        + Mul<&'a R::Elem, Output = R::Elem>
        + Div<&'a R::Elem, Output = R::Elem>,
{
    fn set_w(&mut self, w: ArrayBase<R, D>) {
        self.w = w;
    }
    fn set_h(&mut self, h: ArrayBase<R, D>) {
        self.h = h;
    }
    fn freeze_w(&mut self) {
        self.fix_w = true;
    }
    fn freeze_h(&mut self) {
        self.fix_h = true;
    }
    fn free_w(&mut self) {
        self.fix_w = false;
    }
    fn free_h(&mut self) {
        self.fix_h = false;
    }
    fn step(&mut self) -> ArrayBase<R, D> {
        let v = self.w.dot(&self.h);

        if !self.fix_w {
            let mut new_w = self.w.clone();
            self.loss
                .majorize_w(&mut new_w, &self.w, &self.h, &v, &self.update_type);
            self.loss
                .minorize_w(&mut new_w, &self.w, &self.h, &v, &self.update_type);
            self.w = new_w;
        }

        if !self.fix_h {
            let mut new_h = self.h.clone();
            self.loss
                .majorize_h(&mut new_h, &self.w, &self.h, &v, &self.update_type);
            self.loss
                .minorize_h(&mut new_h, &self.w, &self.h, &v, &self.update_type);
            self.h = new_h;
        }
        v
    }
    fn fit(&mut self, max_iter: usize, tol: f64) {
        let mut v_guessed = self.step();
        for _ in 1..max_iter {
            let loss = self
                .loss
                .compute(&self.w, &self.h, &v_guessed, &self.v_target);
            if loss < tol {
                break;
            }
            v_guessed = self.step();
        }
    }
}

fn set_threads(n: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap();
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
