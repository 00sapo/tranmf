use ndarray::{ArrayBase, DataMut, Dimension, Ix2, OwnedRepr, RawData, RawDataClone};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, PyErr};
use std::ops::{Add, Div, Mul, Sub};

mod base;
mod losses;

use base::Arrays;
use base::Elem;
use losses::EuclideanLoss2D;
use losses::Loss;
use losses::UpdateType;

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
    ArrayBase<R, D>: Arrays<R, D>,
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

fn validate_shapes(
    w: &ArrayBase<OwnedRepr<f64>, Ix2>,
    h: &ArrayBase<OwnedRepr<f64>, Ix2>,
    v_target: &ArrayBase<OwnedRepr<f64>, Ix2>,
) -> Result<(), &'static str> {
    if w.shape()[1] != h.shape()[0] {
        return Err("The number of columns in W must equal the number of rows in H for matrix multiplication.");
    }
    if v_target.shape() != [w.shape()[0], h.shape()[1]] {
        return Err("The target matrix V must have dimensions equal to the rows of W and the columns of H respectively.");
    }
    Ok(())
}

#[pyfunction]
fn nmf_2d_f64(
    py: Python<'_>,
    v_target: PyReadonlyArray2<f64>,
    w: PyReadonlyArray2<f64>,
    h: PyReadonlyArray2<f64>,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let v_target = v_target.as_array().to_owned();
    let w = w.as_array().to_owned();
    let h = h.as_array().to_owned();

    validate_shapes(&w, &h, &v_target).map_err(PyErr::new::<PyValueError, _>)?;

    let mut nmf = Nmf {
        w,
        h,
        v_target,
        loss: Loss::<OwnedRepr<f64>, Ix2>::new(vec![Box::new(EuclideanLoss2D {})]),
        update_type: UpdateType::Additive,
        fix_w: false,
        fix_h: false,
    };
    nmf.fit(max_iter, tol);
    let result = nmf.step().into_pyarray(py).to_owned();

    Ok(result)
}

#[pyfunction]
fn set_threads(n: usize) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .unwrap();
}

// TODO: modify
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _rustnmf(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_threads, m)?)?;
    Ok(())
}
