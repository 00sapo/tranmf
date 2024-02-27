use ndarray::linalg::Dot;
use ndarray::{ArrayBase, Dimension, Ix2, NdProducer, OwnedRepr, RawData, ViewRepr};
use num::{Num, Signed, Zero};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

pub trait Elem<R: RawData<Elem = Self>>: // Make sure R::Elem is Self
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

impl Elem<OwnedRepr<f64>> for f64 {}
impl Elem<OwnedRepr<f32>> for f32 {}

pub trait Arrays<R: RawData, D: Dimension>:
    MulAssign + AddAssign + Dot<ArrayBase<R, D>, Output = ArrayBase<R, D>> + Sync + Send + Sized
where
    R::Elem: Elem<R>,
{
}

impl Arrays<OwnedRepr<f64>, Ix2> for ArrayBase<OwnedRepr<f64>, Ix2> {}
impl Arrays<OwnedRepr<f32>, Ix2> for ArrayBase<OwnedRepr<f32>, Ix2> {}
// impl Arrays<ViewRepr<f64>, Ix2> for ArrayBase<ViewRepr<f64>, Ix2> {}
// impl Arrays<ViewRepr<f32>, Ix2> for ArrayBase<ViewRepr<f32>, Ix2> {}
