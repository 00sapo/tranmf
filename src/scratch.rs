use std::ops::{Add, Div, Mul, Sub};

// This trait encapsulates the arithmetic operations for references of a type
pub trait RefElementArithmetic<T>: Sized
where
    for<'a> &'a T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
}

// You would then implement this trait for your specific type that satisfies these constraints.
// In this case, it seems like `R::Elem` should satisfy this, where R is some type that has
// an associated type `Elem`.
// Assume `R` is defined with an associated type `Elem` elsewhere in your code.

use ndarray::RawData;

// You would typically have something like this:
struct R; // Placeholder for the actual definition of R
impl RawData for R {
    type Elem = i32; // Placeholder for the actual element type, for example `i32`.
}

// Implementation of the `RefElementArithmetic` trait for `R::Elem`
impl RefElementArithmetic<R::Elem> for R::Elem {}

// Then, when defining other traits or functions where you want these constraints, you would use:
// <T as RefElementArithmetic<U>>::... or simply T: RefElementArithmetic<U> depending on context.

// Additionally, if you extract `R` to a trait itself, you can even use this new trait in
// generics by combining this trait with others.

// Usage:
fn main() {
    let a = 1.0f64;
    let b = 2.0f64;
    // Call `sum_elements` with references to `f64` (`&f64`).
    let result = sum_elements::<MyR>(&a, &b);
    println!("Result: {}", result); // Prints: Result: 3
}
