//! Scalar (row-at-a-time) built-in functions, split into families so each family
//! owns its own file. This hub only declares the families and fans out
//! registration; it holds no function implementations itself.

use crate::FunctionRegistry;

pub(crate) mod math;
pub(crate) mod misc;
pub(crate) mod string;

/// Register every scalar family into `reg`. Adding a family is a one-line change
/// here plus its own module; a family with nothing implemented yet is a no-op.
pub(crate) fn register(reg: &mut FunctionRegistry) {
    misc::register(reg);
    string::register(reg);
    math::register(reg);
}
