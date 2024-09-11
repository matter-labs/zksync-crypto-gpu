#![feature(allocator_api)]
#![feature(slice_index_methods)]
#![feature(ptr_metadata)]

#[cfg(feature = "sanity")]
pub(crate) const SANITY_CHECK: bool = true;
#[cfg(not(feature = "sanity"))]
pub(crate) const SANITY_CHECK: bool = false;

use bellman::compact_bn256::G1Affine as CompactG1Affine;
use bellman::{CurveProjective, Engine, Field, PrimeField};
pub use fflonk::bellman;

mod allocator;
use allocator::*;

mod dbuffer;
use dbuffer::*;

mod context;
pub use context::*;

mod error;
use error::*;

mod primitives;
use primitives::*;

mod relations;
use relations::*;

mod poly;
use poly::*;

mod prover;
pub use prover::*;

mod utils;
pub(crate) use utils::*;

pub use gpu_ffi;
use gpu_ffi::{bc_event, bc_stream};
use std::alloc::Allocator;

pub use context::{DeviceContext, DeviceContextWithSingleDevice};
