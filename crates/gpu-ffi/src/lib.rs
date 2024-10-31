#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![feature(array_chunks)]
pub mod error;
pub mod utils;

pub mod bindings;
pub mod bindings_extra;
pub mod other;
pub mod wrapper;

pub use bindings::*;
pub use bindings_extra::*;
pub use other::*;

pub use error::*;
pub use utils::*;
pub use wrapper::*;

use std::ffi::c_void;
use std::ptr::addr_of_mut;

const FIELD_ELEMENT_LEN: usize = 32;
const LDE_FACTOR: usize = 4;

mod arithmetic;
mod msm;
mod ntt;

pub use arithmetic::*;
pub use msm::*;
pub use ntt::*;
