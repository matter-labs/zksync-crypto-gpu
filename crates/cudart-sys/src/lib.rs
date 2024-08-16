#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod utils;
pub use utils::*;

use std::backtrace::Backtrace;
use std::error::Error;
use std::ffi::CStr;
use std::fmt::{Debug, Display, Formatter};
use std::mem::MaybeUninit;

#[macro_export]
macro_rules! cuda_fn_and_stub {
    ($vis:vis fn $fn:ident($($arg_ident:ident:$arg_ty:ty),*$(,)?) -> $ret:ty;) => {
        #[cfg(not(no_cuda))]
        extern "C" { $vis fn $fn($($arg_ident: $arg_ty),*) -> $ret; }
        /// # Safety
        /// This function is a CUDA function stub.
        #[cfg(no_cuda)]
        #[allow(unused)]
        $vis unsafe extern "C" fn $fn($($arg_ident: $arg_ty),*) -> $ret { unimplemented!("{}", $crate::no_cuda_message!()) }
    };
    ($vis:vis fn $fn:ident($($arg_ident:ident:$arg_ty:ty),*$(,)?);) => {
        #[cfg(not(no_cuda))]
        extern "C" { $vis fn $fn($($arg_ident: $arg_ty),*); }
        /// # Safety
        /// This function is a CUDA function stub.
        #[cfg(no_cuda)]
        #[allow(unused)]
        $vis unsafe extern "C" fn $fn($($arg_ident: $arg_ty),*) { unimplemented!("{}", $crate::no_cuda_message!()) }
    };
}

#[macro_export]
macro_rules! cuda_struct_and_stub {
    ($vis:vis static $name:ident: $type:ty;) => {
        #[cfg(not(no_cuda))]
        extern "C" { $vis static $name: $type; }
        #[cfg(no_cuda)]
        #[allow(non_upper_case_globals)]
        $vis static $name: $type =  unsafe { ::std::mem::zeroed() };
    };
}

include!("bindings.rs");

impl CudaError {
    pub fn eprint_error(self) {
        if self != CudaError::Success {
            eprintln!("CUDA Error: {self}");
        }
    }

    pub fn eprint_error_and_backtrace(self) {
        if self != CudaError::Success {
            self.eprint_error();
            let backtrace = Backtrace::capture();
            eprintln!("Backtrace: {backtrace}");
        }
    }
}

impl Display for CudaError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = unsafe { CStr::from_ptr(cudaGetErrorName(*self)) };
        name.fmt(f)
    }
}

impl Error for CudaError {}

impl Default for CudaMemPoolProperties {
    fn default() -> Self {
        let mut s = MaybeUninit::<Self>::uninit();
        unsafe {
            std::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
