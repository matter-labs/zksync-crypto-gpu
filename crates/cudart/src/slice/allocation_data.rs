use std::marker::PhantomData;
use std::ptr::NonNull;
use std::slice;

use crate::slice::{CudaSlice, CudaSliceMut};

#[derive(Debug)]
pub(crate) struct AllocationData<T> {
    pub ptr: NonNull<T>,
    pub len: usize,
    _owns_t: PhantomData<T>,
}

impl<T> AllocationData<T> {
    pub fn new(ptr: NonNull<T>, len: usize) -> Self {
        Self {
            ptr,
            len,
            _owns_t: PhantomData,
        }
    }

    pub unsafe fn new_unchecked(ptr: *mut T, len: usize) -> Self {
        Self {
            ptr: NonNull::new_unchecked(ptr),
            len,
            _owns_t: PhantomData,
        }
    }
}

unsafe impl<T> Send for AllocationData<T> where Vec<T>: Send {}

unsafe impl<T> Sync for AllocationData<T> where Vec<T>: Sync {}

impl<T> CudaSlice<T> for AllocationData<T> {
    unsafe fn as_slice(&self) -> &[T] {
        slice::from_raw_parts(self.ptr.as_ptr(), self.len)
    }
}

impl<T> CudaSliceMut<T> for AllocationData<T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len)
    }
}
