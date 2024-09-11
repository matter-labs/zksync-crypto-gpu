use std::{alloc::Layout, ptr::NonNull};

use super::{DeviceAllocator, GlobalDevice};

enum AllocInit {
    /// The contents of the new memory are uninitialized.
    Uninitialized,
    /// The new memory is guaranteed to be zeroed.
    Zeroed,
}

pub(crate) struct RawDDVec<T, A: DeviceAllocator = GlobalDevice> {
    ptr: std::ptr::NonNull<T>,
    cap: usize,
    pub(crate) stream: Option<A::Stream>,
    alloc: A,
}

impl<T, A: DeviceAllocator> RawDDVec<T, A> {
    pub fn new_in(alloc: A) -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling(),
            cap: 0,
            stream: None,
            alloc,
        }
    }

    pub fn with_capacity_zeroed_in(capacity: usize, alloc: A, stream: Option<A::Stream>) -> Self {
        Self::allocate_in(capacity, AllocInit::Zeroed, alloc, stream)
    }

    pub fn with_capacity_in(capacity: usize, alloc: A, stream: Option<A::Stream>) -> Self {
        Self::allocate_in(capacity, AllocInit::Uninitialized, alloc, stream)
    }

    fn allocate_in(capacity: usize, init: AllocInit, alloc: A, stream: Option<A::Stream>) -> Self {
        let layout = match std::alloc::Layout::array::<T>(capacity) {
            Ok(layout) => layout,
            Err(_) => panic!("allocation error: capacity overflow"),
        };

        let result = match stream {
            Some(stream) => match init {
                AllocInit::Uninitialized => alloc.allocate_async(layout, stream),
                AllocInit::Zeroed => alloc.allocate_zeroed_async(layout, stream),
            },
            None => alloc.allocate(layout),
        };

        let ptr = match result {
            Ok(ptr) => ptr,
            Err(err) => panic!("allocation error: {err}"),
        };

        Self {
            ptr: ptr.cast(),
            cap: capacity,
            stream,
            alloc,
        }
    }

    pub(crate) unsafe fn from_raw_parts_in(
        ptr: *mut T,
        capacity: usize,
        alloc: A,
        stream: Option<A::Stream>,
    ) -> Self {
        Self {
            ptr: unsafe { std::ptr::NonNull::new_unchecked(ptr) },
            cap: capacity,
            stream,
            alloc,
        }
    }

    pub(crate) fn ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    fn current_memory(&self) -> Option<(NonNull<u8>, Layout, Option<A::Stream>)> {
        if self.cap == 0 {
            None
        } else {
            assert!(std::mem::size_of::<T>() % std::mem::align_of::<T>() == 0);
            unsafe {
                let align = std::mem::align_of::<T>();
                let size = std::mem::size_of::<T>().unchecked_mul(self.cap);
                let layout = Layout::from_size_align_unchecked(size, align);
                Some((self.ptr.cast().into(), layout, self.stream))
            }
        }
    }
}

impl<T, A: DeviceAllocator> Drop for RawDDVec<T, A> {
    fn drop(&mut self) {
        if let Some((ptr, layout, stream)) = self.current_memory() {
            match stream {
                Some(stream) => self.alloc.deallocate_async(ptr, layout, stream),
                None => self.alloc.deallocate(ptr, layout),
            }
        }
    }
}
