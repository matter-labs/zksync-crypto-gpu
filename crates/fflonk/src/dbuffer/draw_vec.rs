use std::{alloc::Layout, ptr::NonNull};

use gpu_ffi::bc_stream;

use super::{DeviceAllocator, DropOn, GlobalDevice};

enum AllocInit {
    /// The contents of the new memory are uninitialized.
    Uninitialized,
    /// The new memory is guaranteed to be zeroed.
    Zeroed,
}

// A buffer doesn't have a `capacity` property
// since it doesn't need to be shrinked/extended
pub(crate) struct RawDVec<T, A: DeviceAllocator = GlobalDevice> {
    ptr: std::ptr::NonNull<T>,
    len: usize,
    pub(crate) stream: Option<A::Stream>,
    alloc: A,
}

impl<T> RawDVec<T> {
    pub fn dangling() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling(),
            len: 0,
            stream: None,
            alloc: GlobalDevice,
        }
    }
}

impl<T, A: DeviceAllocator> RawDVec<T, A> {
    pub fn allocate_zeroed_on(length: usize, alloc: A, stream: Option<A::Stream>) -> Self {
        Self::inner_allocate_in(length, AllocInit::Zeroed, alloc, stream)
    }

    pub fn allocate_on(length: usize, alloc: A, stream: Option<A::Stream>) -> Self {
        Self::inner_allocate_in(length, AllocInit::Uninitialized, alloc, stream)
    }

    fn inner_allocate_in(
        length: usize,
        init: AllocInit,
        alloc: A,
        stream: Option<A::Stream>,
    ) -> Self {
        let layout = match std::alloc::Layout::array::<T>(length) {
            Ok(layout) => layout,
            Err(_) => panic!("allocation error: length overflow"),
        };

        let result = match stream {
            Some(stream) => match init {
                AllocInit::Uninitialized => alloc.allocate_async(layout, stream),
                AllocInit::Zeroed => alloc.allocate_zeroed_async(layout, stream),
            },
            None => match init {
                AllocInit::Uninitialized => alloc.allocate(layout),
                AllocInit::Zeroed => alloc.allocate_zeroed(layout),
            },
        };

        let ptr = match result {
            Ok(ptr) => ptr,
            Err(err) => panic!("allocation error: {err}"),
        };

        Self {
            ptr: ptr.cast(),
            len: length,
            stream,
            alloc,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub(crate) unsafe fn from_raw_parts_in(
        ptr: *mut T,
        len: usize,
        alloc: A,
        stream: Option<A::Stream>,
    ) -> Self {
        Self {
            ptr: unsafe { std::ptr::NonNull::new_unchecked(ptr) },
            len,
            stream,
            alloc,
        }
    }

    pub(crate) fn ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    fn current_memory(&self) -> Option<(NonNull<u8>, Layout, Option<A::Stream>)> {
        if self.len == 0 {
            None
        } else {
            assert!(std::mem::size_of::<T>() % std::mem::align_of::<T>() == 0);
            unsafe {
                let align = std::mem::align_of::<T>();
                let size = std::mem::size_of::<T>().unchecked_mul(self.len);
                let layout = Layout::from_size_align_unchecked(size, align);
                Some((self.ptr.cast().into(), layout, self.stream))
            }
        }
    }
}

impl<T, A: DeviceAllocator> Drop for RawDVec<T, A> {
    fn drop(&mut self) {
        if let Some((ptr, layout, stream)) = self.current_memory() {
            match stream {
                Some(stream) => self.alloc.deallocate_async(ptr, layout, stream),
                None => self.alloc.deallocate(ptr, layout),
            }
        }
    }
}

impl<T> DropOn for RawDVec<T> {
    fn drop_on(&mut self, stream: bc_stream) {
        if let Some((ptr, layout, inner_stream)) = self.current_memory() {
            assert!(inner_stream.is_some());
            self.alloc.deallocate_async(ptr, layout, stream)
        }
    }
}
