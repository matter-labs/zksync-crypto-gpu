use std::{alloc::Layout, ptr::NonNull};

use gpu_ffi::{bc_mem_pool, bc_stream};

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
    pub(crate) pool: Option<bc_mem_pool>,
    pub(crate) stream: Option<bc_stream>,
    alloc: A,
}

impl<T> RawDVec<T> {
    pub fn dangling() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling(),
            len: 0,
            pool: None,
            stream: None,
            alloc: GlobalDevice,
        }
    }
}

impl<T, A: DeviceAllocator> RawDVec<T, A> {
    pub fn allocate_zeroed(length: usize, alloc: A) -> Self {
        Self::inner_allocate_in(length, AllocInit::Zeroed, alloc, None, None)
    }
    pub fn allocate_zeroed_on(
        length: usize,
        alloc: A,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> Self {
        Self::inner_allocate_in(length, AllocInit::Zeroed, alloc, Some(pool), Some(stream))
    }

    pub fn allocate_on(length: usize, alloc: A, pool: bc_mem_pool, stream: bc_stream) -> Self {
        Self::inner_allocate_in(
            length,
            AllocInit::Uninitialized,
            alloc,
            Some(pool),
            Some(stream),
        )
    }

    fn inner_allocate_in(
        length: usize,
        init: AllocInit,
        alloc: A,
        pool: Option<bc_mem_pool>,
        stream: Option<bc_stream>,
    ) -> Self {
        let layout = match std::alloc::Layout::array::<T>(length) {
            Ok(layout) => layout,
            Err(_) => panic!("allocation error: length overflow"),
        };

        let result = match (pool, stream) {
            (Some(pool), Some(stream)) => match init {
                AllocInit::Uninitialized => alloc.allocate_async(layout, pool, stream),
                AllocInit::Zeroed => alloc.allocate_zeroed_async(layout, pool, stream),
            },
            (None, None) => match init {
                AllocInit::Uninitialized => alloc.allocate(layout),
                AllocInit::Zeroed => alloc.allocate_zeroed(layout),
            },
            _ => unimplemented!(),
        };

        let ptr = match result {
            Ok(ptr) => ptr,
            Err(err) => panic!("allocation error: {:?}", err),
        };

        Self {
            ptr: ptr.cast(),
            len: length,
            stream,
            pool,
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
        pool: Option<bc_mem_pool>,
        stream: Option<bc_stream>,
    ) -> Self {
        Self {
            ptr: unsafe { std::ptr::NonNull::new_unchecked(ptr) },
            len,
            pool,
            stream,
            alloc,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr().cast()
    }

    fn current_memory(&self) -> Option<(NonNull<u8>, Layout, Option<bc_stream>)> {
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

impl<T, A: DeviceAllocator> DropOn for RawDVec<T, A> {
    fn drop_on(&mut self, stream: bc_stream) {
        if let Some((ptr, layout, inner_stream)) = self.current_memory() {
            assert!(inner_stream.is_some());
            self.alloc.deallocate_async(ptr, layout, stream)
        }
    }
}
