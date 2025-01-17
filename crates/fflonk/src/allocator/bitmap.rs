use std::{
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use super::*;

#[derive(Clone)]
pub(crate) struct UnsafeNonNullPtr(pub(crate) Arc<NonNull<[u8]>>);
unsafe impl Send for UnsafeNonNullPtr {}
unsafe impl Sync for UnsafeNonNullPtr {}

impl UnsafeNonNullPtr {
    pub(crate) fn new(ptr: NonNull<[u8]>) -> Self {
        Self(Arc::new(ptr))
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr().cast()
    }
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_ptr().cast()
    }
}

#[derive(Clone)]
pub(crate) struct StaticBitmapAllocator {
    pub(crate) memory: UnsafeNonNullPtr,
    pub(crate) memory_size: usize,
    pub(crate) block_size_in_bytes: usize,
    pub(crate) bitmap: Arc<Mutex<Vec<bool>>>,
}

impl StaticBitmapAllocator {
    pub(crate) fn init(
        memory: NonNull<[u8]>,
        num_blocks: usize,
        block_size_in_bytes: usize,
    ) -> Self {
        let memory_size_in_bytes = num_blocks * block_size_in_bytes;
        Self {
            memory: UnsafeNonNullPtr::new(memory),
            memory_size: memory_size_in_bytes,
            block_size_in_bytes,
            bitmap: Arc::new(Mutex::new(vec![false; num_blocks])),
        }
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.memory.as_ptr().cast()
    }

    pub(crate) fn find_free_block(&self) -> Option<usize> {
        for (idx, entry) in self.bitmap.lock().unwrap().iter_mut().enumerate() {
            if !*entry {
                *entry = true;
                return Some(idx);
            }
        }
        None
    }

    #[allow(unreachable_code)]
    pub(crate) fn find_adjacent_free_blocks(
        &self,
        requested_num_blocks: usize,
    ) -> Option<std::ops::Range<usize>> {
        let mut bitmap = self.bitmap.lock().unwrap();
        if requested_num_blocks > bitmap.len() {
            return None;
        }
        let _range_of_blocks_found = false;
        let _found_range = 0..0;

        let mut start = 0;
        let mut end = requested_num_blocks;
        let mut busy_block_idx = 0;
        loop {
            let mut has_busy_block = false;
            for (idx, sub_entry) in bitmap[start..end].iter().copied().enumerate() {
                if sub_entry {
                    has_busy_block = true;
                    busy_block_idx = start + idx;
                }
            }
            if !has_busy_block {
                for entry in bitmap[start..end].iter_mut() {
                    *entry = true;
                }
                return Some(start..end);
            } else {
                start = busy_block_idx + 1;
                end = start + requested_num_blocks;
                if end > bitmap.len() {
                    break;
                }
            }
        }
        // panic!("not found block {} {} {}", start, end, self.bitmap.len());
        None
    }

    pub(crate) fn free_blocks(&self, index: usize, num_blocks: usize) {
        assert!(num_blocks > 0);
        let mut guard = self.bitmap.lock().unwrap();
        for i in index..index + num_blocks {
            guard[i] = false;
        }
    }

    pub(crate) fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> CudaResult<std::ptr::NonNull<[u8]>> {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
        let num_blocks = size / self.block_size_in_bytes;

        if size > self.block_size_in_bytes {
            if let Some(range) = self.find_adjacent_free_blocks(num_blocks) {
                let index = range.start;
                let offset = index * self.block_size_in_bytes;
                let ptr = unsafe { self.as_ptr().add(offset) };
                let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
                return Ok(NonNull::slice_from_raw_parts(ptr, size));
            }
            panic!("allocation of {} blocks has failed", num_blocks);
            // return Err(CudaError::AllocationError(format!(
            //     "allocation of {} blocks has failed",
            //     num_blocks
            // )));
        }

        if let Some(index) = self.find_free_block() {
            let offset = index * self.block_size_in_bytes;
            let ptr = unsafe { self.as_ptr().add(offset) };
            let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
            Ok(NonNull::slice_from_raw_parts(ptr, size))
        } else {
            panic!("allocation of 1 block has failed");
            // return Err(CudaError::AllocationError(format!(
            //     "allocation of 1 block has failed",
            // )));
        }
    }

    pub(crate) fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
        let offset = unsafe { ptr.as_ptr().offset_from(self.as_ptr()) } as usize;
        if offset >= self.memory_size {
            return;
        }
        assert_eq!(offset % self.block_size_in_bytes, 0);
        let index = offset / self.block_size_in_bytes;
        let num_blocks = size / self.block_size_in_bytes;
        self.free_blocks(index, num_blocks);
    }
}
