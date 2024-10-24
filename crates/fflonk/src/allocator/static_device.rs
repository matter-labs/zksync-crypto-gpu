use std::{cell::RefCell, ptr::NonNull, rc::Rc};

use bellman::bn256::Fr;

use super::*;

static mut _STATIC_ALLOC: Option<GlobalDeviceStatic> = None;

pub(crate) fn _static_alloc() -> GlobalDeviceStatic {
    unsafe {
        _STATIC_ALLOC
            .as_ref()
            .expect("initialize static allocator")
            .clone()
    }
}

pub(crate) fn init_static_alloc(domain_size: usize) {
    let num_blocks = Device::static_alloc_num_blocks();
    let allocator =
        GlobalDeviceStatic::init(num_blocks, domain_size).expect("initialize static allocator");

    unsafe { _STATIC_ALLOC = Some(allocator) }
}

pub(crate) fn free_static_alloc() {
    unsafe {
        let alloc = _STATIC_ALLOC.take();
        alloc
            .unwrap()
            .free()
            .expect("Couldn't free static allocator");
    }
}

#[derive(Clone)]
pub struct GlobalDeviceStatic {
    memory: Rc<NonNull<[u8]>>,
    memory_size: usize,
    block_size_in_bytes: usize,
    bitmap: Rc<RefCell<Vec<bool>>>,
}

impl GlobalDeviceStatic {
    fn init_bitmap(num_blocks: usize) -> Vec<bool> {
        vec![false; num_blocks]
    }

    pub fn init(num_blocks: usize, block_size: usize) -> CudaResult<Self> {
        assert_ne!(num_blocks, 0);
        assert!(block_size.is_power_of_two());

        let memory_size = num_blocks * block_size;
        let memory_size_in_bytes = memory_size * size_of::<Fr>();
        let block_size_in_bytes = block_size * size_of::<Fr>();

        let memory = allocate(memory_size_in_bytes)
            .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
            .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, memory_size_in_bytes))?;

        println!("allocated {memory_size_in_bytes} bytes on device");

        let alloc = GlobalDeviceStatic {
            memory: Rc::new(memory),
            memory_size: memory_size_in_bytes,
            block_size_in_bytes,
            bitmap: Rc::new(RefCell::new(Self::init_bitmap(num_blocks))),
        };

        return Ok(alloc);
    }
    fn as_ptr(&self) -> *const u8 {
        self.memory.as_ptr().cast()
    }

    fn find_free_block(&self) -> Option<usize> {
        for (idx, entry) in self.bitmap.borrow_mut().iter_mut().enumerate() {
            if !*entry {
                *entry = true;
                return Some(idx);
            }
        }
        None
    }

    #[allow(unreachable_code)]
    fn find_adjacent_free_blocks(
        &self,
        requested_num_blocks: usize,
    ) -> Option<std::ops::Range<usize>> {
        let mut bitmap = self.bitmap.borrow_mut();
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

    fn free_blocks(&self, index: usize, num_blocks: usize) {
        assert!(num_blocks > 0);
        let mut guard = self.bitmap.borrow_mut();
        for i in index..index + num_blocks {
            guard[i] = false;
        }
    }

    pub fn free(self) -> CudaResult<()> {
        println!("freeing static cuda allocation");
        assert_eq!(Rc::weak_count(&self.memory), 0);
        // TODO
        // assert_eq!(Rc::strong_count(&self.memory), 1);
        let Self { memory, .. } = self;
        // let memory = Rc::try_unwrap(memory).expect("exclusive access");
        dealloc(memory.as_ptr().cast())?;
        Ok(())
    }
}

impl Default for GlobalDeviceStatic {
    fn default() -> Self {
        _static_alloc()
    }
}

impl DeviceAllocator for GlobalDeviceStatic {
    fn allocate(&self, layout: std::alloc::Layout) -> CudaResult<std::ptr::NonNull<[u8]>> {
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

    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> CudaResult<std::ptr::NonNull<[u8]>> {
        let ptr = self.allocate(layout)?;
        let num_bytes = layout.size();
        unsafe {
            let result = gpu_ffi::bc_memset(ptr.as_ptr().cast(), 0, num_bytes as u64);
            if result != 0 {
                panic!("Couldn't allocate zeroed buffer")
            }
        }

        Ok(ptr)
    }

    fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
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

    fn allocate_async(
        &self,
        layout: std::alloc::Layout,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> CudaResult<std::ptr::NonNull<[u8]>> {
        unreachable!("Can only statically allocate")
    }

    fn deallocate_async(
        &self,
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
        stream: bc_stream,
    ) {
        unreachable!("Can only statically allocate")
    }

    fn allocate_zeroed_async(
        &self,
        layout: std::alloc::Layout,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> CudaResult<std::ptr::NonNull<[u8]>> {
        unreachable!("Can only statically allocate")
    }
}
