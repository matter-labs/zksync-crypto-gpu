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
    let block_size_in_bytes = std::mem::size_of::<Fr>() * domain_size;
    let allocator = GlobalDeviceStatic::init(num_blocks, block_size_in_bytes)
        .expect("initialize static allocator");

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
pub struct GlobalDeviceStatic(StaticBitmapAllocator);

impl GlobalDeviceStatic {
    pub fn init(num_blocks: usize, block_size_in_bytes: usize) -> CudaResult<Self> {
        assert_ne!(num_blocks, 0);
        assert!(block_size_in_bytes.is_power_of_two());

        let memory_size_in_bytes = num_blocks * block_size_in_bytes;
        let memory = allocate(memory_size_in_bytes)
            .map(|ptr| unsafe { NonNull::new_unchecked(ptr as _) })
            .map(|ptr| NonNull::slice_from_raw_parts(ptr, memory_size_in_bytes))?;

        println!("allocated {memory_size_in_bytes} bytes on device");

        let allocator = StaticBitmapAllocator::init(memory, num_blocks, block_size_in_bytes);
        Ok(Self(allocator))
    }

    pub(crate) fn free(self) -> CudaResult<()> {
        println!("freeing static cuda allocation");
        assert_eq!(std::sync::Arc::weak_count(&self.0.memory.0), 0);
        // TODO
        // assert_eq!(Arc::strong_count(&self.memory), 1);
        let StaticBitmapAllocator { mut memory, .. } = self.0;
        // let memory = Arc::try_unwrap(memory).expect("exclusive access");
        dealloc(memory.as_mut_ptr().cast())
    }
}

impl Default for GlobalDeviceStatic {
    fn default() -> Self {
        _static_alloc()
    }
}

impl DeviceAllocator for GlobalDeviceStatic {
    fn allocate(&self, layout: std::alloc::Layout) -> CudaResult<std::ptr::NonNull<[u8]>> {
        self.0.allocate(layout)
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
        self.0.deallocate(ptr, layout);
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
