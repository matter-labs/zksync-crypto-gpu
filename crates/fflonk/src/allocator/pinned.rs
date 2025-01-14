use super::*;

// Both assembly and device setup has an ability to store data on the pinned memory
// - Assembly uses for the variables(7487741), state and setup columns
// - Device setup uses variable indexes and gate selectors
static mut _STATIC_HOST_ALLOC: Option<GlobalStaticHost> = None;

pub(crate) fn _static_host_alloc() -> GlobalStaticHost {
    unsafe {
        _STATIC_HOST_ALLOC
            .as_ref()
            .expect("initialize static host allocator")
            .clone()
    }
}

pub(crate) fn init_static_host_alloc(domain_size: usize) {
    unsafe {
        // Pinned memory could be initialized before device initialization
        if _STATIC_HOST_ALLOC.is_some() {
            println!("fflonk pinned memory already initialized, ignoring");
            return;
        }
    }
    // Bitmap allocator with small block size and high number of allocations doesn't make
    // sense, and doesn't give good runtime performance compared to default allocator.
    // However it provides satisfying improvement for 3 combined monomials, since prover
    // transfers them them back and forth in case of L4 devices.
    let num_blocks = 3;
    let block_size_in_bytes = 9 * 32 * domain_size;
    let allocator = GlobalStaticHost::init(num_blocks, block_size_in_bytes)
        .expect("initialize static allocator");

    unsafe { _STATIC_HOST_ALLOC = Some(allocator) }
}

pub(crate) fn free_static_host_alloc() {
    unsafe {
        if let Some(alloc) = _STATIC_HOST_ALLOC.take() {
            alloc.free().expect("Couldn't free static allocator");
        }
    }
}

#[derive(Clone)]
pub struct GlobalStaticHost(StaticBitmapAllocator);

impl Default for GlobalStaticHost {
    fn default() -> Self {
        _static_host_alloc()
    }
}

pub trait HostAllocator: Allocator + Default + Clone + Send + Sync + 'static {}

impl GlobalStaticHost {
    pub fn init(num_blocks: usize, block_size_in_bytes: usize) -> CudaResult<Self> {
        assert_ne!(num_blocks, 0);

        let memory_size_in_bytes = num_blocks * block_size_in_bytes;
        let memory = host_allocate(memory_size_in_bytes)
            .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
            .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, memory_size_in_bytes))?;
        println!("allocated {memory_size_in_bytes} bytes on pinned host memory");
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
        host_dealloc(memory.as_mut_ptr().cast())
    }
}

unsafe impl Allocator for GlobalStaticHost {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        self.0.allocate(layout).map_err(|_| std::alloc::AllocError)
    }

    fn allocate_zeroed(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let ptr = self.allocate(layout)?;
        let num_bytes = layout.size();
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr() as *mut u8, 0, layout.size());
            let result = gpu_ffi::bc_memset(ptr.as_ptr().cast(), 0, num_bytes as u64);
            if result != 0 {
                panic!("Couldn't allocate zeroed buffer")
            }
        }

        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        self.0.deallocate(ptr, layout);
    }
}

impl HostAllocator for GlobalStaticHost {}
impl HostAllocator for std::alloc::Global {}
