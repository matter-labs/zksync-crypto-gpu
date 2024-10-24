use super::*;

pub trait DeviceAllocator: Default {
    fn allocate(&self, layout: std::alloc::Layout) -> CudaResult<std::ptr::NonNull<[u8]>>;
    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> CudaResult<std::ptr::NonNull<[u8]>>;
    fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout);
    fn allocate_async(
        &self,
        layout: std::alloc::Layout,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> CudaResult<std::ptr::NonNull<[u8]>>;
    fn deallocate_async(
        &self,
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
        stream: bc_stream,
    );
    fn allocate_zeroed_async(
        &self,
        layout: std::alloc::Layout,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> CudaResult<std::ptr::NonNull<[u8]>>;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PoolAllocator;

impl DeviceAllocator for PoolAllocator {
    fn allocate(&self, layout: std::alloc::Layout) -> CudaResult<std::ptr::NonNull<[u8]>> {
        unimplemented!("Pool allocator can't do static allocation/deallocation")
    }

    fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        unimplemented!("Pool allocator can't do static allocation/deallocation")
    }

    fn allocate_async(
        &self,
        layout: std::alloc::Layout,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> CudaResult<std::ptr::NonNull<[u8]>> {
        allocate_async_on(layout.size(), pool, stream)
            .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
            .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    fn deallocate_async(
        &self,
        ptr: std::ptr::NonNull<u8>,
        _layout: std::alloc::Layout,
        stream: bc_stream,
    ) {
        dealloc_async(ptr.as_ptr().cast(), stream).expect("deallocate")
    }

    fn allocate_zeroed(&self, layout: std::alloc::Layout) -> CudaResult<std::ptr::NonNull<[u8]>> {
        let ptr = self.allocate(layout)?;
        Ok(ptr)
    }
    fn allocate_zeroed_async(
        &self,
        layout: std::alloc::Layout,
        pool: bc_mem_pool,
        stream: bc_stream,
    ) -> CudaResult<std::ptr::NonNull<[u8]>> {
        allocate_zeroed_async_on(layout.size(), pool, stream)
            .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
            .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, layout.size()))
    }
}

unsafe impl Send for PoolAllocator {}
unsafe impl Sync for PoolAllocator {}
