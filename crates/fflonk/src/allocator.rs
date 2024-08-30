use super::*;

pub trait HostAllocator: Allocator + Default + Clone + Send + Sync + 'static {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GlobalHost;
unsafe impl Allocator for GlobalHost {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        todo!()
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        todo!()
    }
}

impl HostAllocator for GlobalHost {}
impl HostAllocator for std::alloc::Global {}

pub trait DeviceAllocator: Default {
    type Stream: Copy;
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError>;
    fn allocate_zeroed(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError>;
    fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout);
    fn allocate_async(
        &self,
        layout: std::alloc::Layout,
        stream: Self::Stream,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError>;
    fn deallocate_async(
        &self,
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
        stream: Self::Stream,
    );
    fn allocate_zeroed_async(
        &self,
        layout: std::alloc::Layout,
        stream: Self::Stream,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError>;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GlobalDevice;

impl DeviceAllocator for GlobalDevice {
    type Stream = bc_stream;

    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        allocate(layout.size())
            .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
            .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, layout.size()))
            .map_err(|_| std::alloc::AllocError)
    }

    fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        dealloc(ptr.as_ptr().cast()).expect("deallocate static buffer")
    }

    fn allocate_async(
        &self,
        layout: std::alloc::Layout,
        stream: Self::Stream,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        assert!(is_context_initialized());
        allocate_async(layout.size(), stream)
            .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
            .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, layout.size()))
            .map_err(|_| std::alloc::AllocError)
    }

    fn deallocate_async(
        &self,
        ptr: std::ptr::NonNull<u8>,
        _layout: std::alloc::Layout,
        stream: Self::Stream,
    ) {
        assert!(is_context_initialized());
        dealloc_async(ptr.as_ptr().cast(), stream).expect("deallocate")
    }

    fn allocate_zeroed(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let ptr = self.allocate(layout)?;
        let stream = bc_stream::new().unwrap();
        unsafe {
            let result =
                gpu_ffi::ff_set_value_zero(ptr.as_ptr().cast(), layout.size() as u32, stream);
            if result != 0 {
                panic!("Couldn't allocate zeroed buffer")
            }
        }
        stream.sync().unwrap();
        Ok(ptr)
    }
    fn allocate_zeroed_async(
        &self,
        layout: std::alloc::Layout,
        stream: Self::Stream,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let ptr = self.allocate_async(layout, stream)?;
        // SAFETY: `alloc` returns a valid memory block
        unsafe {
            let result =
                gpu_ffi::ff_set_value_zero(ptr.as_ptr().cast(), layout.size() as u32, stream);
            if result != 0 {
                panic!("Couldn't allocate zeroed buffer")
            }
        }
        Ok(ptr)
    }
}

unsafe impl Send for GlobalDevice {}
unsafe impl Sync for GlobalDevice {}
