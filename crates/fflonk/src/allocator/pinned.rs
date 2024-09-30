use super::*;

pub trait HostAllocator: Allocator + Default + Clone + Send + Sync + 'static {}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GlobalHost;
unsafe impl Allocator for GlobalHost {
    fn allocate(
        &self,
        layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        host_allocate(layout.size())
            .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
            .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, layout.size()))
            .map_err(|_| std::alloc::AllocError)
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
        host_dealloc(ptr.as_ptr().cast()).expect("deallocate static buffer")
    }
}

impl HostAllocator for GlobalHost {}
impl HostAllocator for std::alloc::Global {}
