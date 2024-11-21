use super::*;
// 
// // Both assembly and device setup has an ability to store data on the pinned memory
// // - Assembly uses for the variables(7487741), state and setup columns
// // - Device setup uses variable indexes and gate selectors
// static mut _STATIC_HOST_ALLOC: Option<GlobalHost> = None;
// 
// #[derive(Clone, Debug, Default, Eq, PartialEq)]
// pub struct GlobalHost;
// 
// impl GlobalHost {
//     pub fn init(domain_size: usize) -> CudaResult<Self> {
//         let num_variables = 0;
//         let num_cols = 3;
// 
//         let size_of_indexes_in_bytes = 8 * num_cols * domain_size;
//         let size_of_vars_in_bytes = 32 * num_variables;
// 
//         let total_size_in_bytes = size_of_indexes_in_bytes + size_of_vars_in_bytes;
// 
//         todo!()
//     }
// }
// 
pub trait HostAllocator: Allocator + Default + Clone + Send + Sync + 'static {}
// 
// unsafe impl Allocator for GlobalHost {
//     fn allocate(
//         &self,
//         layout: std::alloc::Layout,
//     ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
//         let size = layout.size();
//         dbg!("allocate", size);
//         host_allocate(size)
//             .map(|ptr| unsafe { std::ptr::NonNull::new_unchecked(ptr as _) })
//             .map(|ptr| std::ptr::NonNull::slice_from_raw_parts(ptr, size))
//             .map_err(|_| std::alloc::AllocError)
//     }
// 
//     unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: std::alloc::Layout) {
//         let size = layout.size();
//         dbg!("deallocate", size);
//         host_dealloc(ptr.as_ptr().cast()).expect("deallocate static buffer")
//     }
// }
// 
// impl HostAllocator for GlobalHost {}
impl HostAllocator for std::alloc::Global {}
