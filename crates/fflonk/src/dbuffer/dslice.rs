use gpu_ffi::{bc_mem_pool, bc_stream};

use super::{
    CudaResult, DChunks, DChunksMut, DIter, DIterMut, DVec, GlobalDeviceStatic, HostAllocator,
};

pub struct DSlice<T>([T]);

impl<T> DSlice<T> {
    pub const fn len(&self) -> usize {
        std::ptr::metadata(self)
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub unsafe fn from_raw_parts<'a>(ptr: *const T, len: usize) -> &'a Self {
        std::mem::transmute(std::slice::from_raw_parts(ptr, len))
    }

    pub unsafe fn from_raw_parts_mut<'a>(ptr: *mut T, len: usize) -> &'a mut Self {
        std::mem::transmute(std::slice::from_raw_parts_mut(ptr, len))
    }

    pub fn as_ptr(&self) -> *const T {
        &self.0 as *const [T] as *mut T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        &mut self.0 as *mut [T] as *mut T
    }

    pub fn get(&self, index: usize) -> &T {
        unsafe { &*self.as_ptr().add(index) }
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        unsafe { &mut *self.as_mut_ptr().add(index) }
    }

    pub fn split_at(&self, mid: usize) -> (&Self, &Self) {
        let ptr = self.as_ptr();
        let len = self.len();
        unsafe {
            (
                DSlice::from_raw_parts(ptr, mid),
                DSlice::from_raw_parts(ptr.add(mid), len - mid),
            )
        }
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut Self, &mut Self) {
        let ptr = self.as_mut_ptr();
        let len = self.len();
        unsafe {
            (
                DSlice::from_raw_parts_mut(ptr, mid),
                DSlice::from_raw_parts_mut(ptr.add(mid), len - mid),
            )
        }
    }

    pub fn chunks(&self, chunk_size: usize) -> DChunks<T> {
        DChunks::new(self, chunk_size)
    }

    pub fn chunks_mut(&mut self, chunk_size: usize) -> DChunksMut<T> {
        DChunksMut::new(self, chunk_size)
    }

    pub fn iter<'a>(&'a self) -> DIter<'a, T> {
        DIter::new(self)
    }

    pub fn iter_mut<'a>(&'a mut self) -> DIterMut<'a, T> {
        DIterMut::new(self)
    }

    pub fn copy_from_slice_on(&mut self, other: &DSlice<T>, stream: bc_stream) -> CudaResult<()> {
        super::mem::d2d_on(other, self, stream)
    }

    pub fn to_vec(&self, stream: bc_stream) -> CudaResult<Vec<T>> {
        self.to_vec_in_on::<std::alloc::Global>(stream)
    }

    pub fn to_vec_in_on<A: HostAllocator>(&self, stream: bc_stream) -> CudaResult<Vec<T, A>> {
        let mut dst: Vec<_, A> = Vec::with_capacity_in(self.len(), A::default());
        unsafe { dst.set_len(self.len()) };
        super::mem::d2h_on(self, &mut dst, stream)?;

        Ok(dst)
    }

    pub fn to_dvec(&self, stream: bc_stream) -> CudaResult<DVec<T, GlobalDeviceStatic>> {
        let mut dst = DVec::allocate_zeroed(self.len());
        super::mem::d2d_on(self, &mut dst, stream)?;

        Ok(dst)
    }
}

impl<T> std::ops::Index<usize> for DSlice<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}
impl<T> std::ops::IndexMut<usize> for DSlice<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}
impl<T> std::ops::Index<std::ops::Range<usize>> for DSlice<T> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        let start = index.start;
        let len = index.len();

        unsafe {
            let ptr = self.as_ptr().add(start);
            Self::from_raw_parts(ptr, len)
        }
    }
}
impl<T> std::ops::IndexMut<std::ops::Range<usize>> for DSlice<T> {
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        let start = index.start;
        let len = index.len();

        unsafe {
            let ptr = self.as_mut_ptr().add(start);
            Self::from_raw_parts_mut(ptr, len)
        }
    }
}

impl<T> std::ops::Index<std::ops::RangeFrom<usize>> for DSlice<T> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        let end = self.len();
        &self[index.start..end]
    }
}
impl<T> std::ops::IndexMut<std::ops::RangeFrom<usize>> for DSlice<T> {
    fn index_mut(&mut self, index: std::ops::RangeFrom<usize>) -> &mut Self::Output {
        let end = self.len();
        &mut self[index.start..end]
    }
}

impl<T> std::ops::Index<std::ops::RangeTo<usize>> for DSlice<T> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        &self[0..index.end]
    }
}
impl<T> std::ops::IndexMut<std::ops::RangeTo<usize>> for DSlice<T> {
    fn index_mut(&mut self, index: std::ops::RangeTo<usize>) -> &mut Self::Output {
        &mut self[0..index.end]
    }
}

impl<T> std::ops::Index<std::ops::RangeFull> for DSlice<T> {
    type Output = DSlice<T>;

    fn index(&self, _index: std::ops::RangeFull) -> &Self::Output {
        &self[0..self.len()]
    }
}

impl<T> std::ops::IndexMut<std::ops::RangeFull> for DSlice<T> {
    fn index_mut(&mut self, _index: std::ops::RangeFull) -> &mut Self::Output {
        let end = self.len();
        &mut self[0..end]
    }
}
