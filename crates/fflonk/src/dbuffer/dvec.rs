use super::*;

#[derive(derivative::Derivative)]
#[derivative(Debug)]
pub struct DVec<T, A: DeviceAllocator = GlobalDeviceStatic> {
    #[derivative(Debug = "ignore")]
    pub(crate) buf: RawDVec<T, A>,
}

impl<T, A: DeviceAllocator> DVec<T, A> {
    pub unsafe fn into_owned_chunks(self, chunk_size: usize) -> Vec<Self> {
        assert_eq!(self.len() % chunk_size, 0);
        let num_chunks = self.len() / chunk_size;
        let mut result = vec![];
        let (ptr, len, alloc, pool, stream) = self.into_raw_parts();
        assert_eq!(len, chunk_size * num_chunks);
        for chunk_idx in 0..num_chunks {
            let ptr = ptr.add(chunk_idx * chunk_size);
            let new = Self::from_raw_parts_in(ptr, chunk_size, A::default(), pool, stream);
            result.push(new);
        }
        result
    }

    pub unsafe fn split_into_owned_array(self, mid: usize) -> [Self; 2] {
        assert!(mid < self.len());
        let (ptr, len, _, pool, stream) = self.into_raw_parts();

        let left = Self::from_raw_parts_in(ptr, mid, A::default(), pool, stream);
        let right = Self::from_raw_parts_in(ptr.add(mid), len - mid, A::default(), pool, stream);

        [left, right]
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn as_ptr(&self) -> *const T {
        self.buf.as_ptr().cast()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.as_ptr()
    }

    pub fn get(&self, index: usize) -> &T {
        self.as_ref().get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> &mut T {
        self.as_mut().get_mut(index)
    }

    pub fn split_at(&self, mid: usize) -> (&DSlice<T>, &DSlice<T>) {
        self.as_ref().split_at(mid)
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut DSlice<T>, &mut DSlice<T>) {
        self.as_mut().split_at_mut(mid)
    }

    pub fn iter(&self) -> DIter<T> {
        DIter::new(self)
    }

    pub fn iter_mut(&mut self) -> DIterMut<T> {
        DIterMut::new(self)
    }

    pub fn chunks(&self, chunk_size: usize) -> DChunks<T> {
        DChunks::new(self, chunk_size)
    }

    pub fn chunks_mut(&mut self, chunk_size: usize) -> DChunksMut<T> {
        DChunksMut::new(self, chunk_size)
    }

    pub fn to_vec_on(&self, stream: bc_stream) -> CudaResult<Vec<T>> {
        self.as_ref().to_vec_on(stream)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub unsafe fn into_raw_parts(
        self,
    ) -> (*mut T, usize, A, Option<bc_mem_pool>, Option<bc_stream>) {
        let mut me = std::mem::ManuallyDrop::new(self);
        let len = me.buf.len();
        let ptr = me.as_mut_ptr();
        (ptr, len, A::default(), me.buf.pool, me.buf.stream)
    }

    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        len: usize,
        alloc: A,
        pool: Option<bc_mem_pool>,
        stream: Option<bc_stream>,
    ) -> Self {
        unsafe {
            DVec {
                buf: RawDVec::from_raw_parts_in(ptr, len, alloc, pool, stream),
            }
        }
    }
}

impl<T> DVec<T, PoolAllocator> {
    pub fn allocate_on(length: usize, pool: bc_mem_pool, stream: bc_stream) -> Self {
        Self {
            buf: RawDVec::allocate_on(length, PoolAllocator, pool, stream),
        }
    }

    pub fn allocate_zeroed_on(length: usize, pool: bc_mem_pool, stream: bc_stream) -> Self {
        Self {
            buf: RawDVec::allocate_zeroed_on(length, PoolAllocator, pool, stream),
        }
    }
}

impl<T> DVec<T, GlobalDeviceStatic> {
    pub fn dangling() -> Self {
        Self {
            buf: RawDVec::dangling(),
        }
    }

    pub fn allocate(length: usize) -> Self {
        Self {
            buf: RawDVec::allocate(length, _static_alloc()),
        }
    }

    pub fn allocate_zeroed(length: usize) -> Self {
        Self {
            buf: RawDVec::allocate_zeroed(length, _static_alloc()),
        }
    }

    pub fn from_host_slice(src: &[T], stream: bc_stream) -> CudaResult<Self> {
        let mut dst = Self::allocate(src.len());
        mem::h2d_on(src, &mut dst, stream)?;

        Ok(dst)
    }
}

impl<T> CloneStatic for DVec<T, GlobalDeviceStatic> {
    fn clone(&self, stream: bc_stream) -> CudaResult<Self> {
        let mut new = Self::allocate(self.len());
        mem::d2d_on(self, &mut new, stream)?;

        Ok(new)
    }
}

impl<T> CloneOnPool for DVec<T, PoolAllocator> {
    fn clone_on(&self, pool: bc_mem_pool, stream: bc_stream) -> CudaResult<Self> {
        let mut new = Self::allocate_on(self.len(), pool, stream);
        mem::d2d_on(self, &mut new, stream)?;

        Ok(new)
    }
}

impl<T, A: DeviceAllocator> std::ops::Deref for DVec<T, A> {
    type Target = DSlice<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { DSlice::from_raw_parts(self.buf.as_ptr(), self.buf.len()) }
    }
}

impl<T, A: DeviceAllocator> std::ops::DerefMut for DVec<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { DSlice::from_raw_parts_mut(self.buf.as_mut_ptr(), self.buf.len()) }
    }
}

impl<T, A: DeviceAllocator> AsRef<DSlice<T>> for DVec<T, A> {
    fn as_ref(&self) -> &DSlice<T> {
        self
    }
}

impl<T, A: DeviceAllocator> AsMut<DSlice<T>> for DVec<T, A> {
    fn as_mut(&mut self) -> &mut DSlice<T> {
        self
    }
}

impl<T, A: DeviceAllocator> std::ops::Index<usize> for DVec<T, A> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}

impl<T, A: DeviceAllocator> std::ops::IndexMut<usize> for DVec<T, A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

impl<T, A: DeviceAllocator> std::ops::Index<std::ops::Range<usize>> for DVec<T, A> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<T, A: DeviceAllocator> std::ops::IndexMut<std::ops::Range<usize>> for DVec<T, A> {
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

impl<T, A: DeviceAllocator> std::ops::Index<std::ops::RangeTo<usize>> for DVec<T, A> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<T, A: DeviceAllocator> std::ops::IndexMut<std::ops::RangeTo<usize>> for DVec<T, A> {
    fn index_mut(&mut self, index: std::ops::RangeTo<usize>) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

impl<T, A: DeviceAllocator> std::ops::Index<std::ops::RangeFrom<usize>> for DVec<T, A> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<T, A: DeviceAllocator> std::ops::IndexMut<std::ops::RangeFrom<usize>> for DVec<T, A> {
    fn index_mut(&mut self, index: std::ops::RangeFrom<usize>) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}
