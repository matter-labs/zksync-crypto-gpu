use super::*;

#[derive(derivative::Derivative)]
#[derivative(Debug)]
pub struct DVec<T, A: DeviceAllocator = GlobalDevice> {
    #[derivative(Debug = "ignore")]
    buf: RawDVec<T, A>,
}

impl<T> DVec<T> {
    pub fn dangling() -> Self {
        Self {
            buf: RawDVec::dangling(),
        }
    }
    pub unsafe fn into_owned_chunks(self, chunk_size: usize) -> Vec<Self> {
        assert_eq!(self.len() % chunk_size, 0);
        let num_chunks = self.len() / chunk_size;
        let mut result = vec![];
        let (ptr, len, stream) = self.into_raw_parts();
        assert_eq!(len, chunk_size * num_chunks);
        for chunk_idx in 0..num_chunks {
            let ptr = ptr.add(chunk_idx * chunk_size);
            let new = Self::from_raw_parts(ptr, chunk_size, stream);
            result.push(new);
        }
        result
    }

    pub fn allocate_on(length: usize, stream: bc_stream) -> Self {
        Self {
            buf: RawDVec::allocate_on(length, GlobalDevice, Some(stream)),
        }
    }

    pub fn allocate_zeroed_on(length: usize, stream: bc_stream) -> Self {
        Self {
            buf: RawDVec::allocate_zeroed_on(length, GlobalDevice, Some(stream)),
        }
    }

    pub fn allocate_zeroed(length: usize) -> Self {
        Self {
            buf: RawDVec::allocate_zeroed_on(length, GlobalDevice, None),
        }
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn from_host_slice_on(src: &[T], stream: bc_stream) -> CudaResult<Self> {
        let mut dst = Self::allocate_on(src.len(), stream);
        mem::h2d_on(src, &mut dst, stream)?;

        Ok(dst)
    }

    pub fn as_ptr(&self) -> *const T {
        self.buf.ptr().cast()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.ptr()
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

    pub fn clone_on(&self, stream: bc_stream) -> CudaResult<Self> {
        let mut new = Self::allocate_on(self.len(), stream);
        mem::d2d_on(self, &mut new, stream)?;

        Ok(new)
    }

    pub fn to_vec(self) -> CudaResult<Vec<T>> {
        self.to_vec_on(_d2h_stream())
    }

    pub fn to_vec_on(&self, stream: bc_stream) -> CudaResult<Vec<T>> {
        self.as_ref().to_vec_on(stream)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub unsafe fn into_raw_parts(self) -> (*mut T, usize, Option<bc_stream>) {
        let mut me = std::mem::ManuallyDrop::new(self);
        let len = me.buf.len();
        let ptr = me.as_mut_ptr();
        (ptr, len, me.buf.stream)
    }

    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, stream: Option<bc_stream>) -> Self {
        unsafe {
            DVec {
                buf: RawDVec::<_, GlobalDevice>::from_raw_parts_in(ptr, len, GlobalDevice, stream),
            }
        }
    }
}

impl<T> std::ops::Deref for DVec<T> {
    type Target = DSlice<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { DSlice::from_raw_parts(self.as_ptr(), self.buf.len()) }
    }
}

impl<T> std::ops::DerefMut for DVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { DSlice::from_raw_parts_mut(self.as_mut_ptr(), self.buf.len()) }
    }
}

impl<T> AsRef<DSlice<T>> for DVec<T> {
    fn as_ref(&self) -> &DSlice<T> {
        self
    }
}

impl<T> AsMut<DSlice<T>> for DVec<T> {
    fn as_mut(&mut self) -> &mut DSlice<T> {
        self
    }
}

impl<T> std::ops::Index<usize> for DVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index)
    }
}

impl<T> std::ops::IndexMut<usize> for DVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
    }
}

impl<T> std::ops::Index<std::ops::Range<usize>> for DVec<T> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<T> std::ops::IndexMut<std::ops::Range<usize>> for DVec<T> {
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

impl<T> std::ops::Index<std::ops::RangeTo<usize>> for DVec<T> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<T> std::ops::IndexMut<std::ops::RangeTo<usize>> for DVec<T> {
    fn index_mut(&mut self, index: std::ops::RangeTo<usize>) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

impl<T> std::ops::Index<std::ops::RangeFrom<usize>> for DVec<T> {
    type Output = DSlice<T>;

    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        &self.as_ref()[index]
    }
}

impl<T> std::ops::IndexMut<std::ops::RangeFrom<usize>> for DVec<T> {
    fn index_mut(&mut self, index: std::ops::RangeFrom<usize>) -> &mut Self::Output {
        &mut self.as_mut()[index]
    }
}

pub trait DropOn {
    fn drop_on(&mut self, stream: bc_stream);
}

pub fn drop_on<T: DropOn>(data: T, stream: bc_stream) {
    let mut data = std::mem::ManuallyDrop::new(data);
    data.drop_on(stream)
}

impl<F> DropOn for DVec<F>
where
    F: PrimeField,
{
    fn drop_on(&mut self, stream: bc_stream) {
        self.buf.drop_on(stream)
    }
}
