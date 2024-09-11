use super::*;

#[derive(derivative::Derivative)]
#[derivative(Debug)]
pub struct DVec<T, A: DeviceAllocator = GlobalDevice> {
    #[derivative(Debug = "ignore")]
    buf: RawDDVec<T, A>,
    len: usize,
}

impl<T> DVec<T> {
    pub unsafe fn into_owned_chunks(self, chunk_size: usize) -> Vec<Self> {
        assert_eq!(self.len() % chunk_size, 0);
        let num_chunks = self.len() / chunk_size;
        let mut result = vec![];
        let (ptr, len, cap, stream) = self.into_raw_parts();
        assert_eq!(len, chunk_size * num_chunks);
        for chunk_idx in 0..num_chunks {
            let ptr = ptr.add(chunk_idx * chunk_size);
            let new = Self::from_raw_parts(ptr, chunk_size, chunk_size, stream);
            result.push(new);
        }
        result
    }

    pub fn with_capacity_on(capacity: usize, stream: bc_stream) -> Self {
        Self {
            buf: RawDDVec::with_capacity_in(capacity, GlobalDevice, Some(stream)),
            len: capacity,
        }
    }

    pub fn with_capacity_zeroed_on(capacity: usize, stream: bc_stream) -> Self {
        Self {
            buf: RawDDVec::with_capacity_zeroed_in(capacity, GlobalDevice, Some(stream)),
            len: capacity,
        }
    }

    pub fn with_capacity_zeroed(capacity: usize) -> Self {
        Self {
            buf: RawDDVec::with_capacity_zeroed_in(capacity, GlobalDevice, None),
            len: capacity,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn from_host_slice(src: &[T]) -> CudaResult<Self> {
        Self::from_host_slice_on(src, _h2d_stream())
    }

    pub fn from_host_slice_on(src: &[T], stream: bc_stream) -> CudaResult<Self> {
        let mut dst = Self::with_capacity_on(src.len(), _h2d_stream());
        mem::h2d_on_stream(src, &mut dst, stream)?;

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

    pub fn to_vec(self) -> CudaResult<Vec<T>> {
        self.to_vec_on(_d2h_stream())
    }

    pub fn to_vec_on(self, stream: bc_stream) -> CudaResult<Vec<T>> {
        let mut dst: Vec<_> = Vec::with_capacity(self.len());
        unsafe { dst.set_len(self.len()) };
        mem::d2h_on_stream(&self, &mut dst, stream)?;

        Ok(dst)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub unsafe fn into_raw_parts(self) -> (*mut T, usize, usize, Option<bc_stream>) {
        let mut me = std::mem::ManuallyDrop::new(self);
        let len = me.len;
        let capacity = me.len;
        let ptr = me.as_mut_ptr();
        (ptr, len, capacity, me.buf.stream)
    }

    pub unsafe fn from_raw_parts(
        ptr: *mut T,
        length: usize,
        capacity: usize,
        stream: Option<bc_stream>,
    ) -> Self {
        unsafe {
            DVec {
                buf: RawDDVec::<_, GlobalDevice>::from_raw_parts_in(
                    ptr,
                    capacity,
                    GlobalDevice,
                    stream,
                ),
                len: length,
            }
        }
    }
}

impl<F> DVec<DScalar<F>>
where
    F: PrimeField,
{
    pub fn from_host_scalars_on(scalars: &[F], stream: bc_stream) -> CudaResult<Self> {
        let new = DVec::from_host_slice_on(scalars, stream)?;
        let d_scalars = unsafe { std::mem::transmute(new) };

        Ok(d_scalars)
    }

    pub fn to_scalars_vec(self, stream: bc_stream) -> CudaResult<Vec<F>> {
        let new = self.to_vec_on(stream)?;
        let scalars = unsafe { std::mem::transmute(new) };

        Ok(scalars)
    }
}
impl<T> std::ops::Deref for DVec<T> {
    type Target = DSlice<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { DSlice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

impl<T> std::ops::DerefMut for DVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { DSlice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
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
