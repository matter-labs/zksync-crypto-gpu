use super::DSlice;

pub struct DIter<'a, T> {
    buf: &'a DSlice<T>,
}

impl<'a, T> DIter<'a, T> {
    pub fn new(slice: &'a DSlice<T>) -> Self {
        Self { buf: slice }
    }
}

impl<'a, T> Iterator for DIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() {
            return None;
        }

        let (this, rest) = self.buf.split_at(1);
        self.buf = rest;

        Some(&this[0])
    }
}

pub struct DIterMut<'a, T> {
    buf: *mut [T],
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T> DIterMut<'a, T> {
    pub fn new(slice: &'a mut DSlice<T>) -> Self {
        Self {
            buf: slice as *mut DSlice<T> as *mut [T],
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T> Iterator for DIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() {
            return None;
        }

        unsafe {
            let (this, rest) = self.buf.split_at_mut(1);
            self.buf = rest;
            let this = DSlice::from_raw_parts_mut(this as *mut T, 1);
            Some(&mut this[0])
        }
    }
}

pub struct DChunks<'a, T> {
    buf: &'a DSlice<T>,
    chunk_size: usize,
}

impl<'a, T> DChunks<'a, T> {
    pub fn new(slice: &'a DSlice<T>, chunk_size: usize) -> Self {
        Self {
            buf: slice,
            chunk_size,
        }
    }
}

impl<'a, T> Iterator for DChunks<'a, T> {
    type Item = &'a DSlice<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.is_empty() {
            return None;
        }

        let (this, rest) = self.buf.split_at(self.chunk_size);
        self.buf = rest;

        Some(this)
    }
}

pub struct DChunksMut<'a, T> {
    buf: *mut [T],
    chunk_size: usize,
    _marker: std::marker::PhantomData<&'a mut T>,
}

impl<'a, T> DChunksMut<'a, T> {
    pub fn new(slice: &'a mut DSlice<T>, chunk_size: usize) -> Self {
        Self {
            buf: slice as *mut DSlice<T> as *mut [T],
            chunk_size,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, T> Iterator for DChunksMut<'a, T> {
    type Item = &'a mut DSlice<T>;

    fn next(&mut self) -> Option<&'a mut DSlice<T>> {
        if self.buf.is_empty() {
            return None;
        }

        unsafe {
            let (this, rest) = self.buf.split_at_mut(self.chunk_size);
            self.buf = rest;
            Some(DSlice::from_raw_parts_mut(this as *mut T, self.chunk_size))
        }
    }
}
