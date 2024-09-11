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
        todo!()
    }
}

pub struct DIterMut<'a, T> {
    buf: &'a mut DSlice<T>,
}

impl<'a, T> DIterMut<'a, T> {
    pub fn new(slice: &'a mut DSlice<T>) -> Self {
        Self { buf: slice }
    }
}

impl<'a, T> Iterator for DIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
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
        todo!()
    }
}

pub struct DChunksMut<'a, T> {
    buf: &'a mut DSlice<T>,
    chunk_size: usize,
}

impl<'a, T> DChunksMut<'a, T> {
    pub fn new(slice: &'a mut DSlice<T>, chunk_size: usize) -> Self {
        Self {
            buf: slice,
            chunk_size,
        }
    }
}

impl<'a, T> Iterator for DChunksMut<'a, T> {
    type Item = &'a mut DSlice<T>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
