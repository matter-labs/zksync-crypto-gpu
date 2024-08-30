use super::*;
use std::os::raw::c_void;

pub(crate) fn allocate(num_bytes: usize) -> CudaResult<*mut c_void> {
    let mut ptr = std::ptr::null_mut();
    unsafe {
        let result = gpu_ffi::bc_malloc(std::ptr::addr_of_mut!(ptr), num_bytes as u64);
        if result != 0 {
            panic!("Couln't statically allocate buffer");
        }
    }

    Ok(ptr)
}

pub(crate) fn dealloc(ptr: *mut c_void) -> CudaResult<()> {
    unsafe {
        let result = gpu_ffi::bc_free(ptr);
        if result != 0 {
            panic!("Couln't free static buffer");
        }
    }

    Ok(())
}

pub(crate) fn allocate_async(num_bytes: usize, stream: bc_stream) -> CudaResult<*mut c_void> {
    let mut ptr = std::ptr::null_mut();
    unsafe {
        let result = gpu_ffi::bc_malloc_from_pool_async(
            std::ptr::addr_of_mut!(ptr),
            num_bytes as u64,
            _mem_pool(),
            stream,
        );
        if result != 0 {
            return Err(CudaError::AllocationError(result.to_string()));
        }
    }

    Ok(ptr)
}

pub(crate) fn dealloc_async(ptr: *mut c_void, stream: bc_stream) -> CudaResult<()> {
    unsafe {
        let result = gpu_ffi::bc_free_async(ptr, stream);
        if result != 0 {
            return Err(CudaError::AllocationError(result.to_string()));
        }

        Ok(())
    }
}

pub(crate) fn memcopy_async<'a, 'b, T>(
    dst: &mut DSlice<T>,
    src: &'b DSlice<T>,
    stream: bc_stream,
) -> CudaResult<()> {
    assert_eq!(dst.is_empty(), false);
    assert_eq!(dst.len(), src.len());
    let num_bytes = src.len() * std::mem::size_of::<T>();
    let src_ptr = src.as_ptr() as *mut c_void;
    let dst_ptr = dst.as_mut_ptr() as *mut c_void;
    memcopy_async_inner(dst_ptr, src_ptr, num_bytes, stream)?;

    Ok(())
}

pub(crate) fn memcopy_from_host_async<'a, 'b, T>(
    dst: &mut DSlice<T>,
    src: &'b [T],
    stream: bc_stream,
) -> CudaResult<()> {
    assert_eq!(dst.is_empty(), false);
    assert_eq!(dst.len(), src.len());
    let num_bytes = src.len() * std::mem::size_of::<T>();
    memcopy_async_inner(dst.as_mut_ptr(), src.as_ptr(), num_bytes, stream)?;

    Ok(())
}

pub(crate) fn memcopy_to_host_async<'a, 'b, T>(
    dst: &'a mut [T],
    src: &'b DSlice<T>,
    stream: bc_stream,
) -> CudaResult<()> {
    assert_eq!(dst.is_empty(), false);
    assert_eq!(dst.len(), src.len());
    let num_bytes = src.len() * std::mem::size_of::<T>();
    memcopy_async_inner(dst.as_mut_ptr(), src.as_ptr(), num_bytes, stream)?;
    Ok(())
}

pub(crate) fn memcopy_async_inner<T>(
    dst_ptr: *mut T,
    src_ptr: *const T,
    num_bytes: usize,
    stream: bc_stream,
) -> CudaResult<()> {
    unsafe {
        let result =
            gpu_ffi::bc_memcpy_async(dst_ptr.cast(), src_ptr.cast(), num_bytes as u64, stream);
        if result != 0 {
            return Err(CudaError::TransferError(result.to_string()));
        }
    }

    Ok(())
}

pub(crate) fn memcopy_from_host<'a, 'b, T>(dst: &mut DSlice<T>, src: &'b [T]) -> CudaResult<()> {
    assert_eq!(dst.is_empty(), false);
    assert_eq!(dst.len(), src.len());
    let num_bytes = src.len() * std::mem::size_of::<T>();
    memcopy_inner(dst.as_mut_ptr(), src.as_ptr(), num_bytes)?;

    Ok(())
}

pub(crate) fn memcopy_to_host<'a, 'b, T>(dst: &'a mut [T], src: &'b DSlice<T>) -> CudaResult<()> {
    assert_eq!(dst.is_empty(), false);
    assert_eq!(dst.len(), src.len());
    let num_bytes = src.len() * std::mem::size_of::<T>();
    memcopy_inner(dst.as_mut_ptr(), src.as_ptr(), num_bytes)?;

    Ok(())
}

pub(crate) fn memcopy_inner<T>(
    dst_ptr: *mut T,
    src_ptr: *const T,
    num_bytes: usize,
) -> CudaResult<()> {
    unsafe {
        let result = gpu_ffi::bc_memcpy(dst_ptr.cast(), src_ptr.cast(), num_bytes as u64);
        if result != 0 {
            return Err(CudaError::TransferError(result.to_string()));
        }
    }

    Ok(())
}

pub(crate) fn h2d<'a, 'b, T>(host: &'a [T], device: &'b mut DSlice<T>) -> CudaResult<()> {
    h2d_on_stream(host, device, _h2d_stream())
}

pub fn h2d_on_stream<'a, 'b, T>(
    host: &'a [T],
    device: &'b mut DSlice<T>,
    stream: bc_stream,
) -> CudaResult<()> {
    memcopy_from_host_async(device, host, stream)
}

pub(crate) fn d2h<'a, 'b, T>(device: &'a DSlice<T>, host: &'b mut [T]) -> CudaResult<()> {
    memcopy_to_host_async(host, device, _d2h_stream())
}

pub(crate) fn d2h_on_stream<'a, 'b, T>(
    device: &'a DSlice<T>,
    host: &'b mut [T],
    stream: bc_stream,
) -> CudaResult<()> {
    memcopy_to_host_async(host, device, stream)
}

pub(crate) fn d2d<'a, 'b, T>(src: &'a DSlice<T>, dst: &'b mut DSlice<T>) -> CudaResult<()> {
    d2d_on_stream(src, dst, _d2d_stream())
}

pub(crate) fn d2d_on_stream<'a, 'b, T>(
    src: &'a DSlice<T>,
    dst: &'b mut DSlice<T>,
    stream: bc_stream,
) -> CudaResult<()> {
    memcopy_async(dst, src, stream)
}
pub(crate) fn d2d_on_stream_<T, S, D>(src: &S, dst: &mut D, stream: bc_stream) -> CudaResult<()>
where
    S: AsRef<DSlice<T>>,
    D: AsMut<DSlice<T>>,
{
    memcopy_async(dst.as_mut(), src.as_ref(), stream)
}

pub(crate) fn set_one<F>(buf: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let len = buf.len();
    unsafe {
        let result = gpu_ffi::ff_set_value_one(buf.as_mut_ptr().cast(), len as u32, stream);
        if result != 0 {
            return Err(CudaError::Error("Couldn't set buffer to 1".to_string()));
        }

        Ok(())
    }
}

pub(crate) fn set_value<F>(
    buf: &mut DSlice<F>,
    value: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let len = buf.len();
    unsafe {
        let result = gpu_ffi::ff_set_value(
            buf.as_mut_ptr().cast(),
            value.as_ptr().cast(),
            len as u32,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(
                "Couldn't set buffer to value(?)".to_string(),
            ));
        }

        Ok(())
    }
}

pub(crate) fn set_zero<T>(buf: &mut DSlice<T>, stream: bc_stream) -> CudaResult<()> {
    let len = buf.len();
    unsafe {
        let result = gpu_ffi::ff_set_value_zero(buf.as_mut_ptr().cast(), len as u32, stream);
        if result != 0 {
            return Err(CudaError::Error("Couldn't zeroing buffer".to_string()));
        }

        Ok(())
    }
}

pub(crate) fn sync_all() {
    println!("Synchronizing all streams");
    _h2d_stream().sync().unwrap();
    _d2d_stream().sync().unwrap();
    _d2h_stream().sync().unwrap();
}
