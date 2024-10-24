use super::*;

pub fn intersperse_coeffs<F: PrimeField>(
    coeffs: &DSlice<F>,
    result: &mut DSlice<F>,
    poly_idx: usize,
    num_polys: usize,
    stream: bc_stream,
) -> CudaResult<()> {
    unsafe {
        let src_ptr = coeffs.as_ptr();
        let dst_ptr = result.as_mut_ptr();
        let length = coeffs.len();
        // we need to calculate proper ptr based on the poly idx
        let dst_ptr_offset = dst_ptr.add(poly_idx);
        let result = gpu_ffi::pn_distribute_values(
            src_ptr.cast(),
            dst_ptr_offset.cast(),
            length as u32,
            num_polys as u32,
            stream,
        );

        if result != 0 {
            return Err(CudaError::Error(format!(
                "Error: Interspersing coeffs {result}"
            )));
        }
    }
    Ok(())
}

pub fn materialize_domain_elems<F>(
    buf: &mut DSlice<F>,
    domain_size: usize,
    bitreversed_output: bool,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let ptr = buf.as_mut_ptr();
    let log_degree = domain_size.trailing_zeros();
    unsafe {
        let result = gpu_ffi::ff_get_powers_of_w(
            ptr.cast(),
            log_degree,
            0,
            domain_size as u32,
            false,
            bitreversed_output,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("Materialize omegas {result}")));
        }
    }

    Ok(())
}

pub fn set_default_device() -> CudaResult<()> {
    unsafe {
        let result = gpu_ffi::bc_set_device(0);
        if result != 0 {
            return Err(CudaError::Error(format!("Set device error: {result}")));
        }
    }

    Ok(())
}
