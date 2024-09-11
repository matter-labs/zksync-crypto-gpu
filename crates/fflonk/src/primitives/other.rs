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
    sync_all();
    Ok(())
}
