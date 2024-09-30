use gpu_ffi::{ntt_configuration, ntt_execute_async};

use super::*;

pub fn bitreverse<F>(coeffs: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    assert!(coeffs.len().is_power_of_two());

    let ptr = coeffs.as_mut_ptr();
    let log_n = coeffs.len().trailing_zeros();
    unsafe {
        let result = gpu_ffi::ff_bit_reverse(ptr.cast(), ptr.cast(), log_n, stream);
        if result != 0 {
            return Err(CudaError::BitreverseError(result.to_string()));
        }
    }

    Ok(())
}

pub fn fft_on<F>(coeffs: &DSlice<F>, values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let inverse = false;
    unsafe { outplace_ntt(coeffs, values, inverse, None, None, stream) }
}

pub fn ifft_on<F>(values: &DSlice<F>, coeffs: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let inverse = true;
    unsafe { outplace_ntt(values, coeffs, inverse, None, None, stream) }
}

pub fn coset_fft_on<F>(
    coeffs: &DSlice<F>,
    evals: &mut DSlice<F>,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert!(coset_idx < lde_factor);
    unsafe {
        outplace_ntt(
            coeffs,
            evals,
            false,
            Some(coset_idx),
            Some(lde_factor),
            stream,
        )
    }
}

pub fn inplace_coset_fft_for_gen_on<F>(
    coeffs: &mut DSlice<F>,
    coset_gen: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let inverse = false;
    mul_assign_by_powers(coeffs, coset_gen, stream)?;
    unsafe { inplace_ntt(coeffs, inverse, None, None, stream) }
}

pub fn inplace_coset_ifft_for_gen_on<F>(
    values: &mut DSlice<F>,
    coset_gen_inv: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let inverse = true;
    unsafe { inplace_ntt(values, inverse, None, None, stream)? }
    mul_assign_by_powers(values, coset_gen_inv, stream)
}

pub fn inplace_coset_fft_on<F>(
    coeffs: &mut DSlice<F>,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert!(coset_idx < lde_factor);

    let inverse = false;
    unsafe { inplace_ntt(coeffs, inverse, Some(coset_idx), Some(lde_factor), stream) }
}

pub fn inplace_fft_on<F>(values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let inverse = false;
    unsafe { inplace_ntt(values, inverse, None, None, stream) }
}

pub fn inplace_ifft_on<F>(values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let inverse = true;
    unsafe { inplace_ntt(values, inverse, None, None, stream) }
}

pub fn coset_ifft_on<F>(
    values: &DSlice<F>,
    coeffs: &mut DSlice<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let inverse = true;
    unsafe { outplace_ntt(values, coeffs, inverse, None, None, stream) }
}

unsafe fn inplace_ntt<F>(
    scalars: &mut DSlice<F>,
    inverse: bool,
    coset_idx: Option<usize>,
    lde_factor: Option<usize>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let ptr = scalars.as_mut_ptr();

    schedule_ntt(
        (ptr).cast(),
        ptr,
        true,
        scalars.len(),
        inverse,
        coset_idx,
        lde_factor,
        stream,
    )
}

pub(crate) unsafe fn outplace_ntt<F>(
    input: &DSlice<F>,
    output: &mut DSlice<F>,
    inverse: bool,
    coset_idx: Option<usize>,
    lde_factor: Option<usize>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(input.len(), output.len());

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_mut_ptr();

    schedule_ntt(
        input_ptr,
        output_ptr,
        false,
        input.len(),
        inverse,
        coset_idx,
        lde_factor,
        stream,
    )
}

unsafe fn schedule_ntt<F>(
    input_ptr: *const F,
    output_ptr: *mut F,
    can_overwrite_inputs: bool,
    input_len: usize,
    inverse: bool,
    coset_idx: Option<usize>,
    lde_factor: Option<usize>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert!(input_len.is_power_of_two());
    let log_values_count = input_len.trailing_zeros();
    let log_extension_degree = lde_factor.map(|v| v.trailing_zeros()).unwrap_or(0);
    let coset_idx = coset_idx
        .map(|idx| bitreverse_idx(idx, log_extension_degree as usize))
        .unwrap_or(0);

    let input_ptr = input_ptr as *mut F;
    let cfg = ntt_configuration {
        mem_pool: _tmp_mempool(),
        stream,
        inputs: input_ptr.cast(),
        outputs: output_ptr.cast(),
        log_values_count,
        bit_reversed_inputs: false,
        inverse,
        h2d_copy_finished: bc_event::null(),
        h2d_copy_finished_callback: None,
        h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
        d2h_copy_finished: bc_event::null(),
        d2h_copy_finished_callback: None,
        d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
        can_overwrite_inputs,
        coset_index: coset_idx as u32,
        log_extension_degree,
    };
    let result = ntt_execute_async(cfg);
    if result != 0 {
        return Err(CudaError::NttError(result.to_string()));
    }

    if inverse {
        let result = gpu_ffi::ff_bit_reverse(
            output_ptr.cast(),
            output_ptr.cast(),
            log_values_count,
            stream,
        );
        if result != 0 {
            return Err(CudaError::BitreverseError(result.to_string()));
        }
    }

    Ok(())
}
