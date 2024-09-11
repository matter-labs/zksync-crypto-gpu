use gpu_ffi::{ntt_configuration, ntt_execute_async};

use super::*;

pub fn fft_on<F: PrimeField>(
    coeffs: &DSlice<F>,
    values: &mut DSlice<F>,
    stream: bc_stream,
) -> CudaResult<()> {
    let inverse = false;
    unsafe { outplace_ntt(coeffs, values, inverse, false, None, None, stream) }
}

pub fn inplace_fft_on<F: PrimeField>(coeffs: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()> {
    let inverse = false;
    unsafe { inplace_ntt(coeffs, inverse, false, None, None, stream) }
}

pub fn coset_fft_on<F: PrimeField>(
    values: &DSlice<F>,
    coeffs: &mut DSlice<F>,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<()> {
    let inverse = false;
    unsafe {
        outplace_ntt(
            values,
            coeffs,
            inverse,
            false,
            Some(coset_idx),
            Some(lde_factor),
            stream,
        )
    }
}

pub fn ifft_on<F: PrimeField>(
    values: &DSlice<F>,
    coeffs: &mut DSlice<F>,
    stream: bc_stream,
) -> CudaResult<()> {
    let inverse = true;
    unsafe { outplace_ntt(values, coeffs, inverse, false, None, None, stream) }
}

pub fn inplace_ifft_on<F: PrimeField>(values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()> {
    let inverse = true;
    unsafe { inplace_ntt(values, inverse, false, None, None, stream) }
}

pub fn icoset_fft_on<F: PrimeField>(
    values: &DSlice<F>,
    coeffs: &mut DSlice<F>,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<()> {
    let inverse = true;
    unsafe {
        outplace_ntt(
            values,
            coeffs,
            inverse,
            false,
            Some(coset_idx),
            Some(lde_factor),
            stream,
        )
    }
}

unsafe fn inplace_ntt<F: PrimeField>(
    scalars: &mut DSlice<F>,
    inverse: bool,
    bits_reversed: bool,
    coset_idx: Option<usize>,
    lde_factor: Option<usize>,
    stream: bc_stream,
) -> CudaResult<()> {
    assert!(scalars.len().is_power_of_two());
    assert!(coset_idx < lde_factor);

    let mem_pool = _mem_pool();
    let input_ptr = scalars.as_ptr() as *mut F;
    let output_ptr = scalars.as_mut_ptr();
    let log_values_count = scalars.len().trailing_zeros();
    let log_extension_degree = lde_factor.unwrap_or(0).trailing_zeros();

    let cfg = ntt_configuration {
        mem_pool,
        stream,
        inputs: input_ptr.cast(),
        outputs: output_ptr.cast(),
        log_values_count,
        bit_reversed_inputs: bits_reversed,
        inverse,
        h2d_copy_finished: bc_event::null(),
        h2d_copy_finished_callback: None,
        h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
        d2h_copy_finished: bc_event::null(),
        d2h_copy_finished_callback: None,
        d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
        can_overwrite_inputs: false,
        coset_index: coset_idx.unwrap_or(0) as u32,
        log_extension_degree,
    };
    run_ntt(cfg)
}

unsafe fn outplace_ntt<F: PrimeField>(
    input: &DSlice<F>,
    output: &mut DSlice<F>,
    inverse: bool,
    bits_reversed: bool,
    coset_idx: Option<usize>,
    lde_factor: Option<usize>,
    stream: bc_stream,
) -> CudaResult<()> {
    assert!(input.len().is_power_of_two());
    assert_eq!(input.len(), output.len());
    assert!(coset_idx < lde_factor);

    let mem_pool = _mem_pool();
    let input_ptr = input.as_ptr() as *mut F;
    let output_ptr = output.as_mut_ptr();
    let log_values_count = input.len().trailing_zeros();
    let log_extension_degree = lde_factor.unwrap_or(0).trailing_zeros();

    let cfg = ntt_configuration {
        mem_pool,
        stream,
        inputs: input_ptr.cast(),
        outputs: output_ptr.cast(),
        log_values_count,
        bit_reversed_inputs: bits_reversed,
        inverse,
        h2d_copy_finished: bc_event::null(),
        h2d_copy_finished_callback: None,
        h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
        d2h_copy_finished: bc_event::null(),
        d2h_copy_finished_callback: None,
        d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
        can_overwrite_inputs: false,
        coset_index: coset_idx.unwrap_or(0) as u32,
        log_extension_degree,
    };
    run_ntt(cfg)
}

unsafe fn run_ntt(cfg: ntt_configuration) -> CudaResult<()> {
    let result = ntt_execute_async(cfg);
    if result != 0 {
        return Err(CudaError::NttError(result.to_string()));
    }

    // TODO
    sync_all();

    Ok(())
}
