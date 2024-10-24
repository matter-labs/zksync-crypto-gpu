use gpu_ffi::{msm_configuration, msm_execute_async};

use super::*;

const NUM_BUCKETS: usize = 254;

static mut _MSM_RESULT_MEMPOOL: Option<bc_mem_pool> = None;

pub(crate) fn init_msm_result_mempool() {
    unsafe {
        _MSM_RESULT_MEMPOOL = Some(bc_mem_pool::new(DEFAULT_DEVICE_ID).unwrap());
    }
}

pub(crate) fn _msm_result_mempool() -> bc_mem_pool {
    unsafe { _MSM_RESULT_MEMPOOL.expect("small msm mempool intialized") }
}

pub(crate) fn is_msm_result_mempool_initialized() -> bool {
    unsafe { _MSM_RESULT_MEMPOOL.is_some() }
}

pub(crate) fn destroy_msm_result_mempool() {
    unsafe {
        let pool = _MSM_RESULT_MEMPOOL.take().unwrap();
        let result = gpu_ffi::bc_mem_pool_destroy(pool);
        if result != 0 {
            panic!("Couldn't destroy msm result pool");
        }
    }
}

fn previous_power_of_two(x: usize) -> usize {
    if x == 0 {
        return 0;
    }
    1 << (31 - (x as u32).leading_zeros())
}

pub fn msm<E: Engine>(
    scalars: &DSlice<E::Fr>,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<E::G1Affine> {
    assert_eq!(domain_size.is_power_of_two(), true);
    let bases = _bases();
    assert!(scalars.len() <= bases.len());
    let mut intermediate_sums = vec![];
    let mut scalars_ref = scalars;
    let mut bases_ref = &bases[..scalars.len()];
    println!("Scheduling MSM kernels on device");
    loop {
        let chunk_size = std::cmp::min(domain_size, MSM_CHUNK_SIZE);
        let (input_scalars, remaining_scalars) = scalars_ref.split_at(chunk_size);
        let (input_bases, remaining_bases) = bases_ref.split_at(chunk_size);
        let intermediate_sum = raw_msm::<E>(input_scalars, input_bases, stream)?;
        intermediate_sums.push(intermediate_sum);
        if remaining_scalars.is_empty() {
            break;
        }
        if remaining_scalars.len() <= domain_size {
            let mut buf = DVec::allocate_zeroed(domain_size);
            mem::d2d_on(
                remaining_scalars,
                &mut buf[..remaining_scalars.len()],
                stream,
            )?;
            let intermediate_sum = raw_msm::<E>(&buf, &remaining_bases[..domain_size], stream)?;
            intermediate_sums.push(intermediate_sum);
            break;
        }
        scalars_ref = remaining_scalars;
        bases_ref = remaining_bases;
    }
    stream.sync().unwrap();
    let mut final_sum = intermediate_sums.pop().unwrap();
    for point in intermediate_sums.iter() {
        final_sum.add_assign(point);
    }

    Ok(final_sum.into_affine())
}

fn raw_msm<E: Engine>(
    scalars: &DSlice<E::Fr>,
    bases: &DSlice<CompactG1Affine>,
    stream: bc_stream,
) -> CudaResult<E::G1> {
    assert!(is_context_initialized());
    assert!(scalars.len().is_power_of_two());
    assert_eq!(scalars.len(), bases.len());
    assert_eq!(64, std::mem::size_of_val(&bases[0]));

    let log_scalars_count = scalars.len().trailing_zeros();

    let bases_ptr = bases.as_ptr() as *mut CompactG1Affine;
    let scalars_ptr = scalars.as_ptr() as *mut E::Fr;
    let mut result = DVec::allocate_on(NUM_BUCKETS, _msm_result_mempool(), stream);
    let result_ptr = result.as_mut_ptr() as *mut E::G1;

    let cfg = msm_configuration {
        mem_pool: _tmp_mempool(),
        stream: stream,
        bases: bases_ptr.cast(),
        scalars: scalars_ptr.cast(),
        results: result_ptr.cast(),
        log_scalars_count,
        h2d_copy_finished: bc_event::null(),
        h2d_copy_finished_callback: None,
        h2d_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
        d2h_copy_finished: bc_event::null(),
        d2h_copy_finished_callback: None,
        d2h_copy_finished_callback_data: std::ptr::null_mut() as *mut _,
    };
    unsafe {
        let result = msm_execute_async(cfg);

        if result != 0 {
            return Err(CudaError::MsmError(result.to_string()));
        };
    }
    let mut h_result: Vec<E::G1> = Vec::with_capacity(NUM_BUCKETS);
    unsafe { h_result.set_len(NUM_BUCKETS) };
    mem::d2h_on(&result, &mut h_result, stream).unwrap();

    // aggregate buckets with horner trick
    // b0 + 2*b1 + 2^2*b2 + .. 2^253*b253
    let mut sum = h_result.last().unwrap().clone();
    for point in h_result.iter().rev().skip(1) {
        sum.double();
        sum.add_assign(point);
    }

    Ok(sum)
}
