use gpu_ffi::{msm_configuration, msm_execute_async};

use super::*;

const NUM_BUCKETS: usize = 254;

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
    let result = if scalars.len().is_power_of_two() {
        let (bases, _) = bases.split_at(scalars.len());
        let result = raw_msm::<E>(&scalars[..scalars.len()], &bases, stream)?;
        result.into_affine()
    } else {
        let mut intermediate_sums = vec![];
        let mut scalars_ref = &scalars[..scalars.len()];
        let mut bases_ref = &bases[..scalars.len()];
        loop {
            let num_sub_polys = scalars_ref.len() / domain_size;
            let chunk_size = previous_power_of_two(num_sub_polys) * domain_size;
            let (input_scalars, remaining_scalars) = scalars_ref.split_at(chunk_size);
            let (input_bases, remaining_bases) = bases_ref.split_at(chunk_size);
            let intermediate_sum = raw_msm::<E>(input_scalars, input_bases, stream)?;
            intermediate_sums.push(intermediate_sum);
            if remaining_scalars.is_empty() {
                break;
            }
            if remaining_scalars.len() <= domain_size {
                let mut buf = DVec::allocate_zeroed_on(domain_size, stream);
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

        let mut final_sum = intermediate_sums.pop().unwrap();
        for point in intermediate_sums.iter() {
            final_sum.add_assign(point);
        }

        final_sum.into_affine()
    };

    Ok(result)
}

fn raw_msm<E: Engine>(
    scalars: &DSlice<E::Fr>,
    bases: &DSlice<CompactG1Affine>,
    stream: bc_stream,
) -> CudaResult<E::G1> {
    assert!(scalars.len().is_power_of_two());
    assert_eq!(scalars.len(), bases.len());
    assert_eq!(64, std::mem::size_of_val(&bases[0]));

    let log_scalars_count = scalars.len().trailing_zeros();

    let bases_ptr = bases.as_ptr() as *mut CompactG1Affine;
    let scalars_ptr = scalars.as_ptr() as *mut E::Fr;
    let mut result: DVec<E::G1> = DVec::allocate_on(NUM_BUCKETS, stream);
    let result_ptr = result.as_mut_ptr() as *mut E::G1;

    let cfg = msm_configuration {
        mem_pool: _mem_pool(),
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
    println!("Scheduling MSM kernel on device");
    unsafe {
        if msm_execute_async(cfg) != 0 {
            return Err(CudaError::MsmError("Scheduling Error".to_string()));
        };
    }
    stream.sync().unwrap();
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
