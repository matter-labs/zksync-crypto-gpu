use super::*;

pub(crate) fn add_assign<F>(
    this: &mut DSlice<F>,
    other: &DSlice<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);
    assert_eq!(this.len(), other.len());

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let other_ptr = other.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_x_plus_y(
            this_ptr.cast(),
            other_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("AddAssign error {}", result)));
        }
    }

    Ok(())
}
pub(crate) fn add_assign_scaled<F>(
    this: &mut DSlice<F>,
    other: &DSlice<F>,
    scalar: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);
    assert_eq!(this.len(), other.len());

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let other_ptr = other.as_ptr();
    let scalar_ptr = scalar.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_ax_plus_y(
            scalar_ptr.cast(),
            other_ptr.cast(),
            this_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!(
                "AddAssignScaled error {}",
                result
            )));
        }
    }

    Ok(())
}

pub(crate) fn sub_assign_scaled<F>(
    this: &mut DSlice<F>,
    other: &DSlice<F>,
    scalar: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);
    assert_eq!(this.len(), other.len());

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let other_ptr = other.as_ptr();
    let scalar_ptr = scalar.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_ax_minus_y(
            scalar_ptr.cast(),
            other_ptr.cast(),
            this_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!(
                "SubAssignScaled error {}",
                result
            )));
        }
    }

    Ok(())
}

pub(crate) fn sub_assign<F>(
    this: &mut DSlice<F>,
    other: &DSlice<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);
    assert_eq!(this.len(), other.len());

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let other_ptr = other.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_x_minus_y(
            this_ptr.cast(),
            other_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("SubAssign error {}", result)));
        }
    }

    Ok(())
}

pub fn mul_assign<F>(this: &mut DSlice<F>, other: &DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);
    assert_eq!(this.len(), other.len());

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let other_ptr = other.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_x_mul_y(
            this_ptr.cast(),
            other_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("MulAssign error {}", result)));
        }
    }

    Ok(())
}

pub(crate) fn add_constant<F>(
    this: &mut DSlice<F>,
    scalar: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let scalar_ptr = scalar.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_a_plus_x(
            scalar_ptr.cast(),
            this_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("AddConstant error {}", result)));
        }
    }

    Ok(())
}

// TODO: bellman-cuda has no interface for it
pub(crate) fn sub_constant<F>(
    this: &mut DSlice<F>,
    scalar: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();

    let mut h_negated = scalar.to_host_value_on(stream)?;
    h_negated.negate();
    let negated = DScalar::from_host_value_on(&h_negated, stream)?;
    let negated_ptr = negated.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_a_plus_x(
            negated_ptr.cast(),
            this_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("AddConstant error {}", result)));
        }
    }

    Ok(())
}

pub(crate) fn mul_constant<F>(
    this: &mut DSlice<F>,
    scalar: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(this.is_empty(), false);

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let scalar_ptr = scalar.as_ptr();

    unsafe {
        let result = gpu_ffi::ff_ax(
            scalar_ptr.cast(),
            this_ptr.cast(),
            this_ptr.cast(),
            len,
            stream,
        );
        if result != 0 {
            return Err(CudaError::Error(format!("MulConstantErr: {}", result)));
        }
    }

    Ok(())
}

pub fn batch_inverse<F>(values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    for chunk in values.chunks_mut(BATCH_INV_CHUNK_SIZE) {
        let ptr = chunk.as_mut_ptr();
        let len = chunk.len();
        let cfg = gpu_ffi::ff_inverse_configuration {
            mem_pool: _tmp_mempool(),
            stream,
            inputs: ptr.cast(),
            outputs: ptr.cast(),
            count: len as u32,
        };
        unsafe {
            let result = gpu_ffi::ff_inverse(cfg);
            if result != 0 {
                return Err(CudaError::BatchInverseError(result.to_string()));
            }
        }
    }

    Ok(())
}

pub fn grand_product<F>(values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let ptr = values.as_mut_ptr();
    let len = values.len();
    let cfg = gpu_ffi::ff_grand_product_configuration {
        mem_pool: _tmp_mempool(),
        stream,
        inputs: ptr.cast(),
        outputs: ptr.cast(),
        count: len as u32,
    };
    unsafe {
        let result = gpu_ffi::ff_grand_product(cfg);
        if result != 0 {
            return Err(CudaError::GrandProductErr(result.to_string()));
        }
    }

    Ok(())
}

pub fn evaluate_at_into<F>(
    coeffs: &DSlice<F>,
    point: &DScalar<F>,
    final_result: &mut DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let mut intermediate_results = vec![];
    for chunk in coeffs.chunks(POLY_EVAL_CHUNK_SIZE) {
        let ptr = chunk.as_ptr() as *mut F;
        let len = chunk.len();
        let point_ptr = point.as_ptr() as *mut F;
        let mut result: DScalar<F> = DScalar::zero(stream).unwrap();
        let result_ptr = result.as_mut_ptr();
        let cfg = gpu_ffi::ff_poly_evaluate_configuration {
            mem_pool: _tmp_mempool(),
            stream,
            values: ptr.cast(),
            point: point_ptr.cast(),
            result: result_ptr.cast(),
            count: len as u32,
        };
        unsafe {
            let result = gpu_ffi::ff_poly_evaluate(cfg);
            if result != 0 {
                return Err(CudaError::PolyEvaluationErr(result.to_string()));
            }
        }
        let h_result = result.to_host_value_on(stream).unwrap();
        intermediate_results.push(h_result);
    }
    let mut h_final_result = F::zero();
    for intermediate_result in intermediate_results.iter() {
        h_final_result.add_assign(&intermediate_result);
    }
    final_result.copy_from_host_value_on(&h_final_result, stream)?;

    Ok(())
}

pub fn mul_assign_by_powers<F>(
    values: &mut DSlice<F>,
    el: &DScalar<F>,
    pool: bc_mem_pool,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let input_ptr = values.as_mut_ptr();
    let base_ptr = el.as_ptr() as *mut F;
    let output_ptr = values.as_mut_ptr();
    let size = values.len();

    unsafe {
        let cfg = gpu_ffi::ff_multiply_by_powers_configuration {
            mem_pool: pool,
            inputs: input_ptr.cast(),
            base: base_ptr.cast(),
            outputs: output_ptr.cast(),
            count: size as u32,
            stream,
        };
        let result = gpu_ffi::ff_multiply_by_powers(cfg);
        if result != 0 {
            return Err(CudaError::Error(format!(
                "Materialize omegas err: {result}"
            )));
        }
    }

    Ok(())
}
