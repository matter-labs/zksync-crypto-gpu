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
            return Err(CudaError::Error("AddAssign error".to_string()));
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
            return Err(CudaError::Error("AddAssignScaled error".to_string()));
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
            return Err(CudaError::Error("SubAssignScaled error".to_string()));
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
            return Err(CudaError::Error("SubAssign error".to_string()));
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
            return Err(CudaError::Error("MulAsssign error".to_string()));
        }
    }

    Ok(())
}

pub(crate) fn add_constant<F>(
    this: &mut DSlice<F>,
    value: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let mut other = DVec::allocate_zeroed_on(this.len(), stream);
    mem::set_value(&mut other, &value, stream)?;

    add_assign(this, &other, stream)
}

pub(crate) fn sub_constant<F>(
    this: &mut DSlice<F>,
    value: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let mut other = DVec::allocate_zeroed_on(this.len(), stream);
    mem::set_value(&mut other, &value, stream)?;

    sub_assign(this, &other, stream)
}

pub(crate) fn mul_constant<F>(
    this: &mut DSlice<F>,
    value: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let mut other = DVec::allocate_zeroed_on(this.len(), stream);
    mem::set_value(&mut other, &value, stream)?;

    mul_assign(this, &other, stream)
}

pub fn batch_inverse<F>(values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let ptr = values.as_mut_ptr();
    let len = values.len();
    let cfg = gpu_ffi::ff_inverse_configuration {
        mem_pool: _mem_pool(),
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

    Ok(())
}

pub fn grand_product<F>(values: &mut DSlice<F>, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
{
    let ptr = values.as_mut_ptr();
    let len = values.len();
    let cfg = gpu_ffi::ff_grand_product_configuration {
        mem_pool: _mem_pool(),
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
    result: &mut DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let ptr = coeffs.as_ptr() as *mut F;
    let len = coeffs.len();
    let point_ptr = point.as_ptr() as *mut DScalar<F>;
    let result_ptr = result.as_mut_ptr() as *mut DScalar<F>;
    let cfg = gpu_ffi::ff_poly_evaluate_configuration {
        mem_pool: _mem_pool(),
        stream,
        values: ptr.cast(),
        point: point_ptr.cast(),
        result: result_ptr.cast(),
        count: len as u32,
    };
    unsafe {
        let result = gpu_ffi::ff_poly_evaluate(cfg);
        if result != 0 {
            return Err(CudaError::GrandProductErr(result.to_string()));
        }
    }

    Ok(())
}

pub fn mul_assign_by_powers<F>(
    values: &mut DSlice<F>,
    el: &DScalar<F>,
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
            mem_pool: _mem_pool(),
            inputs: input_ptr.cast(),
            base: base_ptr.cast(),
            outputs: output_ptr.cast(),
            count: size as u32,
            stream,
        };
        let result = gpu_ffi::ff_multiply_by_powers(cfg);
        if result != 0 {
            return Err(CudaError::Error(format!("Materialize omegas {result}")));
        }
    }

    Ok(())
}
