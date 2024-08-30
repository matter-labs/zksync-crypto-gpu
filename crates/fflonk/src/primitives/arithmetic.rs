use super::*;

pub(crate) fn add_assign<F, A, B>(this: &mut A, other: &B, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
    A: AsMut<DSlice<F>>,
    B: AsRef<DSlice<F>>,
{
    use gpu_ffi::ff_x_plus_y;
    let this = this.as_mut();
    let other = other.as_ref();

    assert_eq!(this.is_empty(), false);
    assert_eq!(this.len(), other.len());

    let len = this.len() as u32;
    let this_ptr = this.as_mut_ptr();
    let other_ptr = other.as_ptr();

    unsafe {
        let result = ff_x_plus_y(
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

pub(crate) fn sub_assign<F, A, B>(this: &mut A, other: &B, stream: bc_stream) -> CudaResult<()>
where
    F: PrimeField,
    A: AsMut<DSlice<F>>,
    B: AsRef<DSlice<F>>,
{
    let this = this.as_mut();
    let other = other.as_ref();

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

pub(crate) fn mul_assign_lagrange<F>(
    this: &mut Poly<F, LagrangeBasis>,
    other: &Poly<F, LagrangeBasis>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let this = this.as_mut();
    let other = other.as_ref();
    unsafe { mul_assign(this, other, stream) }
}

pub(crate) fn mul_assign_coset<F>(
    this: &mut Poly<F, CosetEvals>,
    other: &Poly<F, CosetEvals>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let this = this.as_mut();
    let other = other.as_ref();
    unsafe { mul_assign(this, other, stream) }
}

pub(crate) unsafe fn mul_assign<F>(
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

    Ok(())
}

pub(crate) fn add_constant<F, A>(
    this: &mut A,
    value: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
    A: AsMut<DSlice<F>>,
{
    let mut other = DVec::with_capacity_zeroed_on(this.as_mut().len(), stream);
    mem::set_value(&mut other, &value, stream)?;

    add_assign(this, &other, stream)
}

pub(crate) fn sub_constant<F, A>(
    this: &mut A,
    value: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
    A: AsMut<DSlice<F>>,
{
    let mut other = DVec::with_capacity_zeroed_on(this.as_mut().len(), stream);
    mem::set_value(&mut other, &value, stream)?;

    sub_assign(this, &other, stream)
}

pub(crate) fn mul_constant<F, A>(
    this: &mut A,
    value: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
    A: AsMut<DSlice<F>>,
{
    let mut other = DVec::with_capacity_zeroed_on(this.as_mut().len(), stream);
    mem::set_value(&mut other, &value, stream)?;

    unsafe { mul_assign(this.as_mut(), &other, stream) }
}
