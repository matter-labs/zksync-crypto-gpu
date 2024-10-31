use super::*;
pub trait PolyRepr: 'static {}

pub struct MonomialBasis;
impl PolyRepr for MonomialBasis {}
pub struct CosetEvals;
impl PolyRepr for CosetEvals {}
pub struct LagrangeBasis;
impl PolyRepr for LagrangeBasis {}

pub struct LDE;
impl PolyRepr for LDE {}

#[derive(Debug)]
pub struct Poly<F: PrimeField, R: PolyRepr, A: DeviceAllocator = GlobalDeviceStatic> {
    pub(crate) storage: DVec<F, A>,
    _p: std::marker::PhantomData<R>,
}

impl<F, R, A: DeviceAllocator> AsRef<DSlice<F>> for Poly<F, R, A>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn as_ref(&self) -> &DSlice<F> {
        self.storage.as_ref()
    }
}
impl<F, R, A: DeviceAllocator> AsMut<DSlice<F>> for Poly<F, R, A>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn as_mut(&mut self) -> &mut DSlice<F> {
        self.storage.as_mut()
    }
}

impl<F, R> Poly<F, R, PoolAllocator>
where
    F: PrimeField,
    R: PolyRepr,
{
    pub fn zero_on(domain_size: usize, pool: bc_mem_pool, stream: bc_stream) -> Self {
        Self::from_buffer(DVec::allocate_zeroed_on(domain_size, pool, stream))
    }

    pub fn allocate_on(domain_size: usize, pool: bc_mem_pool, stream: bc_stream) -> Self {
        let storage = DVec::allocate_on(domain_size, pool, stream);
        Self {
            storage,
            _p: std::marker::PhantomData,
        }
    }
}

impl<F, R> Poly<F, R, GlobalDeviceStatic>
where
    F: PrimeField,
    R: PolyRepr,
{
    pub fn zero(domain_size: usize) -> Self {
        Self::from_buffer(DVec::allocate_zeroed(domain_size))
    }

    pub fn allocate(domain_size: usize) -> Self {
        let storage = DVec::allocate(domain_size);
        Self {
            storage,
            _p: std::marker::PhantomData,
        }
    }
}

impl<F: PrimeField, R: PolyRepr, A: DeviceAllocator> Poly<F, R, A> {
    pub fn from_buffer(buf: DVec<F, A>) -> Self {
        Self {
            storage: buf,
            _p: std::marker::PhantomData,
        }
    }

    pub fn into_buffer(self) -> DVec<F, A> {
        self.storage
    }

    pub fn add_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        assert!(self.size() >= other.size());
        arithmetic::add_assign(&mut self.as_mut()[..other.size()], other.as_ref(), stream)
    }

    pub fn add_assign_scaled_on(
        &mut self,
        other: &Self,
        scalar: &DScalar<F>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        assert!(self.size() >= other.size());
        arithmetic::add_assign_scaled(
            &mut self.as_mut()[..other.size()],
            other.as_ref(),
            scalar,
            stream,
        )
    }
    pub fn sub_assign_scaled_on(
        &mut self,
        other: &Self,
        scalar: &DScalar<F>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        assert!(self.size() >= other.size());
        arithmetic::sub_assign_scaled(
            &mut self.as_mut()[..other.size()],
            other.as_ref(),
            scalar,
            stream,
        )
    }

    pub fn sub_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        assert!(self.size() >= other.size());
        arithmetic::sub_assign(&mut self.as_mut()[..other.size()], other.as_ref(), stream)
    }

    pub fn scale_on(&mut self, scalar: &DScalar<F>, stream: bc_stream) -> CudaResult<()> {
        arithmetic::mul_constant(self.as_mut(), scalar, stream)
    }

    pub fn add_constant_on(&mut self, scalar: &DScalar<F>, stream: bc_stream) -> CudaResult<()> {
        arithmetic::add_constant(self.as_mut(), scalar, stream)
    }

    pub fn sub_constant_on(&mut self, scalar: &DScalar<F>, stream: bc_stream) -> CudaResult<()> {
        arithmetic::sub_constant(self.as_mut(), scalar, stream)
    }

    pub fn bitreverse(&mut self, stream: bc_stream) -> CudaResult<()> {
        ntt::bitreverse(self.as_mut(), stream)
    }

    pub fn batch_inverse(&mut self, stream: bc_stream) -> CudaResult<()> {
        arithmetic::batch_inverse(self.as_mut(), stream)
    }

    pub fn grand_product(&mut self, stream: bc_stream) -> CudaResult<()> {
        let domain_size = self.size();
        assert!(domain_size.is_power_of_two());
        // rotate by one then set first value to 1 and compute grand product
        arithmetic::grand_product(self.as_mut(), stream)?;
        // TODO
        let mut tmp = DVec::allocate(self.size());
        mem::d2d_on(self.as_ref(), &mut tmp, stream)?;
        mem::set_value(&mut self.as_mut()[0..1], &DScalar::one(stream)?, stream)?;
        mem::d2d_on(&tmp[..domain_size - 1], &mut self.as_mut()[1..], stream)?;

        Ok(())
    }

    pub fn size(&self) -> usize {
        self.storage.len()
    }
}

impl<F: PrimeField, A: DeviceAllocator> Poly<F, CosetEvals, A> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        arithmetic::mul_assign(self.as_mut(), other.as_ref(), stream)
    }

    pub fn coset_ifft_on(&self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis>> {
        let mut coeffs = Poly::<F, MonomialBasis>::zero(self.size());
        ntt::coset_ifft_on(self.as_ref(), coeffs.as_mut(), stream)?;

        Ok(coeffs)
    }
}

impl<F: PrimeField, A: DeviceAllocator> Poly<F, LagrangeBasis, A> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        arithmetic::mul_assign(self.as_mut(), other.as_ref(), stream)
    }

    pub fn ifft_on(mut self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis, A>> {
        ntt::inplace_ifft_on(self.as_mut(), stream)?;
        Ok(Poly::<F, MonomialBasis, A>::from_buffer(self.storage))
    }
}

impl<F: PrimeField, A: DeviceAllocator> Poly<F, MonomialBasis, A> {
    pub fn evaluate_at_into_on(
        &self,
        point: &DScalar<F>,
        into: &mut DScalar<F>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        arithmetic::evaluate_at_into(self.as_ref(), point, into, stream)
    }
    pub fn fft_on(mut self, stream: bc_stream) -> CudaResult<Poly<F, LagrangeBasis, A>> {
        ntt::inplace_fft_on(self.as_mut(), stream)?;
        Ok(Poly::<F, LagrangeBasis, A>::from_buffer(self.storage))
    }

    pub fn coset_fft_on(
        &self,
        coset_idx: usize,
        lde_factor: usize,
        stream: bc_stream,
    ) -> CudaResult<Poly<F, CosetEvals>> {
        let mut evals = Poly::<F, CosetEvals>::zero(self.size());
        ntt::coset_fft_on(self.as_ref(), evals.as_mut(), coset_idx, lde_factor, stream)?;

        Ok(evals)
    }

    pub fn trim_to_degree(self, new_degree: usize, stream: bc_stream) -> CudaResult<Self> {
        assert!(new_degree < self.size());
        let [new, leading_coeffs] = unsafe { self.storage.split_into_owned_array(new_degree) };

        if SANITY_CHECK {
            let leading_coeffs = leading_coeffs.to_vec_on(stream)?;
            leading_coeffs
                .iter()
                .rev()
                .enumerate()
                .for_each(|(idx, coeff)| assert!(coeff.is_zero(), "{idx}-th coeff should be zero"));
        }

        Ok(Poly::from_buffer(new))
    }
}

impl<F: PrimeField, A: DeviceAllocator> Poly<F, LDE, A> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        arithmetic::mul_assign(self.as_mut(), other.as_ref(), stream)
    }

    pub fn coset_ifft_on(mut self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis, A>> {
        let gen_inv = DScalar::inv_multiplicative_generator(stream)?;

        ntt::inplace_coset_ifft_for_gen_on(self.as_mut(), &gen_inv, stream)?;
        let monomial = self.into_buffer();

        Ok(Poly::from_buffer(monomial))
    }
}

pub trait CloneStatic: Sized {
    fn clone(&self, stream: bc_stream) -> CudaResult<Self>;
}

impl<F, R> CloneStatic for Poly<F, R, GlobalDeviceStatic>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn clone(&self, stream: bc_stream) -> CudaResult<Self> {
        Ok(Self {
            storage: self.storage.clone(stream)?,
            _p: std::marker::PhantomData,
        })
    }
}

pub trait CloneOnPool: Sized {
    fn clone_on(&self, pool: bc_mem_pool, stream: bc_stream) -> CudaResult<Self>;
}

impl<F, R> CloneOnPool for Poly<F, R, PoolAllocator>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn clone_on(&self, pool: bc_mem_pool, stream: bc_stream) -> CudaResult<Self> {
        Ok(Self {
            storage: self.storage.clone_on(pool, stream)?,
            _p: std::marker::PhantomData,
        })
    }
}
