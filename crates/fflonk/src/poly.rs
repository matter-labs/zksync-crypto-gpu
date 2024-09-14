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
pub struct Poly<F: PrimeField, P: PolyRepr> {
    storage: DVec<F>,
    _p: std::marker::PhantomData<P>,
}

impl<F, R> AsRef<DSlice<F>> for Poly<F, R>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn as_ref(&self) -> &DSlice<F> {
        self.storage.as_ref()
    }
}
impl<F, R> AsMut<DSlice<F>> for Poly<F, R>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn as_mut(&mut self) -> &mut DSlice<F> {
        self.storage.as_mut()
    }
}

impl<F: PrimeField, P: PolyRepr> Poly<F, P> {
    pub fn from_buffer(buf: DVec<F>) -> Self {
        Self {
            storage: buf,
            _p: std::marker::PhantomData,
        }
    }

    pub fn into_buffer(self) -> DVec<F> {
        self.storage
    }

    pub fn zero(domain_size: usize, stream: bc_stream) -> Self {
        Self::from_buffer(DVec::allocate_zeroed_on(domain_size, stream))
    }

    pub fn allocate_on(domain_size: usize, stream: bc_stream) -> Self {
        let storage = DVec::allocate_on(domain_size, stream);
        Self {
            storage,
            _p: std::marker::PhantomData,
        }
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
        let tmp = self.clone_on(stream)?;
        mem::set_value(&mut self.as_mut()[0..1], &DScalar::one(stream)?, stream)?;
        mem::d2d_on(
            &tmp.as_ref()[..domain_size - 1],
            &mut self.as_mut()[1..],
            stream,
        )?;

        Ok(())
    }

    pub fn size(&self) -> usize {
        self.storage.len()
    }

    pub fn clone_on(&self, stream: bc_stream) -> CudaResult<Self> {
        Ok(Self {
            storage: self.storage.clone_on(stream)?,
            _p: std::marker::PhantomData,
        })
    }
}

impl<F: PrimeField> Poly<F, CosetEvals> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        arithmetic::mul_assign(self.as_mut(), other.as_ref(), stream)
    }

    pub fn coset_ifft_on(&self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis>> {
        let mut coeffs = Poly::<F, MonomialBasis>::zero(self.size(), stream);
        ntt::coset_ifft_on(self.as_ref(), coeffs.as_mut(), stream)?;

        Ok(coeffs)
    }
}

impl<F: PrimeField> Poly<F, LagrangeBasis> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        arithmetic::mul_assign(self.as_mut(), other.as_ref(), stream)
    }

    pub fn ifft_on(mut self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis>> {
        ntt::inplace_ifft_on(self.as_mut(), stream)?;
        let coeffs = unsafe { std::mem::transmute(self) };
        Ok(coeffs)
    }
}

impl<F: PrimeField> Poly<F, MonomialBasis> {
    pub fn evaluate_at_into_on(
        &self,
        point: &DScalar<F>,
        into: &mut DScalar<F>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        arithmetic::evaluate_at_into(self.as_ref(), point, into, stream)
    }

    pub fn fft_on(&self, stream: bc_stream) -> CudaResult<Poly<F, LagrangeBasis>> {
        let mut values = Poly::<F, LagrangeBasis>::zero(self.size(), stream);
        ntt::fft_on(self.as_ref(), values.as_mut(), stream)?;

        Ok(values)
    }

    pub fn lde(&self, lde_factor: usize, stream: bc_stream) -> CudaResult<Poly<F, LDE>> {
        assert_eq!(lde_factor, 16);
        let mut lde = Poly::<F, LDE>::zero(lde_factor * self.size(), stream);
        for (coset_idx, coset_evals) in lde.storage.chunks_mut(self.size()).enumerate() {
            let inverse = false;
            unsafe {
                ntt::outplace_ntt(
                    self.as_ref(),
                    coset_evals,
                    inverse,
                    Some(coset_idx),
                    Some(lde_factor),
                    stream,
                )?;
            }
        }

        Ok(lde)
    }

    pub fn coset_fft_on(
        &self,
        coset_idx: usize,
        lde_factor: usize,
        stream: bc_stream,
    ) -> CudaResult<Poly<F, CosetEvals>> {
        let mut evals = Poly::<F, CosetEvals>::zero(self.size(), stream);
        ntt::coset_fft_on(self.as_ref(), evals.as_mut(), coset_idx, lde_factor, stream)?;

        Ok(evals)
    }

    pub fn trim_to_degree(self, new_degree: usize, stream: bc_stream) -> CudaResult<Self> {
        assert!(new_degree < self.size());
        let (actual_coeffs, leading_coeffs) = self.as_ref().split_at(new_degree);
        let new = actual_coeffs.to_dvec_on(stream)?;
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

impl<F: PrimeField> Poly<F, LDE> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        arithmetic::mul_assign(self.as_mut(), other.as_ref(), stream)
    }

    pub fn coset_ifft_on(mut self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis>> {
        let gen_inv = DScalar::inv_multiplicative_generator(stream)?;
        ntt::inplace_coset_ifft_for_gen_on(self.as_mut(), &gen_inv, stream)?;
        let monomial = self.into_buffer();

        Ok(Poly::from_buffer(monomial))
    }
}

impl<F, R> DropOn for Poly<F, R>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn drop_on(&mut self, stream: bc_stream) {
        self.storage.drop_on(stream)
    }
}
