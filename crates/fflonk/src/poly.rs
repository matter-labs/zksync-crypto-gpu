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
        todo!()
    }
}
impl<F, R> AsMut<DSlice<F>> for Poly<F, R>
where
    F: PrimeField,
    R: PolyRepr,
{
    fn as_mut(&mut self) -> &mut DSlice<F> {
        todo!()
    }
}

impl<F: PrimeField, P: PolyRepr> Poly<F, P> {
    pub fn from_buffer(buf: DVec<F>) -> Self {
        todo!()
    }

    pub fn zero(domain_size: usize, stream: bc_stream) -> Self {
        todo!()
    }

    pub fn with_capacity_on(domain_size: usize, stream: bc_stream) -> Self {
        todo!()
    }

    pub fn new_monomials_on(domain_size: usize, stream: bc_stream) -> Poly<F, MonomialBasis> {
        todo!()
    }

    pub fn new_coset_evals_on(domain_size: usize, stream: bc_stream) -> Poly<F, CosetEvals> {
        todo!()
    }

    pub fn add_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn sub_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn scale_on(&mut self, scalar: &DScalar<F>, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn add_constant_on(&mut self, scalar: &DScalar<F>, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn sub_constant_on(&mut self, scalar: &DScalar<F>, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn bitreverse(&mut self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn batch_inverse(&mut self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn grand_product(&mut self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn size(&self) -> usize {
        todo!()
    }

    pub fn clone_on(&self, stream: bc_stream) -> CudaResult<Self> {
        todo!()
    }

    pub fn clone_into_on(&self, into: &mut Self, stream: bc_stream) -> CudaResult<Self> {
        todo!()
    }
}

impl<F: PrimeField> Poly<F, CosetEvals> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn icoset_fft_on(&self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis>> {
        todo!()
    }
}

impl<F: PrimeField> Poly<F, LagrangeBasis> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn ifft_on(self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis>> {
        todo!()
    }
}

impl<F: PrimeField> Poly<F, MonomialBasis> {
    pub fn evaluate_at_into_on(
        &self,
        point: &DScalar<F>,
        into: &mut DScalar<F>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        todo!()
    }

    pub fn fft_on(&self, stream: bc_stream) -> CudaResult<Poly<F, LagrangeBasis>> {
        todo!()
    }

    pub fn lde(&self, lde_factor: usize, stream: bc_stream) -> CudaResult<Poly<F, LDE>> {
        assert_eq!(lde_factor, 16);

        todo!()
    }

    pub fn coset_fft_on(
        &self,
        coset_idx: usize,
        lde_factor: usize,
        stream: bc_stream,
    ) -> CudaResult<Poly<F, CosetEvals>> {
        todo!()
    }

    pub fn coset_fft_into_on(
        &self,
        into: &mut Poly<F, CosetEvals>,
        coset_idx: usize,
        lde_factor: usize,
        stream: bc_stream,
    ) -> CudaResult<Poly<F, CosetEvals>> {
        todo!()
    }

    pub fn trim_to_degree(&mut self, new_degree: usize) {
        todo!()
    }
}

impl<F: PrimeField> Poly<F, LDE> {
    pub fn mul_assign_on(&mut self, other: &Self, stream: bc_stream) -> CudaResult<()> {
        todo!()
    }

    pub fn icoset_fft_on(&self, stream: bc_stream) -> CudaResult<Poly<F, MonomialBasis>> {
        todo!()
    }
}
