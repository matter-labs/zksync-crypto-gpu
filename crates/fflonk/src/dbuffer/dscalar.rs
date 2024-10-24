use super::*;

pub struct DScalar<F: PrimeField>(DVec<F, PoolAllocator>);

static mut _SMALL_SCALAR_MEMPOOL: Option<bc_mem_pool> = None;

pub(crate) fn init_small_scalar_mempool() {
    assert!(is_small_scalar_mempool_initialized() == false);
    unsafe {
        _SMALL_SCALAR_MEMPOOL = Some(bc_mem_pool::new(DEFAULT_DEVICE_ID).unwrap());
    }
}

pub(crate) fn _small_scalar_mempool() -> bc_mem_pool {
    unsafe { _SMALL_SCALAR_MEMPOOL.expect("small scalar mempool intialized") }
}

pub(crate) fn is_small_scalar_mempool_initialized() -> bool {
    unsafe { _SMALL_SCALAR_MEMPOOL.is_some() }
}

pub(crate) fn destroy_small_scalar_mempool() {
    unsafe {
        let pool = _SMALL_SCALAR_MEMPOOL.take().unwrap();
        let result = gpu_ffi::bc_mem_pool_destroy(pool);
        if result != 0 {
            panic!("Couldn't destroy small scalar pool");
        }
    }
}

impl<F: PrimeField> DScalar<F> {
    pub fn zero(stream: bc_stream) -> CudaResult<DScalar<F>> {
        assert!(is_small_scalar_mempool_initialized());
        Ok(Self(DVec::allocate_zeroed_on(
            1,
            _small_scalar_mempool(),
            stream,
        )))
    }

    pub fn one(stream: bc_stream) -> CudaResult<Self> {
        Self::from_host_value_on(&F::one(), stream)
    }

    pub fn multiplicative_generator(stream: bc_stream) -> CudaResult<Self> {
        Self::from_host_value_on(&F::multiplicative_generator(), stream)
    }

    pub fn inv_multiplicative_generator(stream: bc_stream) -> CudaResult<Self> {
        Self::from_host_value_on(&F::multiplicative_generator().inverse().unwrap(), stream)
    }

    pub fn from_host_value_on(h_el: &F, stream: bc_stream) -> CudaResult<Self> {
        let mut this = Self::zero(stream)?;
        this.copy_from_host_value_on(h_el, stream)?;

        Ok(this)
    }

    pub fn copy_from_host_value_on(&mut self, h_el: &F, stream: bc_stream) -> CudaResult<()> {
        mem::h2d_on(&[*h_el], self.0.as_mut(), stream)
    }

    pub fn to_host_value_on(&self, stream: bc_stream) -> CudaResult<F> {
        let mut buf = vec![F::zero()];
        mem::d2h_on(self.0.as_ref(), &mut buf, stream)?;

        Ok(buf.pop().unwrap())
    }

    pub fn as_ptr(&self) -> *const F {
        self.0.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut F {
        self.0.as_mut_ptr()
    }
}

impl<F> std::fmt::Debug for DScalar<F>
where
    F: PrimeField,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stream = bc_stream::new().unwrap();
        let el = self.to_host_value_on(stream).unwrap();
        stream.sync().unwrap();
        println!("{}", el);
        Ok(())
    }
}

pub struct DScalars<F: PrimeField>(pub(crate) Vec<DScalar<F>>);

impl<F> DScalars<F>
where
    F: PrimeField,
{
    pub fn allocate_zeroed_on(num_elems: usize, stream: bc_stream) -> CudaResult<Self> {
        let mut scalars = vec![];
        for _ in 0..num_elems {
            let scalar = DScalar::zero(stream)?;
            scalars.push(scalar);
        }
        Ok(Self(scalars))
    }

    pub fn from_host_scalars_on(h_scalars: &[F], stream: bc_stream) -> CudaResult<Self> {
        let mut scalars = Self::allocate_zeroed_on(h_scalars.len(), stream)?;
        for (src, dst) in h_scalars.iter().zip(scalars.iter_mut()) {
            dst.copy_from_host_value_on(src, stream)?;
        }

        Ok(scalars)
    }

    pub fn to_host_scalars_on(self, stream: bc_stream) -> CudaResult<Vec<F>> {
        Ok(self
            .0
            .iter()
            .map(|el| el.to_host_value_on(stream).unwrap())
            .collect())
    }

    pub fn to_dvec(self, stream: bc_stream) -> CudaResult<DVec<F, PoolAllocator>> {
        // Vec<DScalars> -> DSlice<F>
        let mut new = DVec::<F, PoolAllocator>::allocate_zeroed_on(
            self.len(),
            _small_scalar_mempool(),
            stream,
        );

        for idx in 0..self.len() {
            mem::set_value(&mut new[idx..idx + 1], &self.0[idx], stream)?;
        }

        Ok(new)
    }

    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (&mut DSlice<DScalar<F>>, &mut DSlice<DScalar<F>>) {
        let (this, other) = self.0.split_at_mut(mid);
        unsafe { (std::mem::transmute(this), std::mem::transmute(other)) }
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> DIter<DScalar<F>> {
        let this = unsafe { std::mem::transmute(&self.0[..]) };
        DIter::new(this)
    }

    pub fn iter_mut(&mut self) -> DIterMut<DScalar<F>> {
        let this = unsafe { std::mem::transmute(&mut self.0[..]) };
        DIterMut::new(this)
    }
}
