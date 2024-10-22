use super::*;

pub trait CombinedMonomialStorage {
    type Poly;

    fn write(&mut self, poly_idx: usize, poly: Self::Poly, stream: bc_stream) -> CudaResult<()>;

    fn read_into(
        &mut self,
        poly_idx: usize,
        into: &mut Self::Poly,
        stream: bc_stream,
    ) -> CudaResult<()>;
}

pub enum GenericCombinedStorage<F, A = std::alloc::Global>
where
    F: PrimeField,
    A: HostAllocator,
{
    HostBased(CombinedMonomialHostStorage<F, A>),
    DeviceBased(CombinedMonomialDeviceStorage<F>),
}

impl<F, A> GenericCombinedStorage<F, A>
where
    F: PrimeField,
    A: HostAllocator,
{
    pub fn allocate_on(device: &Device, domain_size: usize) -> CudaResult<Self> {
        match device {
            Device::A100_40(_) | Device::A100_80(_) => {
                // others
                println!("Using Device based combined storage");
                Ok(GenericCombinedStorage::DeviceBased(
                    CombinedMonomialDeviceStorage::allocate_on(domain_size)?,
                ))
            }
            _ => {
                println!("Using Host based combined storage");
                Ok(GenericCombinedStorage::HostBased(
                    CombinedMonomialHostStorage::<_, A>::allocate_on(domain_size)?,
                ))
            }
        }
    }
}

impl<F> CombinedMonomialStorage for GenericCombinedStorage<F>
where
    F: PrimeField,
{
    type Poly = Poly<F, MonomialBasis>;

    fn write(&mut self, poly_idx: usize, poly: Self::Poly, stream: bc_stream) -> CudaResult<()> {
        match self {
            GenericCombinedStorage::HostBased(storage) => storage.write(poly_idx, poly, stream),
            GenericCombinedStorage::DeviceBased(storage) => storage.write(poly_idx, poly, stream),
        }
    }

    fn read_into(
        &mut self,
        poly_idx: usize,
        into: &mut Self::Poly,
        stream: bc_stream,
    ) -> CudaResult<()> {
        match self {
            GenericCombinedStorage::HostBased(storage) => storage.read_into(poly_idx, into, stream),
            GenericCombinedStorage::DeviceBased(storage) => {
                storage.read_into(poly_idx, into, stream)
            }
        }
    }
}

pub struct CombinedMonomialDeviceStorage<F: PrimeField> {
    pub(crate) combined_monomials: [Option<Poly<F, MonomialBasis>>; 3],
}

impl<F> CombinedMonomialDeviceStorage<F>
where
    F: PrimeField,
{
    fn allocate_on(domain_size: usize) -> CudaResult<Self> {
        let _common_combined_degree = MAX_COMBINED_DEGREE_FACTOR * domain_size;

        Ok(Self {
            combined_monomials: [None, None, None],
        })
    }
}
impl<F> CombinedMonomialStorage for CombinedMonomialDeviceStorage<F>
where
    F: PrimeField,
{
    type Poly = Poly<F, MonomialBasis>;

    fn write(
        &mut self,
        poly_idx: usize,
        poly: Poly<F, MonomialBasis>,
        _stream: bc_stream,
    ) -> CudaResult<()> {
        assert!(self.combined_monomials[poly_idx].is_none());
        self.combined_monomials[poly_idx] = Some(poly);

        Ok(())
    }

    fn read_into(
        &mut self,
        poly_idx: usize,
        dst: &mut Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        let poly = self.combined_monomials[poly_idx]
            .as_ref()
            .expect(&format!("{poly_idx}-th combined poly"));
        assert!(poly.size() <= dst.size());
        let len = poly.size();
        mem::d2d_on(poly.as_ref(), &mut dst.as_mut()[..len], stream)?;
        mem::set_zero(&mut dst.as_mut()[len..], stream)?;

        Ok(())
    }
}

pub struct CombinedMonomialHostStorage<F: PrimeField, A: HostAllocator = std::alloc::Global> {
    pub(crate) combined_monomials: [Vec<F, A>; 3],
    pub(crate) events: [bc_event; 3],
}

impl<F, A> CombinedMonomialHostStorage<F, A>
where
    F: PrimeField,
    A: HostAllocator,
{
    fn allocate_on(domain_size: usize) -> CudaResult<Self> {
        let common_combined_degree = MAX_COMBINED_DEGREE_FACTOR * domain_size;
        let combined_monomials = std::array::from_fn(|_| {
            let mut buf = Vec::with_capacity_in(common_combined_degree, A::default());
            unsafe { buf.set_len(common_combined_degree) };
            buf
        });
        Ok(Self {
            combined_monomials,
            events: [
                bc_event::new().unwrap(),
                bc_event::new().unwrap(),
                bc_event::new().unwrap(),
            ],
        })
    }
}

impl<F, A> CombinedMonomialStorage for CombinedMonomialHostStorage<F, A>
where
    F: PrimeField,
    A: HostAllocator,
{
    type Poly = Poly<F, MonomialBasis>;

    fn write(
        &mut self,
        poly_idx: usize,
        src: Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        assert_eq!(src.size(), self.combined_monomials[poly_idx].len());
        mem::d2h_on(
            src.as_ref(),
            self.combined_monomials[poly_idx].as_mut(),
            stream,
        )?;
        self.events[poly_idx]
            .record(stream)
            .map_err(|err| CudaError::Error(format!("EventRecordErr: {:?}", err)))?;
        Ok(())
    }

    fn read_into(
        &mut self,
        poly_idx: usize,
        dst: &mut Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        stream.wait(self.events[poly_idx]).unwrap();

        let h_poly = &self.combined_monomials[poly_idx];
        assert!(h_poly.len() <= dst.size());
        let len = h_poly.len();
        mem::h2d_on(&h_poly, &mut dst.as_mut()[..len], stream)?;
        mem::set_zero(&mut dst.as_mut()[len..], stream)?;

        Ok(())
    }
}

pub(crate) trait PolyStorage<F, const N: usize>: Sized {
    fn allocate_zeroed(domain_size: usize) -> Self;
    fn num_polys(&self) -> usize {
        N
    }
    fn as_mut_ptr(&mut self) -> *mut F;
}

pub struct MultiMonomialStorage<F: PrimeField, const N: usize>([Poly<F, MonomialBasis>; N]);

pub(crate) type MainGateSelectors<F> = MultiMonomialStorage<F, 5>;
pub(crate) type Permutations<F> = MultiMonomialStorage<F, 3>;
pub(crate) type Trace<F> = MultiMonomialStorage<F, 3>;

impl<F, const N: usize> PolyStorage<F, N> for MultiMonomialStorage<F, N>
where
    F: PrimeField,
{
    fn allocate_zeroed(domain_size: usize) -> Self {
        // constructing permutation polys require storage to be adjacent
        let mut chunks =
            unsafe { DVec::allocate_zeroed(domain_size * N).into_owned_chunks(domain_size) };
        chunks.reverse();
        Self(std::array::from_fn(|_| {
            Poly::<F, MonomialBasis>::from_buffer(chunks.pop().unwrap())
        }))
    }

    fn as_mut_ptr(&mut self) -> *mut F {
        self.0[0].as_mut().as_mut_ptr()
    }
}

impl<F, const N: usize> MultiMonomialStorage<F, N>
where
    F: PrimeField,
{
    pub fn iter(&self) -> std::slice::Iter<Poly<F, MonomialBasis>> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Poly<F, MonomialBasis>> {
        self.0.iter_mut()
    }
}

impl<F, const N: usize> Drop for MultiMonomialStorage<F, N>
where
    F: PrimeField,
{
    fn drop(&mut self) {
        // TODO
        let mut polys = vec![];
        for poly in self.0.iter_mut() {
            let poly = std::mem::replace(
                poly,
                Poly::<F, MonomialBasis>::from_buffer(DVec::dangling()),
            );
            polys.push(poly);
        }
        unsafe {
            let _ = into_owned_poly(polys);
        };
    }
}

unsafe fn into_owned_poly<F>(mut polys: Vec<Poly<F, MonomialBasis>>) -> Poly<F, MonomialBasis>
where
    F: PrimeField,
{
    let num_polys = polys.len();
    let first_poly = polys.remove(0);
    let mut current_ptr = first_poly.storage.as_ptr();
    let chunk_size = first_poly.size();
    for poly in polys.into_iter() {
        assert_eq!(poly.size(), chunk_size);
        assert_eq!(current_ptr.add(chunk_size), poly.storage.as_ptr());
        current_ptr = current_ptr.add(chunk_size);
        std::mem::forget(poly);
    }

    let (ptr, len, alloc, pool, stream) = DVec::into_raw_parts(first_poly.storage);
    assert_eq!(len, chunk_size);
    assert!(pool.is_none());
    assert!(stream.is_none());
    let original_size = chunk_size * num_polys;
    let original_storage = DVec::from_raw_parts_in(ptr, original_size, alloc, pool, stream);

    Poly::from_buffer(original_storage)
}
