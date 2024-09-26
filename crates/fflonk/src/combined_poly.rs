use std::alloc::Global;

use super::*;

pub trait CombinedMonomialStorage<F>: Sized
where
    F: PrimeField,
{
    fn allocate_on(domain_size: usize, stream: bc_stream) -> CudaResult<Self>;

    fn write(
        &mut self,
        poly_idx: usize,
        poly: Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()>;

    fn read(&mut self, poly_idx: usize, stream: bc_stream) -> CudaResult<&Poly<F, MonomialBasis>>;

    fn read_into(
        &mut self,
        poly_idx: usize,
        into: &mut Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()>;
}

pub struct CombinedMonomialDeviceStorage<F: PrimeField> {
    pub(crate) combined_monomials: [Option<Poly<F, MonomialBasis>>; 3],
}

impl<F> CombinedMonomialStorage<F> for CombinedMonomialDeviceStorage<F>
where
    F: PrimeField,
{
    fn allocate_on(domain_size: usize, stream: bc_stream) -> CudaResult<Self> {
        Ok(Self {
            combined_monomials: [None, None, None],
        })
    }

    fn write(
        &mut self,
        poly_idx: usize,
        poly: Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        assert!(self.combined_monomials[poly_idx].is_none());
        self.combined_monomials[poly_idx] = Some(poly);

        Ok(())
    }

    fn read(&mut self, poly_idx: usize, stream: bc_stream) -> CudaResult<&Poly<F, MonomialBasis>> {
        Ok(self.combined_monomials[poly_idx]
            .as_ref()
            .expect(&format!("{poly_idx}-th combined poly")))
    }

    fn read_into(
        &mut self,
        poly_idx: usize,
        dst: &mut Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        let len = self.combined_monomials[poly_idx]
            .as_ref()
            .map(|p| p.size())
            .expect(&format!("{poly_idx}-th combined poly"));
        mem::d2d_on(
            self.combined_monomials[poly_idx]
                .as_ref()
                .expect(&format!("{poly_idx}-th combined poly"))
                .as_ref(),
            &mut dst.as_mut()[..len],
            stream,
        )?;
        mem::set_zero(&mut dst.as_mut()[len..], stream)?;

        Ok(())
    }
}

pub struct CombinedMonomialHostStorage<F: PrimeField, A: HostAllocator = Global> {
    pub(crate) combined_monomials: [Vec<F, A>; 3],
    pub(crate) events: [bc_event; 3],
    pub(crate) device_storage: Poly<F, MonomialBasis>,
}

impl<F, A> CombinedMonomialStorage<F> for CombinedMonomialHostStorage<F, A>
where
    F: PrimeField,
    A: HostAllocator,
{
    fn allocate_on(domain_size: usize, stream: bc_stream) -> CudaResult<Self> {
        todo!()
    }
    fn write(
        &mut self,
        poly_idx: usize,
        src: Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        let event = bc_event::new().map_err(|_| CudaError::Error(format!("EventErr")))?;
        mem::d2h_on(
            src.as_ref(),
            self.combined_monomials[poly_idx].as_mut(),
            stream,
        )?;
        event
            .record(stream)
            .map_err(|_| CudaError::Error(format!("EventErr")))?;

        Ok(())
    }

    fn read(&mut self, poly_idx: usize, stream: bc_stream) -> CudaResult<&Poly<F, MonomialBasis>> {
        self.events[poly_idx]
            .sync()
            .map_err(|_| CudaError::Error(format!("EventSyncErr")))?;
        let len = self.combined_monomials[poly_idx].len();

        mem::h2d_on(
            &self.combined_monomials[poly_idx],
            &mut self.device_storage.as_mut()[..len],
            stream,
        )?;

        mem::set_zero(&mut self.device_storage.as_mut()[len..], stream)?;

        Ok(&self.device_storage)
    }

    fn read_into(
        &mut self,
        poly_idx: usize,
        dst: &mut Poly<F, MonomialBasis>,
        stream: bc_stream,
    ) -> CudaResult<()> {
        self.events[poly_idx]
            .sync()
            .map_err(|_| CudaError::Error(format!("EventSyncErr")))?;

        let len = self.combined_monomials[poly_idx].len();
        mem::h2d_on(
            &self.combined_monomials[poly_idx],
            &mut dst.as_mut()[..len],
            stream,
        )?;

        mem::set_zero(&mut dst.as_mut()[len..], stream)?;
        Ok(())
    }
}
