use super::*;

pub struct DScalar<F: PrimeField>(F);

impl<F: PrimeField> DScalar<F> {
    pub fn from_host_value_on(el: &F, stream: bc_stream) -> CudaResult<Self> {
        todo!()
    }

    pub fn one(stream: bc_stream) -> CudaResult<Self> {
        todo!()
    }

    pub fn as_ptr(&self) -> *const F {
        std::ptr::addr_of!(self.0)
    }

    pub fn as_mut_ptr(&mut self) -> *mut F {
        std::ptr::addr_of_mut!(self.0)
    }
}
