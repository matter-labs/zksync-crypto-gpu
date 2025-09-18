use super::*;

use crate::bellman::bn256::Bn256;
use gpu_prover::ManagerConfigs;
use shivini::{
    boojum::cs::implementations::fast_serialization::MemcopySerializable, cs::GpuSetup,
    GpuTreeHasher,
};

pub trait GenericWrapper: Sized {
    type Inner;
    fn into_inner(self) -> Self::Inner;
    fn into_inner_ref(&self) -> &Self::Inner;
    fn from_inner(inner: Self::Inner) -> Self;
}

use crate::PlonkSnarkVerifierCircuitDeviceSetup;

pub struct BoojumDeviceSetupWrapper<H: GpuTreeHasher>(pub GpuSetup<H>);
impl<H> GenericWrapper for BoojumDeviceSetupWrapper<H>
where
    H: GpuTreeHasher,
{
    type Inner = GpuSetup<H>;
    fn into_inner(self) -> Self::Inner {
        self.0
    }
    fn into_inner_ref(&self) -> &Self::Inner {
        &self.0
    }
    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }
}

impl<H: GpuTreeHasher> shivini::boojum::cs::implementations::fast_serialization::MemcopySerializable
    for BoojumDeviceSetupWrapper<H>
{
    fn write_into_buffer<W: std::io::Write>(
        &self,
        dst: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(bincode::serialize_into(dst, &self.0).unwrap())
    }

    fn read_from_buffer<R: std::io::Read>(src: R) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self(bincode::deserialize_from(src).unwrap()))
    }
}

#[derive(Debug, Clone)]
pub struct PlonkSnarkVerifierCircuitDeviceSetupWrapper(pub PlonkSnarkVerifierCircuitDeviceSetup);

impl MemcopySerializable for PlonkSnarkVerifierCircuitDeviceSetupWrapper {
    fn write_into_buffer<W: std::io::Write>(
        &self,
        dst: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.0.write(dst).unwrap();
        Ok(())
    }

    fn read_from_buffer<R: std::io::Read>(src: R) -> Result<Self, Box<dyn std::error::Error>> {
        let mut precomputation = PlonkSnarkVerifierCircuitDeviceSetup::allocate(
            1 << PlonkProverDeviceMemoryManagerConfig::FULL_SLOT_SIZE_LOG,
        );
        precomputation.read(src).unwrap();
        Ok(Self(precomputation))
    }
}

impl GenericWrapper for PlonkSnarkVerifierCircuitDeviceSetupWrapper {
    type Inner = PlonkSnarkVerifierCircuitDeviceSetup;
    fn into_inner(self) -> Self::Inner {
        self.0
    }
    fn into_inner_ref(&self) -> &Self::Inner {
        &self.0
    }
    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }
}

pub struct FflonkSnarkVerifierCircuitDeviceSetupWrapper<A: HostAllocator>(
    pub ::fflonk::FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, A>,
);

impl<A> MemcopySerializable for FflonkSnarkVerifierCircuitDeviceSetupWrapper<A>
where
    A: HostAllocator,
{
    fn write_into_buffer<W: std::io::Write>(
        &self,
        dst: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(self.0.write(dst).unwrap())
    }

    fn read_from_buffer<R: std::io::Read>(src: R) -> Result<Self, Box<dyn std::error::Error>> {
        let precomputation = ::fflonk::FflonkDeviceSetup::<Bn256, _, A>::read(src).unwrap();

        Ok(Self(precomputation))
    }
}

impl<A> GenericWrapper for FflonkSnarkVerifierCircuitDeviceSetupWrapper<A>
where
    A: HostAllocator,
{
    type Inner = ::fflonk::FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, A>;
    fn into_inner(self) -> Self::Inner {
        self.0
    }
    fn into_inner_ref(&self) -> &Self::Inner {
        &self.0
    }
    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct MarkerPrecomputation;
impl MemcopySerializable for MarkerPrecomputation {
    fn write_into_buffer<W: std::io::Write>(
        &self,
        _dst: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        todo!()
    }

    fn read_from_buffer<R: std::io::Read>(_src: R) -> Result<Self, Box<dyn std::error::Error>> {
        todo!()
    }
}
