use super::*;

use gpu_prover::ManagerConfigs;
use shivini::{
    boojum::cs::implementations::fast_serialization::MemcopySerializable, cs::GpuSetup,
    GpuTreeHasher,
};

use crate::PlonkSnarkVerifierCircuitDeviceSetup;

pub struct BoojumDeviceSetupWrapper<H: GpuTreeHasher>(GpuSetup<H>);

impl<H: GpuTreeHasher> boojum::cs::implementations::fast_serialization::MemcopySerializable
    for BoojumDeviceSetupWrapper<H>
{
    fn write_into_buffer<W: std::io::Write>(
        &self,
        dst: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(bincode::serialize_into(dst, self.0).unwrap())
    }

    fn read_from_buffer<R: std::io::Read>(src: R) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self(bincode::deserialize_from(src).unwrap()))
    }
}

pub struct PlonkSnarkVerifierCircuitDeviceSetupWrapper(PlonkSnarkVerifierCircuitDeviceSetup);

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

impl PlonkSnarkVerifierCircuitDeviceSetupWrapper {
    pub fn into_inner(self) -> PlonkSnarkVerifierCircuitDeviceSetup {
        self.0
    }
}

pub struct FflonkSnarkVerifierCircuitDeviceSetupWrapper<A: HostAllocator>(
    pub fflonk::FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, A>,
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
        let precomputation = fflonk::FflonkDeviceSetup::<Bn256, _, A>::read(src).unwrap();

        Ok(Self(precomputation))
    }
}

impl<A> FflonkSnarkVerifierCircuitDeviceSetupWrapper<A>
where
    A: HostAllocator,
{
    pub fn into_inner(self) -> fflonk::FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, A> {
        self.0
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MarkerPrecomputation;
impl MemcopySerializable for MarkerPrecomputation {
    fn write_into_buffer<W: std::io::Write>(
        &self,
        dst: W,
    ) -> Result<(), Box<dyn std::error::Error>> {
        todo!()
    }

    fn read_from_buffer<R: std::io::Read>(src: R) -> Result<Self, Box<dyn std::error::Error>> {
        todo!()
    }
}
