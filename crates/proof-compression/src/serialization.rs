use super::*;

use fflonk::FflonkSnarkVerifierCircuitDeviceSetup;
use shivini::boojum::cs::implementations::fast_serialization::MemcopySerializable;

use crate::PlonkSnarkVerifierCircuitDeviceSetup;

pub struct PlonkSnarkVerifierCircuitDeviceSetupWrapper(PlonkSnarkVerifierCircuitDeviceSetup);

impl MemcopySerializable for PlonkSnarkVerifierCircuitDeviceSetupWrapper {
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

impl serde::Serialize for PlonkSnarkVerifierCircuitDeviceSetupWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        todo!()
    }
}

impl<'de> serde::Deserialize<'de> for PlonkSnarkVerifierCircuitDeviceSetupWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}

impl PlonkSnarkVerifierCircuitDeviceSetupWrapper {
    pub fn into_inner(self) -> PlonkSnarkVerifierCircuitDeviceSetup {
        self.0
    }
}

pub struct FflonkSnarkVerifierCircuitDeviceSetupWrapper(pub FflonkSnarkVerifierCircuitDeviceSetup);

impl MemcopySerializable for FflonkSnarkVerifierCircuitDeviceSetupWrapper {
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

impl serde::Serialize for FflonkSnarkVerifierCircuitDeviceSetupWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        todo!()
    }
}

impl<'de> serde::Deserialize<'de> for FflonkSnarkVerifierCircuitDeviceSetupWrapper {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        todo!()
    }
}

impl FflonkSnarkVerifierCircuitDeviceSetupWrapper {
    pub fn into_inner(self) -> FflonkSnarkVerifierCircuitDeviceSetup {
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
