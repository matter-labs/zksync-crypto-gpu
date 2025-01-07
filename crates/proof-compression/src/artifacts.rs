use std::io::Read;

use boojum::cs::implementations::fast_serialization::MemcopySerializable;
use boojum::cs::implementations::setup::FinalizationHintsForProver;
use boojum::cs::traits::gate::FinalizationHintSerialized;
use circuit_definitions::circuit_definitions::aux_layer::compression_modes::CompressionTreeHasherForWrapper;
use circuit_definitions::circuit_definitions::aux_layer::{
    CompressionProofsTreeHasher, ZkSyncCompressionVerificationKey,
};
use circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursionVerificationKey;
use shivini::cs::GpuSetup as BoojumDeviceSetup;

use super::*;

pub trait BlobStorage {
    fn save<T>(artifact: T)
    where
        T: MemcopySerializable;
}

pub struct AsyncHandler<T> {
    precomputation: std::sync::mpsc::Receiver<T>, // This is indeed a receiver
}

impl<T> AsyncHandler<T> {
    pub fn into_inner(self) -> T {
        todo!()
    }
}
pub trait ArtifactLoader: Sized {
    fn init<BS>(bs: BS) -> Self
    where
        BS: BlobStorage;
    fn load_scheduler_finalization_hint(&self) -> FinalizationHintsForProver;
    fn read_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read>;
    fn load_compression_layer_finalization_hint(
        &self,
        circuit_id: u8,
    ) -> FinalizationHintsForProver;
    fn load_compression_wrapper_finalization_hint(
        &self,
        circuit_id: u8,
    ) -> FinalizationHintSerialized;
    fn read_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read>;
    fn load_plonk_finalization_hint(&self) -> usize;

    fn load_scheduler_vk(&self) -> ZkSyncRecursionVerificationKey;
    fn load_compression_layer_vk(&self, circuit_id: u8) -> ZkSyncCompressionVerificationKey;
    fn read_scheduler_vk(&self) -> Box<dyn Read>;
    fn read_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Read>;
    fn read_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Read>;
    fn read_plonk_vk(&self) -> Box<dyn Read>;
    fn read_fflonk_vk(&self) -> Box<dyn Read>;
    fn get_compression_layer_precomputation(
        &self,
        circuit_id: u8,
    ) -> AsyncHandler<TreeHasherCompatibleGpuSetup<CompressionProofsTreeHasher>>;
    fn get_compression_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> AsyncHandler<TreeHasherCompatibleGpuSetup<CompressionTreeHasherForWrapper>>;
    fn get_plonk_snark_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> AsyncHandler<PlonkSnarkVerifierCircuitDeviceSetupWrapper>;
    fn get_fflonk_snark_wrapper_precomputation(
        &self,
    ) -> AsyncHandler<FflonkSnarkVerifierCircuitDeviceSetupWrapper>;
}

pub struct SimpleArtifactLoader;

impl ArtifactLoader for SimpleArtifactLoader {
    fn init<BS>(bs: BS) -> Self
    where
        BS: BlobStorage,
    {
        todo!()
    }

    fn load_scheduler_finalization_hint(&self) -> FinalizationHintsForProver {
        todo!()
    }

    fn read_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read> {
        todo!()
    }

    fn load_compression_layer_finalization_hint(
        &self,
        circuit_id: u8,
    ) -> FinalizationHintsForProver {
        todo!()
    }

    fn load_compression_wrapper_finalization_hint(
        &self,
        circuit_id: u8,
    ) -> FinalizationHintSerialized {
        todo!()
    }

    fn read_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read> {
        todo!()
    }

    fn load_plonk_finalization_hint(&self) -> usize {
        todo!()
    }

    fn load_scheduler_vk(&self) -> ZkSyncRecursionVerificationKey {
        todo!()
    }

    fn load_compression_layer_vk(&self, circuit_id: u8) -> ZkSyncCompressionVerificationKey {
        todo!()
    }

    fn read_scheduler_vk(&self) -> Box<dyn Read> {
        todo!()
    }

    fn read_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Read> {
        todo!()
    }

    fn read_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Read> {
        todo!()
    }

    fn read_plonk_vk(&self) -> Box<dyn Read> {
        todo!()
    }

    fn read_fflonk_vk(&self) -> Box<dyn Read> {
        todo!()
    }

    fn get_compression_layer_precomputation(
        &self,
        circuit_id: u8,
    ) -> AsyncHandler<TreeHasherCompatibleGpuSetup<CompressionProofsTreeHasher>> {
        todo!()
    }

    fn get_compression_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> AsyncHandler<TreeHasherCompatibleGpuSetup<CompressionTreeHasherForWrapper>> {
        todo!()
    }

    fn get_plonk_snark_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> AsyncHandler<PlonkSnarkVerifierCircuitDeviceSetupWrapper> {
        todo!()
    }

    fn get_fflonk_snark_wrapper_precomputation(
        &self,
    ) -> AsyncHandler<FflonkSnarkVerifierCircuitDeviceSetupWrapper> {
        todo!()
    }
}
