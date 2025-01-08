use std::io::Read;

use boojum::cs::implementations::fast_serialization::MemcopySerializable;
use boojum::cs::implementations::setup::FinalizationHintsForProver;
use boojum::cs::traits::gate::FinalizationHintSerialized;
use circuit_definitions::circuit_definitions::aux_layer::ZkSyncCompressionVerificationKey;
use circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursionVerificationKey;

use super::*;

pub trait BlobStorage {
    fn save<W>(data: W)
    where
        W: std::io::Write;
}

pub struct AsyncHandler<T> {
    receiver: std::sync::mpsc::Receiver<T>,
}

impl<T> AsyncHandler<T>
where
    T: Send + Sync + 'static,
{
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> std::sync::mpsc::Receiver<T> + Send + Sync + 'static,
    {
        let receiver = std::thread::spawn(f);

        Self {
            receiver: receiver.join().unwrap(),
        }
    }

    pub fn wait(self) -> T {
        self.receiver.recv().unwrap()
    }
}

use shivini::cs::GpuSetup;

pub trait ArtifactLoader: Sized + Send + Sync {
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
    ) -> Box<dyn Read + Send + Sync + 'static>;
    fn get_compression_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> Box<dyn Read + Send + Sync + 'static>;
    fn get_plonk_snark_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> Box<dyn Read + Send + Sync + 'static>;
    fn get_fflonk_snark_wrapper_precomputation(&self) -> Box<dyn Read + Send + Sync + 'static>;
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
    ) -> Box<dyn Read + Send + Sync + 'static> {
        todo!()
    }

    fn get_compression_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> Box<dyn Read + Send + Sync + 'static> {
        todo!()
    }

    fn get_plonk_snark_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> Box<dyn Read + Send + Sync + 'static> {
        todo!()
    }

    fn get_fflonk_snark_wrapper_precomputation(&self) -> Box<dyn Read + Send + Sync + 'static> {
        todo!()
    }
}
