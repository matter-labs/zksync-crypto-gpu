use std::sync::{Arc, OnceLock};

use ::fflonk::FflonkSnarkVerifierCircuitProof;
use circuit_definitions::circuit_definitions::aux_layer::{
    compression_modes::{
        CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
        CompressionMode4, CompressionMode5ForWrapper,
    },
    ZkSyncCompressionForWrapperProof, ZkSyncCompressionLayerProof,
};

use super::*;

// pub struct PlonkSetupData {
//     pub compression_mode1_for_wrapper_setup_data: CompressionSetupData<CompressionMode1ForWrapper>,
//     pub plonk_snark_wrapper_setup_data: SnarkWrapperSetupData<PlonkSnarkWrapper>,
// }

// pub struct FflonkSetupData {
//     pub compression_mode1_setup_data: CompressionSetupData<CompressionMode1>,
//     pub compression_mode2_setup_data: CompressionSetupData<CompressionMode2>,
//     pub compression_mode3_setup_data: CompressionSetupData<CompressionMode3>,
//     pub compression_mode4_setup_data: CompressionSetupData<CompressionMode4>,
//     pub compression_mode5_for_wrapper_setup_data: CompressionSetupData<CompressionMode5ForWrapper>,
//     pub fflonk_snark_wrapper_setup_data: SnarkWrapperSetupData<FflonkSnarkWrapper>,
// }

pub struct PlonkSetupData {
    pub compression_mode1_for_wrapper_setup_data:
        Arc<OnceLock<CompressionSetupData<CompressionMode1ForWrapper>>>,
    pub plonk_snark_wrapper_setup_data: Arc<OnceLock<SnarkWrapperSetupData<PlonkSnarkWrapper>>>,
}

pub struct FflonkSetupData {
    pub compression_mode1_setup_data: Arc<OnceLock<CompressionSetupData<CompressionMode1>>>,
    pub compression_mode2_setup_data: Arc<OnceLock<CompressionSetupData<CompressionMode2>>>,
    pub compression_mode3_setup_data: Arc<OnceLock<CompressionSetupData<CompressionMode3>>>,
    pub compression_mode4_setup_data: Arc<OnceLock<CompressionSetupData<CompressionMode4>>>,
    pub compression_mode5_for_wrapper_setup_data:
        Arc<OnceLock<CompressionSetupData<CompressionMode5ForWrapper>>>,
    pub fflonk_snark_wrapper_setup_data: Arc<OnceLock<SnarkWrapperSetupData<FflonkSnarkWrapper>>>,
}

pub struct CompressorSetupData {
    pub fflonk_setup_data: FflonkSetupData,
    pub plonk_setup_data: PlonkSetupData,
}

// impl CompressorSetupData {
//     pub fn new() -> Self {
//         Self {
//             fflonk_setup_data: FflonkSetupData {
//                 compression_mode1_setup_data: CompressionSetupData::new(),
//                 compression_mode2_setup_data: CompressionSetupData::new(),
//                 compression_mode3_setup_data: CompressionSetupData::new(),
//                 compression_mode4_setup_data: CompressionSetupData::new(),
//                 compression_mode5_for_wrapper_setup_data: CompressionSetupData::new(),
//                 fflonk_snark_wrapper_setup_data: SnarkWrapperSetupData::new(),
//             },
//             plonk_setup_data: PlonkSetupData {
//                 compression_mode1_for_wrapper_setup_data: CompressionSetupData::new(),
//                 plonk_snark_wrapper_setup_data: SnarkWrapperSetupData::new(),
//             },
//         }
//     }
// }

impl CompressorSetupData {
    pub fn new() -> Self {
        Self {
            fflonk_setup_data: FflonkSetupData {
                compression_mode1_setup_data: Arc::new(OnceLock::new()),
                compression_mode2_setup_data: Arc::new(OnceLock::new()),
                compression_mode3_setup_data: Arc::new(OnceLock::new()),
                compression_mode4_setup_data: Arc::new(OnceLock::new()),
                compression_mode5_for_wrapper_setup_data: Arc::new(OnceLock::new()),
                fflonk_snark_wrapper_setup_data: Arc::new(OnceLock::new()),
            },
            plonk_setup_data: PlonkSetupData {
                compression_mode1_for_wrapper_setup_data: Arc::new(OnceLock::new()),
                plonk_snark_wrapper_setup_data: Arc::new(OnceLock::new()),
            },
        }
    }
}

// pub struct FflonkSetupData {
//     pub compression_mode1_setup_data: CompressionSetupData<CompressionMode1>,
//     pub compression_mode2_setup_data: CompressionSetupData<CompressionMode2>,
//     pub compression_mode3_setup_data: CompressionSetupData<CompressionMode3>,
//     pub compression_mode4_setup_data: CompressionSetupData<CompressionMode4>,
//     pub compression_mode5_for_wrapper_setup_data: CompressionSetupData<CompressionMode5ForWrapper>,
//     pub fflonk_snark_wrapper_setup_data: SnarkWrapperSetupData<FflonkSnarkWrapper>,
// }

pub trait ProofStorage {
    fn get_scheduler_proof(&self) -> SchedulerProof;
    fn save_compression_layer_proof(&mut self, circuit_id: u8, proof: ZkSyncCompressionLayerProof);
    fn save_compression_wrapper_proof(
        &mut self,
        circuit_id: u8,
        proof: ZkSyncCompressionForWrapperProof,
    );
    fn save_plonk_proof(&mut self, proof: PlonkSnarkVerifierCircuitProof);
    fn save_fflonk_proof(&mut self, proof: FflonkSnarkVerifierCircuitProof);
}

pub struct SimpleProofStorage {
    compression_layer_storage: std::collections::HashMap<u8, ZkSyncCompressionLayerProof>,
    compression_wrapper_storage: std::collections::HashMap<u8, ZkSyncCompressionForWrapperProof>,
    fflonk: Option<FflonkSnarkVerifierCircuitProof>,
    plonk: Option<PlonkSnarkVerifierCircuitProof>,
}

impl SimpleProofStorage {
    pub fn new() -> Self {
        Self {
            compression_layer_storage: std::collections::HashMap::new(),
            compression_wrapper_storage: std::collections::HashMap::new(),
            fflonk: None,
            plonk: None,
        }
    }
}

impl ProofStorage for SimpleProofStorage {
    fn get_scheduler_proof(&self) -> SchedulerProof {
        let scheduler_proof_file =
            std::fs::File::open("./data/scheduler_recursive_proof.json").unwrap();
        let scheduler_proof: circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursionLayerProof =
            serde_json::from_reader(&scheduler_proof_file).unwrap();
        let scheduler_proof = scheduler_proof.into_inner();
        scheduler_proof
    }
    fn save_compression_layer_proof(&mut self, circuit_id: u8, proof: ZkSyncCompressionLayerProof) {
        self.compression_layer_storage.insert(circuit_id, proof);
    }

    fn save_compression_wrapper_proof(
        &mut self,
        circuit_id: u8,
        proof: ZkSyncCompressionForWrapperProof,
    ) {
        self.compression_wrapper_storage.insert(circuit_id, proof);
    }

    fn save_plonk_proof(&mut self, proof: PlonkSnarkVerifierCircuitProof) {
        self.plonk = Some(proof);
    }

    fn save_fflonk_proof(&mut self, proof: FflonkSnarkVerifierCircuitProof) {
        self.fflonk = Some(proof);
    }
}

pub enum SnarkWrapper {
    Plonk,
    FFfonk,
}
pub enum SnarkWrapperProof {
    Plonk(PlonkSnarkVerifierCircuitProof),
    FFfonk(FflonkSnarkVerifierCircuitProof),
}

// pub enum SnarkWrapperSetup {
//     Plonk(PlonkSetupData),
//     FFfonk(FflonkSetupData),
// }

// impl SnarkWrapperSetup {
//     pub fn new(is_fflonk: bool) -> Self {
//         if is_fflonk {
//             SnarkWrapperSetup::FFfonk(FflonkSetupData {
//                 compression_mode1_setup_data: CompressionSetupData::new(),
//                 compression_mode2_setup_data: CompressionSetupData::new(),
//                 compression_mode3_setup_data: CompressionSetupData::new(),
//                 compression_mode4_setup_data: CompressionSetupData::new(),
//                 compression_mode5_for_wrapper_setup_data: CompressionSetupData::new(),
//                 fflonk_snark_wrapper_setup_data: SnarkWrapperSetupData::new(),
//             })
//         } else {
//             SnarkWrapperSetup::Plonk(PlonkSetupData {
//                 compression_mode1_for_wrapper_setup_data: CompressionSetupData::new(),
//                 plonk_snark_wrapper_setup_data: SnarkWrapperSetupData::new(),
//             })
//         }
//     }
// }

pub type SchedulerProof = franklin_crypto::boojum::cs::implementations::proof::Proof<
    GoldilocksField,
    circuit_definitions::circuit_definitions::recursion_layer::RecursiveProofsTreeHasher,
    GoldilocksExt2,
>;

pub fn run_proof_chain(
    snark_wrapper: SnarkWrapper,
    setup_data_cache: Arc<dyn CompressorBlobStorage>,
    scheduler_proof: SchedulerProof,
) -> SnarkWrapperProof {
    match snark_wrapper {
        SnarkWrapper::Plonk => run_proof_chain_with_plonk(setup_data_cache, scheduler_proof),
        SnarkWrapper::FFfonk => run_proof_chain_with_fflonk(setup_data_cache, scheduler_proof),
    }
}

pub fn run_proof_chain_with_fflonk(
    setup_data_cache: Arc<dyn CompressorBlobStorage>,
    scheduler_proof: SchedulerProof,
) -> SnarkWrapperProof {
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();
    <FflonkSnarkWrapper as SnarkWrapperStep>::run_pre_initialization_tasks();

    let start_setup = std::time::Instant::now();
    let snark_wrapper_setup_data = setup_data_cache.get_fflonk_snark_wrapper_setup_data();
    println!(
        "Loading fflonk snark wrapper setup data took {:?}s",
        start_setup.elapsed()
    );

    let start_setup = std::time::Instant::now();
    let compression_mode1_setup_data = setup_data_cache.get_compression_mode1_setup_data();
    println!(
        "Loading compression mode 1 setup data took {:?}s",
        start_setup.elapsed()
    );
    let next_proof = CompressionMode1::prove_compression_step(
        scheduler_proof,
        compression_mode1_setup_data,
        &context_manager,
    );

    let start_setup = std::time::Instant::now();
    let compression_mode2_setup_data = setup_data_cache.get_compression_mode2_setup_data();
    println!(
        "Loading compression mode 2 setup data took {:?}s",
        start_setup.elapsed()
    );
    let next_proof = CompressionMode2::prove_compression_step::<SimpleContextManager>(
        next_proof,
        compression_mode2_setup_data,
        &context_manager,
    );

    let start_setup = std::time::Instant::now();
    let compression_mode3_setup_data = setup_data_cache.get_compression_mode3_setup_data();
    println!(
        "Loading compression mode 3 setup data took {:?}s",
        start_setup.elapsed()
    );
    let next_proof = CompressionMode3::prove_compression_step::<SimpleContextManager>(
        next_proof,
        compression_mode3_setup_data,
        &context_manager,
    );

    let start_setup = std::time::Instant::now();
    let compression_mode4_setup_data = setup_data_cache.get_compression_mode4_setup_data();
    println!(
        "Loading compression mode 4 setup data took {:?}s",
        start_setup.elapsed()
    );
    let next_proof = CompressionMode4::prove_compression_step::<SimpleContextManager>(
        next_proof,
        compression_mode4_setup_data,
        &context_manager,
    );

    let start_setup = std::time::Instant::now();
    let compression_mode5_for_wrapper_setup_data =
        setup_data_cache.get_compression_mode5_for_wrapper_setup_data();
    println!(
        "Loading compression mode 5 for wrapper setup data took {:?}s",
        start_setup.elapsed()
    );
    let next_proof = CompressionMode5ForWrapper::prove_compression_step::<SimpleContextManager>(
        next_proof,
        compression_mode5_for_wrapper_setup_data,
        &context_manager,
    );
    println!(
        "Proving entire compression chain took {}s",
        start.elapsed().as_secs()
    );
    let final_proof =
        FflonkSnarkWrapper::prove_snark_wrapper_step(next_proof, snark_wrapper_setup_data);
    println!(
        "Proving entire chain with snark wrapper took {}s",
        start.elapsed().as_secs()
    );
    SnarkWrapperProof::FFfonk(final_proof)
}

// pub fn precompute_proof_chain_with_fflonk(setup_data_cache: Arc<dyn CompressorBlobStorageExt>) {
//     let context_manager = SimpleContextManager::new();
//     <FflonkSnarkWrapper as SnarkWrapperStep>::run_pre_initialization_tasks();

//     let start = std::time::Instant::now();

//     setup_data_cache.set_compression_mode1_setup_data(
//         CompressionMode1::precompute_and_store_compression_circuits(
//             setup_data_cache.get_compression_mode1_setup_data().unwrap(),
//             &context_manager,
//         ),
//     );
//     setup_data_cache.set_compression_mode2_setup_data(
//         CompressionMode2::precompute_and_store_compression_circuits(
//             setup_data_cache.get_compression_mode2_setup_data().unwrap(),
//             &context_manager,
//         ),
//     );
//     setup_data_cache.set_compression_mode3_setup_data(
//         CompressionMode3::precompute_and_store_compression_circuits(
//             setup_data_cache.get_compression_mode3_setup_data().unwrap(),
//             &context_manager,
//         ),
//     );
//     setup_data_cache.set_compression_mode4_setup_data(
//         CompressionMode4::precompute_and_store_compression_circuits(
//             setup_data_cache.get_compression_mode4_setup_data().unwrap(),
//             &context_manager,
//         ),
//     );
//     setup_data_cache.set_compression_mode5_for_wrapper_setup_data(
//         CompressionMode5ForWrapper::precompute_and_store_compression_circuits(
//             setup_data_cache.get_compression_mode5_for_wrapper_setup_data().unwrap(),
//             &context_manager,
//         ),
//     );
//     println!(
//         "Precomputation of compression chain took {}s",
//         start.elapsed().as_secs()
//     );
//     setup_data_cache.set_fflonk_snark_wrapper_setup_data(
//         FflonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(
//             setup_data_cache.get_fflonk_snark_wrapper_setup_data().unwrap(),
//         ),
//     );
//     println!(
//         "Precomputation of entire chain with fflonk took {}s",
//         start.elapsed().as_secs()
//     );
// }

pub fn run_proof_chain_with_plonk(
    setup_data_cache: Arc<dyn CompressorBlobStorage>,
    scheduler_proof: SchedulerProof,
) -> SnarkWrapperProof {
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();

    let next_proof = CompressionMode1ForWrapper::prove_compression_step(
        scheduler_proof,
        setup_data_cache.get_compression_mode1_for_wrapper_setup_data(),
        &context_manager,
    );

    let final_proof = PlonkSnarkWrapper::prove_snark_wrapper_step(
        next_proof,
        setup_data_cache.get_plonk_snark_wrapper_setup_data(),
    );
    println!(
        "Entire compression chain with plonk took {}s",
        start.elapsed().as_secs()
    );
    SnarkWrapperProof::Plonk(final_proof)
}

// pub fn precompute_proof_chain_with_plonk(setup_data_cache: Arc<dyn CompressorBlobStorageExt>) {
//     let context_manager = SimpleContextManager::new();
//     let start = std::time::Instant::now();

//     setup_data_cache.set_compression_mode1_for_wrapper_setup_data(
//         CompressionMode1ForWrapper::precompute_and_store_compression_circuits(
//             setup_data_cache.get_compression_mode1_for_wrapper_setup_data().unwrap(),
//             &context_manager,
//         ),
//     );
//     println!(
//         "Precomputation of compression chain took {}s",
//         start.elapsed().as_secs()
//     );
//     setup_data_cache.set_plonk_snark_wrapper_setup_data(
//         PlonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(
//             setup_data_cache.get_plonk_snark_wrapper_setup_data().unwrap(),
//         ),
//     );
//     println!(
//         "Precomputation of entire chain with plonk took {}s",
//         start.elapsed().as_secs()
//     );
// }
