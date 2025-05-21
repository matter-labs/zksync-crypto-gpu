
use ::fflonk::FflonkSnarkVerifierCircuitProof;
use circuit_definitions::circuit_definitions::aux_layer::{
    compression_modes::{
        CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
        CompressionMode4, CompressionMode5ForWrapper,
    },
    ZkSyncCompressionForWrapperProof, ZkSyncCompressionLayerProof,
};

use super::*;

pub struct PlonkSetupData {
    pub compression_mode1_for_wrapper_setup_data: CompressionSetupData<CompressionMode1ForWrapper>,
    pub plonk_snark_wrapper_setup_data: SnarkWrapperSetupData<PlonkSnarkWrapper>,
}

pub struct FflonkSetupData {
    pub compression_mode1_setup_data: CompressionSetupData<CompressionMode1>,
    pub compression_mode2_setup_data: CompressionSetupData<CompressionMode2>,
    pub compression_mode3_setup_data: CompressionSetupData<CompressionMode3>,
    pub compression_mode4_setup_data: CompressionSetupData<CompressionMode4>,
    pub compression_mode5_for_wrapper_setup_data: CompressionSetupData<CompressionMode5ForWrapper>,
    pub fflonk_snark_wrapper_setup_data: SnarkWrapperSetupData<FflonkSnarkWrapper>,
}

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

// #[derive(Clone)]
pub enum SnarkWrapperSetup {
    Plonk(PlonkSetupData),
    FFfonk(FflonkSetupData),
}

pub type SchedulerProof = franklin_crypto::boojum::cs::implementations::proof::Proof<
    GoldilocksField,
    circuit_definitions::circuit_definitions::recursion_layer::RecursiveProofsTreeHasher,
    GoldilocksExt2,
>;

pub fn run_proof_chain(
    snark_wrapper: SnarkWrapper,
    setup_data_cache: &SnarkWrapperSetup,
    scheduler_proof: SchedulerProof,
) -> SnarkWrapperProof {
    match snark_wrapper {
        SnarkWrapper::Plonk => {
            if let SnarkWrapperSetup::Plonk(plonk_setup_data_cache) = setup_data_cache {
                run_proof_chain_with_plonk(plonk_setup_data_cache, scheduler_proof)
            } else {
                panic!("Invalid setup data for Plonk");
            }
        }
        SnarkWrapper::FFfonk => {
            if let SnarkWrapperSetup::FFfonk(fflonk_setup_data_cache) = setup_data_cache {
                run_proof_chain_with_fflonk(fflonk_setup_data_cache, scheduler_proof)
            } else {
                panic!("Invalid setup data for FFfonk");
            }
        }
    }
}

pub fn run_proof_chain_with_fflonk(
    setup_data_cache: &FflonkSetupData,
    scheduler_proof: SchedulerProof,
) -> SnarkWrapperProof {
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();
    <FflonkSnarkWrapper as SnarkWrapperStep>::run_pre_initialization_tasks();
    let next_proof = CompressionMode1::prove_compression_step(
        scheduler_proof,
        &setup_data_cache.compression_mode1_setup_data,
        &context_manager,
    );

    let next_proof = CompressionMode2::prove_compression_step::<SimpleContextManager>(
        next_proof,
        &setup_data_cache.compression_mode2_setup_data,
        &context_manager,
    );
    let next_proof = CompressionMode3::prove_compression_step::<SimpleContextManager>(
        next_proof,
        &setup_data_cache.compression_mode3_setup_data,
        &context_manager,
    );

    let next_proof = CompressionMode4::prove_compression_step::<SimpleContextManager>(
        next_proof,
        &setup_data_cache.compression_mode4_setup_data,
        &context_manager,
    );
    let next_proof = CompressionMode5ForWrapper::prove_compression_step::<SimpleContextManager>(
        next_proof,
        &setup_data_cache.compression_mode5_for_wrapper_setup_data,
        &context_manager,
    );
    println!(
        "Proving entire compression chain took {}s",
        start.elapsed().as_secs()
    );
    let final_proof = FflonkSnarkWrapper::prove_snark_wrapper_step(
        next_proof,
        &setup_data_cache.fflonk_snark_wrapper_setup_data,
    );
    println!(
        "Proving entire chain with snark wrapper took {}s",
        start.elapsed().as_secs()
    );
    SnarkWrapperProof::FFfonk(final_proof)
}

pub fn precompute_proof_chain_with_fflonk(setup_data_cache: FflonkSetupData) -> SnarkWrapperSetup {
    let context_manager = SimpleContextManager::new();
    <FflonkSnarkWrapper as SnarkWrapperStep>::run_pre_initialization_tasks();

    let start = std::time::Instant::now();
    let compression_mode1_setup_data = CompressionMode1::precomputae_and_store_compression_circuits(
        setup_data_cache.compression_mode1_setup_data,
        &context_manager,
    );
    let compression_mode2_setup_data = CompressionMode2::precomputae_and_store_compression_circuits(
        setup_data_cache.compression_mode2_setup_data,
        &context_manager,
    );
    let compression_mode3_setup_data = CompressionMode3::precomputae_and_store_compression_circuits(
        setup_data_cache.compression_mode3_setup_data,
        &context_manager,
    );
    let compression_mode4_setup_data = CompressionMode4::precomputae_and_store_compression_circuits(
        setup_data_cache.compression_mode4_setup_data,
        &context_manager,
    );
    let compression_mode5_for_wrapper_setup_data =
        CompressionMode5ForWrapper::precomputae_and_store_compression_circuits(
            setup_data_cache.compression_mode5_for_wrapper_setup_data,
            &context_manager,
        );
    println!(
        "Precomputation of compression chain took {}s",
        start.elapsed().as_secs()
    );
    let fflonk_snark_wrapper_setup_data =
        FflonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(
            &setup_data_cache.fflonk_snark_wrapper_setup_data,
        );
    println!(
        "Precomputation of entire chain with fflonk took {}s",
        start.elapsed().as_secs()
    );

    SnarkWrapperSetup::FFfonk(FflonkSetupData {
        compression_mode1_setup_data,
        compression_mode2_setup_data,
        compression_mode3_setup_data,
        compression_mode4_setup_data,
        compression_mode5_for_wrapper_setup_data,
        fflonk_snark_wrapper_setup_data,
    })
}

pub fn run_proof_chain_with_plonk(
    setup_data_cache: &PlonkSetupData,
    scheduler_proof: SchedulerProof,
) -> SnarkWrapperProof {
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();

    let next_proof = CompressionMode1ForWrapper::prove_compression_step(
        scheduler_proof,
        &setup_data_cache.compression_mode1_for_wrapper_setup_data,
        &context_manager,
    );

    let final_proof = PlonkSnarkWrapper::prove_snark_wrapper_step(
        next_proof,
        &setup_data_cache.plonk_snark_wrapper_setup_data,
    );
    println!(
        "Entire compression chain with plonk took {}s",
        start.elapsed().as_secs()
    );
    SnarkWrapperProof::Plonk(final_proof)
}

pub fn precompute_proof_chain_with_plonk(setup_data_cache: PlonkSetupData) -> SnarkWrapperSetup {
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();
    let compression_mode1_for_wrapper_setup_data =
        CompressionMode1ForWrapper::precomputae_and_store_compression_circuits(
            setup_data_cache.compression_mode1_for_wrapper_setup_data,
            &context_manager,
        );
    let plonk_snark_wrapper_setup_data =
        PlonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(
            &setup_data_cache.plonk_snark_wrapper_setup_data,
        );
    println!(
        "Precomputation of entire chain with fflonk took {}s",
        start.elapsed().as_secs()
    );
    SnarkWrapperSetup::Plonk(PlonkSetupData {
        compression_mode1_for_wrapper_setup_data,
        plonk_snark_wrapper_setup_data,
    })
}
