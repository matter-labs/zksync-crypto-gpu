use std::sync::Arc;

use ::fflonk::FflonkSnarkVerifierCircuitProof;
use anyhow::Context;
use circuit_definitions::circuit_definitions::aux_layer::{
    compression_modes::{
        CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
        CompressionMode4, CompressionMode5ForWrapper,
    },
    ZkSyncCompressionForWrapperProof, ZkSyncCompressionLayerProof,
};

use super::*;

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
    Fflonk,
}
pub enum SnarkWrapperProof {
    Plonk(PlonkSnarkVerifierCircuitProof),
    Fflonk(FflonkSnarkVerifierCircuitProof),
}

pub type SchedulerProof = franklin_crypto::boojum::cs::implementations::proof::Proof<
    GoldilocksField,
    circuit_definitions::circuit_definitions::recursion_layer::RecursiveProofsTreeHasher,
    GoldilocksExt2,
>;

pub fn run_proof_chain(
    snark_wrapper: SnarkWrapper,
    setup_data_cache: Arc<dyn CompressorBlobStorage>,
    scheduler_proof: SchedulerProof,
) -> anyhow::Result<SnarkWrapperProof> {
    match snark_wrapper {
        SnarkWrapper::Plonk => run_proof_chain_with_plonk(setup_data_cache, scheduler_proof),
        SnarkWrapper::Fflonk => run_proof_chain_with_fflonk(setup_data_cache, scheduler_proof),
    }
}

pub fn run_proof_chain_with_fflonk(
    setup_data_cache: Arc<dyn CompressorBlobStorage>,
    scheduler_proof: SchedulerProof,
) -> anyhow::Result<SnarkWrapperProof> {
    let start = std::time::Instant::now();
    <FflonkSnarkWrapper as SnarkWrapperStep>::run_pre_initialization_tasks();

    let next_proof = CompressionMode1::prove_compression_step(
        scheduler_proof,
        setup_data_cache.get_compression_mode1_setup_data()?,
    ).context("Failed to prove compression mode 1 step")?;

    let next_proof = CompressionMode2::prove_compression_step(
        next_proof,
        setup_data_cache.get_compression_mode2_setup_data()?,
    ).context("Failed to prove compression mode 2 step")?;
    let next_proof = CompressionMode3::prove_compression_step(
        next_proof,
        setup_data_cache.get_compression_mode3_setup_data()?,
    ).context("Failed to prove compression mode 3 step")?;
    let next_proof = CompressionMode4::prove_compression_step(
        next_proof,
        setup_data_cache.get_compression_mode4_setup_data()?,
    ).context("Failed to prove compression mode 4 step")?;
    let next_proof = CompressionMode5ForWrapper::prove_compression_step(
        next_proof,
        setup_data_cache.get_compression_mode5_for_wrapper_setup_data()?,
    ).context("Failed to prove compression mode 5 for wrapper step")?;
    println!(
        "Proving entire compression chain took {}s",
        start.elapsed().as_secs()
    );
    let final_proof = FflonkSnarkWrapper::prove_snark_wrapper_step(
        next_proof,
        setup_data_cache.get_fflonk_snark_wrapper_setup_data()?,
    ).context("Failed to prove Fflonk snark wrapper step")?;
    println!(
        "Proving entire chain with snark wrapper took {}s",
        start.elapsed().as_secs()
    );
    Ok(SnarkWrapperProof::Fflonk(final_proof))
}

pub fn precompute_proof_chain_with_fflonk(setup_data_cache: Arc<dyn CompressorBlobStorageExt>) -> anyhow::Result<()> {
    <FflonkSnarkWrapper as SnarkWrapperStep>::run_pre_initialization_tasks();

    let start = std::time::Instant::now();

    let (precomputation, vk, finalization_hint) = CompressionMode1::precompute_compression_circuits(
        setup_data_cache
            .get_compression_mode1_previous_vk()
            .context("Failed to get compression mode 1 previous vk")?,
    ).context("Failed to precompute compression mode 1 circuits")?;
    setup_data_cache
        .set_compression_mode1_setup_data(&precomputation, &vk, &finalization_hint)
        .context("Failed to set compression mode 1 setup data")?;

    let (precomputation, vk, finalization_hint) = CompressionMode2::precompute_compression_circuits(
        setup_data_cache
            .get_compression_mode2_previous_vk()
            .context("Failed to get compression mode 2 previous vk")?,
    ).context("Failed to precompute compression mode 2 circuits")?;
    setup_data_cache
        .set_compression_mode2_setup_data(&precomputation, &vk, &finalization_hint)
        .context("Failed to set compression mode 2 setup data")?;

    let (precomputation, vk, finalization_hint) = CompressionMode3::precompute_compression_circuits(
        setup_data_cache
            .get_compression_mode3_previous_vk()
            .context("Failed to get compression mode 3 previous vk")?,
    ).context("Failed to precompute compression mode 3 circuits")?;
    setup_data_cache
        .set_compression_mode3_setup_data(&precomputation, &vk, &finalization_hint)
        .context("Failed to set compression mode 3 setup data")?;

    let (precomputation, vk, finalization_hint) = CompressionMode4::precompute_compression_circuits(
        setup_data_cache
            .get_compression_mode4_previous_vk()
            .context("Failed to get compression mode 4 previous vk")?,
    ).context("Failed to precompute compression mode 4 circuits")?;
    setup_data_cache
        .set_compression_mode4_setup_data(&precomputation, &vk, &finalization_hint)
        .context("Failed to set compression mode 4 setup data")?;

    let (precomputation, vk, finalization_hint) =
        CompressionMode5ForWrapper::precompute_compression_circuits(
            setup_data_cache
                .get_compression_mode5_for_wrapper_previous_vk()
                .context("Failed to get compression mode 5 for wrapper previous vk")?,
        ).context("Failed to precompute compression mode 5 for wrapper circuits"
        )?;
    setup_data_cache
        .set_compression_mode5_for_wrapper_setup_data(&precomputation, &vk, &finalization_hint)
        .context("Failed to set compression mode 5 for wrapper setup data")?;

    println!(
        "Precomputation of compression chain took {}s",
        start.elapsed().as_secs()
    );

    let (previous_vk, finalization_hint, ctx) = setup_data_cache
        .get_fflonk_snark_wrapper_previous_vk_finalization_hint_and_ctx()
        .context("Failed to get Fflonk snark wrapper previous vk, finalization hint and context")?;
    let (precomputation, vk) =
        FflonkSnarkWrapper::precompute_snark_wrapper_circuit(previous_vk, finalization_hint, ctx)
        .context("Failed to precompute Fflonk snark wrapper setup data")?;
    setup_data_cache
        .set_fflonk_snark_wrapper_setup_data(&precomputation, &vk)
        .context("Failed to set Fflonk snark wrapper setup data")?;
    println!(
        "Precomputation of entire chain with fflonk took {}s",
        start.elapsed().as_secs()
    );
    Ok(())
}

pub fn run_proof_chain_with_plonk(
    setup_data_cache: Arc<dyn CompressorBlobStorage>,
    scheduler_proof: SchedulerProof,
) -> anyhow::Result<SnarkWrapperProof> {
    let start = std::time::Instant::now();

    let next_proof = CompressionMode1ForWrapper::prove_compression_step(
        scheduler_proof,
        setup_data_cache.get_compression_mode1_for_wrapper_setup_data()?,
    ).context("Failed to prove compression mode 1 for wrapper step")?;

    let final_proof = PlonkSnarkWrapper::prove_plonk_snark_wrapper_step(
        next_proof,
        setup_data_cache.get_plonk_snark_wrapper_setup_data()?,
    ).context("Failed to prove Plonk snark wrapper step")?;
    println!(
        "Entire compression chain with plonk took {}s",
        start.elapsed().as_secs()
    );
    Ok(SnarkWrapperProof::Plonk(final_proof))
}

pub fn precompute_proof_chain_with_plonk(setup_data_cache: Arc<dyn CompressorBlobStorageExt>) -> anyhow::Result<()> {
    let start = std::time::Instant::now();

    let (precomputation, vk, finalization_hint) =
        CompressionMode1ForWrapper::precompute_compression_circuits(
            setup_data_cache
                .get_compression_mode1_for_wrapper_previous_vk()
                .context("Failed to get compression mode 1 for wrapper previous vk")?,
        ).context("Failed to precompute compression mode 1 for wrapper circuits")?;
    setup_data_cache
        .set_compression_mode1_for_wrapper_setup_data(&precomputation, &vk, &finalization_hint)
        .context("Failed to set compression mode 1 for wrapper setup data")?;
    println!(
        "Precomputation of compression chain took {}s",
        start.elapsed().as_secs()
    );
    let (previous_vk, finalization_hint, ctx) = setup_data_cache
        .get_plonk_snark_wrapper_previous_vk_finalization_hint_and_ctx()
        .context("Failed to get Plonk snark wrapper previous vk, finalization hint and context")?;
    let (precomputation, vk) = PlonkSnarkWrapper::precompute_plonk_snark_wrapper_circuit(
        previous_vk,
        finalization_hint,
        ctx,
    ).context("Failed to precompute Plonk snark wrapper setup data")?;
    setup_data_cache
        .set_plonk_snark_wrapper_setup_data(&precomputation, &vk)
        .context("Failed to set Plonk snark wrapper setup data")?;

    println!(
        "Precomputation of entire chain with plonk took {}s",
        start.elapsed().as_secs()
    );
    Ok(())
}
