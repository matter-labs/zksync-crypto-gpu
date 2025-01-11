use circuit_definitions::circuit_definitions::aux_layer::compression_modes::{
    CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
    CompressionMode4, CompressionMode5ForWrapper,
};

use super::*;

pub enum SnarkWrapper {
    Plonk,
    FFfonk,
}
pub enum SnarkWrapperProof {
    Plonk(PlonkSnarkVerifierCircuitProof),
    FFfonk(FflonkSnarkVerifierCircuitProof),
}

pub type SchedulerProof = franklin_crypto::boojum::cs::implementations::proof::Proof<
    GoldilocksField,
    circuit_definitions::circuit_definitions::recursion_layer::RecursiveProofsTreeHasher,
    GoldilocksExt2,
>;

pub fn run_proof_chain<BS>(
    input_proof: SchedulerProof,
    snark_wrapper: SnarkWrapper,
    blob_storage: &BS,
) -> SnarkWrapperProof
where
    BS: BlobStorage,
{
    match snark_wrapper {
        SnarkWrapper::Plonk => {
            let proof = run_proof_chain_with_plonk(input_proof, blob_storage);
            SnarkWrapperProof::Plonk(proof)
        }
        SnarkWrapper::FFfonk => {
            let proof = run_proof_chain_with_fflonk(input_proof, blob_storage);
            SnarkWrapperProof::FFfonk(proof)
        }
    }
}

pub fn run_proof_chain_with_fflonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: &BS,
) -> FflonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();
    let compact_raw_crs =
        <FflonkSnarkWrapper as SnarkWrapperStep>::load_compact_raw_crs(blob_storage);
    let fflonk_precomputation = FflonkSnarkWrapper::get_precomputation(blob_storage);

    let next_proof =
        CompressionMode1::prove_compression_step(input_proof, blob_storage, &context_manager);
    let next_proof = CompressionMode2::prove_compression_step::<_, SimpleContextManager>(
        next_proof,
        blob_storage,
        &context_manager,
    );
    let next_proof = CompressionMode3::prove_compression_step::<_, SimpleContextManager>(
        next_proof,
        blob_storage,
        &context_manager,
    );
    let next_proof = CompressionMode4::prove_compression_step::<_, SimpleContextManager>(
        next_proof,
        blob_storage,
        &context_manager,
    );
    let next_proof = CompressionMode5ForWrapper::prove_compression_step::<_, SimpleContextManager>(
        next_proof,
        blob_storage,
        &context_manager,
    );
    println!(
        "Proving entire compression chain took {}s",
        start.elapsed().as_secs()
    );
    let final_proof = FflonkSnarkWrapper::prove_snark_wrapper_step::<_, SimpleContextManager>(
        compact_raw_crs,
        fflonk_precomputation,
        next_proof,
        blob_storage,
        &context_manager,
    );
    println!(
        "Proving entire chain with snark wrapper took {}s",
        start.elapsed().as_secs()
    );
    final_proof
}

pub fn precompute_proof_chain_with_fflonk<BS>(blob_storage: &BS)
where
    BS: BlobStorageExt,
{
    let context_manager = SimpleContextManager::new();
    let compact_raw_crs =
        <FflonkSnarkWrapper as SnarkWrapperStep>::load_compact_raw_crs(blob_storage);

    let start = std::time::Instant::now();
    CompressionMode1::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    CompressionMode2::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    CompressionMode3::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    CompressionMode4::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    CompressionMode5ForWrapper::precomputae_and_store_compression_circuits(
        blob_storage,
        &context_manager,
    );
    println!(
        "Precomputation of compression chain took {}s",
        start.elapsed().as_secs()
    );
    FflonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(
        compact_raw_crs,
        blob_storage,
        &context_manager,
    );
    println!(
        "Precomputation of entire chain with fflonk took {}s",
        start.elapsed().as_secs()
    );
}

pub fn run_proof_chain_with_plonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: &BS,
) -> PlonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();
    let compact_raw_crs =
        <PlonkSnarkWrapper as SnarkWrapperStep>::load_compact_raw_crs(blob_storage);
    let plonk_precomputation = PlonkSnarkWrapper::get_precomputation(blob_storage);

    let next_proof = CompressionMode1ForWrapper::prove_compression_step(
        input_proof,
        blob_storage,
        &context_manager,
    );

    let final_proof = PlonkSnarkWrapper::prove_snark_wrapper_step(
        compact_raw_crs,
        plonk_precomputation,
        next_proof,
        blob_storage,
        &context_manager,
    );
    println!(
        "Entire compression chain with plonk took {}s",
        start.elapsed().as_secs()
    );
    final_proof
}

pub fn precompute_proof_chain_with_plonk<BS>(blob_storage: &BS)
where
    BS: BlobStorageExt,
{
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();
    let compact_raw_crs =
        <PlonkSnarkWrapper as SnarkWrapperStep>::load_compact_raw_crs(blob_storage);
    CompressionMode1ForWrapper::precomputae_and_store_compression_circuits(
        blob_storage,
        &context_manager,
    );
    PlonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(
        compact_raw_crs,
        blob_storage,
        &context_manager,
    );
    println!(
        "Precomputation of entire chain with fflonk took {}s",
        start.elapsed().as_secs()
    );
}
