use super::*;

pub enum SnarkWrapper {
    Plonk,
    FFfonk,
}
pub enum SnarkWrapperProof {
    Plonk(PlonkSnarkVerifierCircuitProof),
    FFfonk(FflonkSnarkVerifierCircuitProof),
}

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

pub(crate) fn run_proof_chain_with_fflonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: &BS,
) -> FflonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let context_manager = SimpleContextManager::new();
    let start = std::time::Instant::now();
    let snark_context_config =
        context_manager.initialize_snark_context_config::<FflonkSnarkWrapper>();
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
        snark_context_config,
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

pub(crate) fn precompute_proof_chain_with_fflonk<BS>(blob_storage: &BS)
where
    BS: BlobStorageExt,
{
    let context_manager = SimpleContextManager::new();
    let snark_context_config =
        context_manager.initialize_snark_context_config::<FflonkSnarkWrapper>();
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
        snark_context_config,
        blob_storage,
        &context_manager,
    );
    println!(
        "Precomputation of entire chain took {}s",
        start.elapsed().as_secs()
    );
}

pub(crate) fn run_proof_chain_with_plonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: &BS,
) -> PlonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let context_manager = SimpleContextManager::new();
    let snark_context_config =
        context_manager.initialize_snark_context_config::<PlonkSnarkWrapper>();
    let start = std::time::Instant::now();
    let next_proof = CompressionMode1ForWrapper::prove_compression_step(
        input_proof,
        blob_storage,
        &context_manager,
    );

    let final_proof = PlonkSnarkWrapper::prove_snark_wrapper_step(
        snark_context_config,
        next_proof,
        blob_storage,
        &context_manager,
    );
    println!(
        "Entire compression chain took {}s",
        start.elapsed().as_secs()
    );
    final_proof
}

pub(crate) fn precompute_proof_chain_with_plonk<BS>(blob_storage: &BS)
where
    BS: BlobStorageExt,
{
    let context_manager = SimpleContextManager::new();
    let snark_context_config =
        context_manager.initialize_snark_context_config::<PlonkSnarkWrapper>();
    CompressionMode1ForWrapper::precomputae_and_store_compression_circuits(
        blob_storage,
        &context_manager,
    );
    PlonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(
        snark_context_config,
        blob_storage,
        &context_manager,
    );
    println!("All steps in this approach precomputed and saved into blob storage");
}
