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
    let final_proof = FflonkSnarkWrapper::prove_snark_wrapper_step::<_, SimpleContextManager>(
        next_proof,
        blob_storage,
        &context_manager,
    );

    final_proof
}

pub(crate) fn precompute_proof_chain_with_fflonk<BS>(blob_storage: &BS)
where
    BS: BlobStorageExt,
{
    let context_manager = SimpleContextManager::new();
    println!("Precomputing step 1");
    CompressionMode1::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    println!("Precomputing step 2");
    CompressionMode2::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    println!("Precomputing step 3");
    CompressionMode3::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    println!("Precomputing step 4");
    CompressionMode4::precomputae_and_store_compression_circuits(blob_storage, &context_manager);
    println!("Precomputing step 5");
    CompressionMode5ForWrapper::precomputae_and_store_compression_circuits(
        blob_storage,
        &context_manager,
    );
    println!("Precomputing fflonk");
    FflonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(blob_storage, &context_manager);
    println!("All steps in this approach precomputed and saved into blob storage");
}

pub(crate) fn run_proof_chain_with_plonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: &BS,
) -> PlonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let context_manager = SimpleContextManager::new();

    let next_proof = CompressionMode1ForWrapper::prove_compression_step(
        input_proof,
        blob_storage,
        &context_manager,
    );

    let final_proof =
        PlonkSnarkWrapper::prove_snark_wrapper_step(next_proof, blob_storage, &context_manager);

    final_proof
}

pub(crate) fn precompute_proof_chain_with_plonk<BS>(blob_storage: &BS)
where
    BS: BlobStorageExt,
{
    let context_manager = SimpleContextManager::new();
    CompressionMode1ForWrapper::precomputae_and_store_compression_circuits(
        blob_storage,
        &context_manager,
    );
    PlonkSnarkWrapper::precompute_and_store_snark_wrapper_circuit(blob_storage, &context_manager);
    println!("All steps in this approach precomputed and saved into blob storage");
}
