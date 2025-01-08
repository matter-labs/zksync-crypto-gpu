use super::*;

pub enum SnarkWrapper {
    Plonk,
    FFfonk,
}
pub enum SnarkWrapperProof {
    Plonk(PlonkSnarkVerifierCircuitProof),
    FFfonk(FflonkSnarkVerifierCircuitProof),
}
pub fn wrap_proof<BS>(
    input_proof: SchedulerProof,
    snark_wrapper: SnarkWrapper,
    blob_storage: BS,
) -> SnarkWrapperProof
where
    BS: BlobStorage,
{
    match snark_wrapper {
        SnarkWrapper::Plonk => {
            let proof = run_step_chain_with_plonk(input_proof, blob_storage);
            SnarkWrapperProof::Plonk(proof)
        }
        SnarkWrapper::FFfonk => {
            let proof = run_step_chain_with_fflonk(input_proof, blob_storage);
            SnarkWrapperProof::FFfonk(proof)
        }
    }
}

pub(crate) fn run_step_chain_with_fflonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: BS,
) -> FflonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let artifact_loader = SimpleArtifactLoader::init(blob_storage);
    let context_initializor = SimpelContextInitializor::new();

    let next_proof = CompressionMode1::prove_compression_step::<_, SimpelContextInitializor>(
        input_proof,
        &artifact_loader,
    );
    let next_proof = CompressionMode2::prove_compression_step::<_, SimpelContextInitializor>(
        next_proof,
        &artifact_loader,
    );
    let next_proof = CompressionMode3::prove_compression_step::<_, SimpelContextInitializor>(
        next_proof,
        &artifact_loader,
    );
    let next_proof = CompressionMode4::prove_compression_step::<_, SimpelContextInitializor>(
        next_proof,
        &artifact_loader,
    );
    let next_proof = CompressionMode5ForWrapper::prove_compression_step::<
        _,
        SimpelContextInitializor,
    >(next_proof, &artifact_loader);
    let final_proof = FflonkSnarkWrapper::prove_snark_wrapper_step::<_, SimpelContextInitializor>(
        next_proof,
        &artifact_loader,
    );

    final_proof
}

pub(crate) fn run_step_chain_with_plonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: BS,
) -> PlonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let artifact_loader = SimpleArtifactLoader::init(blob_storage);
    let next_proof = CompressionMode1ForWrapper::prove_compression_step::<
        _,
        SimpelContextInitializor,
    >(input_proof, &artifact_loader);

    let final_proof = PlonkSnarkWrapper::prove_snark_wrapper_step::<_, SimpelContextInitializor>(
        next_proof,
        &artifact_loader,
    );

    final_proof
}
