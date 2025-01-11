use super::*;
use circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursionLayerProof;

#[test]
fn test_proof_chain_with_fflonk() {
    let scheduler_proof_file =
        std::fs::File::open("./data/scheduler_recursive_proof.json").unwrap();
    let scheduler_proof: ZkSyncRecursionLayerProof =
        serde_json::from_reader(&scheduler_proof_file).unwrap();
    let scheduler_proof = scheduler_proof.into_inner();

    let simple_blob_storage = FileSystemBlobStorage;
    let proof = run_proof_chain_with_fflonk(scheduler_proof, &simple_blob_storage);

    let proof_file_path = format!("./data/final_fflonk_proof.json");
    let proof_file = std::fs::File::create(&proof_file_path).unwrap();
    serde_json::to_writer(proof_file, &proof).unwrap();
    println!("Final fflonk snark wrapper roof saved at {proof_file_path}");
}

#[test]
fn test_proof_chain_with_plonk() {
    let scheduler_proof_file =
        std::fs::File::open("./data/scheduler_recursive_proof.json").unwrap();
    let scheduler_proof: ZkSyncRecursionLayerProof =
        serde_json::from_reader(&scheduler_proof_file).unwrap();
    let scheduler_proof = scheduler_proof.into_inner();

    let simple_blob_storage = FileSystemBlobStorage;
    let proof = run_proof_chain_with_plonk(scheduler_proof, &simple_blob_storage);

    let proof_file_path = format!("./data/final_plonk_proof.json");
    let proof_file = std::fs::File::create(&proof_file_path).unwrap();
    serde_json::to_writer(proof_file, &proof).unwrap();
    println!("Final plonk snark wrapper roof saved at {proof_file_path}");
}

#[test]
fn test_precompute_compression_chain_artifacts_with_fflonk() {
    let simple_blob_storage = FileSystemBlobStorage;
    precompute_proof_chain_with_fflonk(&simple_blob_storage);
}

#[test]
fn test_precompute_compression_chain_artifacts_with_plonk() {
    let simple_blob_storage = FileSystemBlobStorage;
    precompute_proof_chain_with_plonk(&simple_blob_storage);
}

#[test]
fn test_create_compact_raw_crs() {
    let blob_storage = FileSystemBlobStorage;
    let writer = blob_storage.write_compact_raw_crs();
    create_compact_raw_crs(writer);
}
