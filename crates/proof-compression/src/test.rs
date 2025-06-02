use std::sync::Arc;

use super::*;
#[test]
fn test_proof_chain_with_fflonk() {
    let simple_blob_storage = FileSystemBlobStorage::new().load_all_resources_for_fflonk()
        .expect("Failed to load resources for fflonk");
    let simple_proof_storage = SimpleProofStorage::new();
    let proof = simple_proof_storage.get_scheduler_proof();
    run_proof_chain_with_fflonk(Arc::new(simple_blob_storage), proof)
        .expect("Failed to prove chain with fflonk");
    println!("Final fflonk snark wrapper proof has successfully generated");
}

#[test]
fn test_proof_chain_with_plonk() {
    let simple_blob_storage = FileSystemBlobStorage::new().load_all_resources_for_plonk()
        .expect("Failed to load resources for plonk");
    let simple_proof_storage = SimpleProofStorage::new();
    let proof = simple_proof_storage.get_scheduler_proof();
    run_proof_chain_with_plonk(Arc::new(simple_blob_storage), proof)
        .expect("Failed to prove chain with plonk");

    println!("Final plonk snark wrapper proof has successfully generated");
}

#[test]
fn test_precompute_compression_chain_artifacts_with_fflonk() {
    let simple_blob_storage = FileSystemBlobStorage::new();
    precompute_proof_chain_with_fflonk(Arc::new(simple_blob_storage))
        .expect("Failed to precompute compression chain artifacts with fflonk");
}

#[test]
fn test_precompute_compression_chain_artifacts_with_plonk() {
    let simple_blob_storage = FileSystemBlobStorage::new();
    precompute_proof_chain_with_plonk(Arc::new(simple_blob_storage))
        .expect("Failed to precompute compression chain artifacts with plonk");
}

#[test]
fn test_create_compact_raw_crs() {
    let blob_storage = FileSystemBlobStorage::new();
    let writer = blob_storage.write_compact_raw_crs()
        .expect("Failed to create writer for compact raw CRS");
    create_compact_raw_crs(writer);
}
