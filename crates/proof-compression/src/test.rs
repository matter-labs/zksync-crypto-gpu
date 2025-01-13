use super::*;
#[test]
fn test_proof_chain_with_fflonk() {
    let simple_blob_storage = FileSystemBlobStorage;
    let mut simple_proof_storage = SimpleProofStorage::new();
    run_proof_chain_with_fflonk(&simple_blob_storage, &mut simple_proof_storage);
    println!("Final fflonk snark wrapper proof has successfully generated");
}

#[test]
fn test_proof_chain_with_plonk() {
    let simple_blob_storage = FileSystemBlobStorage;
    let mut simple_proof_storage = SimpleProofStorage::new();
    run_proof_chain_with_plonk(&simple_blob_storage, &mut simple_proof_storage);

    println!("Final plonk snark wrapper proof has successfully generated");
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
