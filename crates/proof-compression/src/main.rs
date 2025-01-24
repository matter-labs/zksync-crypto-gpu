#![feature(generic_const_exprs)]

use anyhow::Context as _;
use circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursionProof;
use clap::Parser;

use proof_compression::proof_system::PlonkSnarkVerifierCircuitProof;
use proof_compression::{
    precompute_proof_chain_with_plonk, run_proof_chain_with_plonk, BlobStorage,
    FileSystemBlobStorage, SimpleProofStorage, SnarkWrapperProof,
};
use serde_json::from_reader;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;

#[derive(Debug, Parser)]
#[command(author = "Matter Labs", version)]
struct Cli {
    pub(crate) zkos_boojum_wrapped_proof: std::path::PathBuf,
    pub(crate) zkos_boojum_wrapped_vk: std::path::PathBuf,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct L1BatchProofForL1 {
    pub aggregation_result_coords: [[u8; 32]; 4],
    pub scheduler_proof: PlonkSnarkVerifierCircuitProof,
}

/// Small tool to read the proof and VK from the zkos proof wrapped into boojum proof and VK files.
fn main() -> anyhow::Result<()> {
    let opt = Cli::parse();
    let simple_blob_storage = FileSystemBlobStorage;
    let mut simple_proof_storage = SimpleProofStorage::new();

    println!("Reading proof & VK");

    let mut file = File::open(&opt.zkos_boojum_wrapped_vk).context("Failed to open VK file")?;
    let mut vk_data = Vec::new();

    file.read_to_end(&mut vk_data)
        .context("Failed to read VK file")?;
    simple_blob_storage.write_scheduler_vk(&vk_data);

    let file = File::open(&opt.zkos_boojum_wrapped_proof).context("Failed to open proof file")?;
    let reader = BufReader::new(file);
    let inner: ZkSyncRecursionProof = from_reader(reader).context("Failed to parse proof file")?;

    precompute_proof_chain_with_plonk(&simple_blob_storage);

    let proof = run_proof_chain_with_plonk(&simple_blob_storage, inner);

    if let SnarkWrapperProof::Plonk(proof) = proof {
        let output_file_path = "final_proof.json";
        let output_file = File::create(output_file_path).context("Failed to create output file")?;
        serde_json::to_writer(output_file, &proof).context("Failed to write proof to file")?;
        println!("Proof successfully written to {}", output_file_path);

        let l1_format = L1BatchProofForL1 {
            aggregation_result_coords: [[0; 32]; 4],
            scheduler_proof: proof,
        };

        let bincode_file_path = "final_proof.bin";
        let bincode_file =
            File::create(bincode_file_path).context("Failed to create bincode file")?;
        bincode::serialize_into(bincode_file, &l1_format)
            .context("Failed to serialize proof to bincode file")?;
        println!("Proof successfully serialized to {}", bincode_file_path);
    }

    Ok(())
}
