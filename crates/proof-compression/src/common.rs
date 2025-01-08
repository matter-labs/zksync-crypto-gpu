use super::*;
use circuit_definitions::circuit_definitions::{
    aux_layer::{
        CompressionProofsTreeHasher, CompressionProofsTreeHasherForWrapper,
        ZkSyncCompressionForWrapperCircuit, ZkSyncCompressionLayerCircuit, ZkSyncCompressionProof,
        ZkSyncCompressionProofForWrapper, ZkSyncCompressionVerificationKey,
        ZkSyncCompressionVerificationKeyForWrapper,
    },
    recursion_layer::{
        ZkSyncRecursionLayerProof, ZkSyncRecursionLayerVerificationKey, ZkSyncRecursionProof,
        ZkSyncRecursionVerificationKey,
    },
};
use fflonk::{
    FflonkSnarkVerifierCircuitProof, FflonkSnarkVerifierCircuitSetup, FflonkSnarkVerifierCircuitVK,
};
use shivini::boojum::{
    algebraic_props::{round_function::AbsorptionModeOverwrite, sponge::GoldilocksPoseidon2Sponge},
    config::{CSConfig, DevCSConfig, ProvingCSConfig, SetupCSConfig},
    cs::{
        cs_builder::new_builder,
        cs_builder_reference::CsReferenceImplementationBuilder,
        implementations::{
            proof::Proof, reference_cs::CSReferenceAssembly, setup::FinalizationHintsForProver,
            transcript::GoldilocksPoisedon2Transcript, verifier::Verifier,
        },
    },
    field::goldilocks::{GoldilocksExt2, GoldilocksField},
};

type F = GoldilocksField;
type EXT = GoldilocksExt2;
type DefaultTreeHasher = GoldilocksPoseidon2Sponge<AbsorptionModeOverwrite>;
type DefaultTranscript = GoldilocksPoisedon2Transcript;

pub type ZksyncProof = Proof<F, DefaultTreeHasher, EXT>;
pub type CompressionProofsTranscript = GoldilocksPoisedon2Transcript;

pub fn load_scheduler_proof_and_vk(
    path: &str,
) -> (ZkSyncRecursionProof, ZkSyncRecursionVerificationKey) {
    let scheduler_vk_file =
        std::fs::File::open(format!("{}/scheduler_recursive_vk.json", path)).unwrap();
    let scheduler_vk: ZkSyncRecursionLayerVerificationKey =
        serde_json::from_reader(&scheduler_vk_file).unwrap();
    let scheduler_proof_file =
        std::fs::File::open(format!("{}/scheduler_recursive_proof.json", path)).unwrap();
    let scheduler_proof: ZkSyncRecursionLayerProof =
        serde_json::from_reader(&scheduler_proof_file).unwrap();

    (scheduler_proof.into_inner(), scheduler_vk.into_inner())
}

#[derive(Copy, Clone, Debug)]
pub enum CompressionMode {
    One = 1,
    Two = 2,
    Three = 3,
    Four = 4,
    Five = 5,
}

impl CompressionMode {
    pub fn from_compression_mode(compression_mode: u8) -> Self {
        match compression_mode {
            1 => CompressionMode::One,
            2 => CompressionMode::Two,
            3 => CompressionMode::Three,
            4 => CompressionMode::Four,
            5 => CompressionMode::Five,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub struct CompressionSchedule {
    pub name: &'static str,
    pub compression_steps: Vec<CompressionMode>,
}

impl CompressionSchedule {
    pub fn name(&self) -> &'static str {
        self.name
    }
    pub fn hard() -> Self {
        CompressionSchedule {
            name: "hard",
            compression_steps: vec![
                CompressionMode::One,
                CompressionMode::Two,
                CompressionMode::Three,
                CompressionMode::Four,
            ],
        }
    }
}

pub enum CompressionInput {
    Recursion(
        Option<ZkSyncRecursionProof>,
        ZkSyncRecursionVerificationKey,
        CompressionMode,
    ),
    Compression(
        Option<ZkSyncCompressionProof>,
        ZkSyncCompressionVerificationKey,
        CompressionMode,
    ),
    CompressionWrapper(
        Option<ZkSyncCompressionProof>,
        ZkSyncCompressionVerificationKey,
        CompressionMode,
    ),
}

impl CompressionInput {
    pub fn into_compression_circuit(self) -> ZkSyncCompressionLayerCircuit {
        match self {
            CompressionInput::Recursion(proof, vk, compression_mode) => {
                assert_eq!(compression_mode as u8, 1);
                ZkSyncCompressionLayerCircuit::from_witness_and_vk(proof, vk, 1)
            }
            CompressionInput::Compression(proof, vk, compression_mode) => {
                ZkSyncCompressionLayerCircuit::from_witness_and_vk(
                    proof,
                    vk,
                    compression_mode as u8,
                )
            }
            CompressionInput::CompressionWrapper(_, _, _) => {
                unreachable!()
            }
        }
    }

    pub fn into_compression_wrapper_circuit(self) -> ZkSyncCompressionForWrapperCircuit {
        match self {
            CompressionInput::Recursion(_, _, _) => {
                unreachable!()
            }
            CompressionInput::Compression(_, _, _) => {
                unreachable!()
            }
            CompressionInput::CompressionWrapper(proof, vk, compression_mode) => {
                ZkSyncCompressionForWrapperCircuit::from_witness_and_vk(
                    proof,
                    vk,
                    compression_mode as u8,
                )
            }
        }
    }
}

pub fn save_compression_proof_and_vk_into_file(
    proof: &ZkSyncCompressionProof,
    vk: &ZkSyncCompressionVerificationKey,
    compression_mode: u8,
    path: &str,
) {
    let proof_file = std::fs::File::create(format!(
        "{}/compression_{}_proof.json",
        path, compression_mode
    ))
    .unwrap();
    serde_json::to_writer(proof_file, &proof).unwrap();
    let vk_file =
        std::fs::File::create(format!("{}/compression_{}_vk.json", path, compression_mode))
            .unwrap();
    serde_json::to_writer(vk_file, &vk).unwrap();
}

pub fn save_compression_wrapper_proof_and_vk_into_file(
    proof: &ZkSyncCompressionProofForWrapper,
    vk: &ZkSyncCompressionVerificationKeyForWrapper,
    compression_mode: u8,
    path: &str,
) {
    let proof_file = std::fs::File::create(format!(
        "{}/compression_wrapper_{}_proof.json",
        path, compression_mode
    ))
    .unwrap();
    serde_json::to_writer(proof_file, &proof).unwrap();
    let vk_file = std::fs::File::create(format!(
        "{}/compression_wrapper_{}_vk.json",
        path, compression_mode
    ))
    .unwrap();
    serde_json::to_writer(vk_file, &vk).unwrap();
}

pub fn load_compression_wrapper_proof_and_vk_from_file(
    blob_path: &str,
    compression_mode: u8,
) -> (
    ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
) {
    let proof_file = std::fs::File::open(format!(
        "{}/compression_wrapper_{}_proof.json",
        blob_path, compression_mode
    ))
    .unwrap();
    let proof = serde_json::from_reader(proof_file).unwrap();
    let vk_file = std::fs::File::open(format!(
        "{}/compression_wrapper_{}_vk.json",
        blob_path, compression_mode
    ))
    .unwrap();
    let vk = serde_json::from_reader(vk_file).unwrap();

    (proof, vk)
}

pub fn verify_compression_layer_circuit<CF: ProofCompressionFunction>(
    _circuit: CompressionLayerCircuit<CF>,
    proof: &ZkSyncCompressionProof,
    vk: &ZkSyncCompressionVerificationKey,
    verifier: Verifier<F, EXT>,
) -> bool {
    verifier.verify::<CompressionProofsTreeHasher, CompressionProofsTranscript, CF::ThisLayerPoW>(
        (),
        vk,
        proof,
    )
}

pub fn verify_compression_wrapper_circuit<CF: ProofCompressionFunction>(
    _circuit: CompressionLayerCircuit<CF>,
    proof: &ZkSyncCompressionProofForWrapper,
    vk: &ZkSyncCompressionVerificationKeyForWrapper,
    verifier: Verifier<F, EXT>,
) -> bool {
    verifier.verify::<CompressionProofsTreeHasherForWrapper, CompressionTranscriptForWrapper, CF::ThisLayerPoW>(
        (),
        vk,
        proof,
    )
}

pub fn save_fflonk_proof_and_vk_into_file(
    proof: &FflonkSnarkVerifierCircuitProof,
    vk: &FflonkSnarkVerifierCircuitVK,
    output_blob_path: &str,
) {
    let proof_file_path = format!("{}/final_proof.json", output_blob_path);
    let proof_file = std::fs::File::create(&proof_file_path).unwrap();
    serde_json::to_writer(proof_file, &proof).unwrap();
    println!("proof saved at {proof_file_path}");
    let vk_file_path = format!("{}/final_vk.json", output_blob_path);
    let vk_file = std::fs::File::create(&vk_file_path).unwrap();
    serde_json::to_writer(vk_file, &vk).unwrap();
    println!("vk saved at {vk_file_path}");
}

pub fn save_fflonk_setup_and_vk_into_file(
    setup: &FflonkSnarkVerifierCircuitSetup,
    vk: &FflonkSnarkVerifierCircuitVK,
    output_blob_path: &str,
) {
    let setup_file_path = format!("{}/fflonk_snark_setup.json", output_blob_path);
    let setup_file = std::fs::File::create(&setup_file_path).unwrap();
    setup.write(&setup_file).unwrap();
    println!("fflonk precomputed setup saved into file at {setup_file_path}");
    let vk_file_path = format!("{}/fflonk_snark_setup.json", output_blob_path);
    let vk_file = std::fs::File::create(&vk_file_path).unwrap();
    vk.write(&vk_file).unwrap();
    println!("fflonk VK saved into file at {vk_file_path}");
}

pub fn load_fflonk_setup_and_vk_from_file(
    output_blob_path: &str,
) -> (
    FflonkSnarkVerifierCircuitSetup,
    FflonkSnarkVerifierCircuitVK,
) {
    let setup_file_path = format!("{}/fflonk_snark_setup.json", output_blob_path);
    println!("reading fflonk precomputed setup from file at {setup_file_path}");
    let setup_file = std::fs::File::open(&setup_file_path).unwrap();
    let setup = FflonkSnarkVerifierCircuitSetup::read(&setup_file).unwrap();
    let vk_file_path = format!("{}/fflonk_snark_setup.json", output_blob_path);
    println!("reading fflonk VK saved from file at {vk_file_path}");
    let vk_file = std::fs::File::open(&vk_file_path).unwrap();
    let vk = FflonkSnarkVerifierCircuitVK::read(&vk_file).unwrap();

    (setup, vk)
}

pub fn synthesize_circuit_for_setup<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
) -> (
    FinalizationHintsForProver,
    CSReferenceAssembly<F, F, SetupCSConfig>,
) {
    let geometry = circuit.geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GoldilocksField, F, SetupCSConfig>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GoldilocksField>(builder_impl);

    let builder = circuit.configure_builder_proxy(builder);
    let mut cs = builder.build(num_vars.unwrap());
    circuit.add_tables(&mut cs);
    circuit.synthesize_into_cs(&mut cs);
    let (_domain_size, finalization_hint) = cs.pad_and_shrink();
    let cs = cs.into_assembly::<std::alloc::Global>();

    (finalization_hint, cs)
}

pub fn synthesize_circuit_for_proving<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
    finalization_hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    let geometry = circuit.geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GoldilocksField, F, ProvingCSConfig>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GoldilocksField>(builder_impl);

    let builder = circuit.configure_builder_proxy(builder);
    let mut cs = builder.build(num_vars.unwrap());
    circuit.add_tables(&mut cs);
    circuit.synthesize_into_cs(&mut cs);
    let _ = cs.pad_and_shrink_using_hint(&finalization_hint);
    let cs = cs.into_assembly::<std::alloc::Global>();

    cs
}
pub fn synthesize_circuit_for_dev<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
) -> CSReferenceAssembly<F, F, DevCSConfig> {
    let geometry = circuit.geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GoldilocksField, F, DevCSConfig>::new(
        geometry,
        max_trace_len.unwrap(),
    );
    let builder = new_builder::<_, GoldilocksField>(builder_impl);

    let builder = circuit.configure_builder_proxy(builder);
    let mut cs = builder.build(num_vars.unwrap());
    circuit.add_tables(&mut cs);
    circuit.synthesize_into_cs(&mut cs);
    let _ = cs.pad_and_shrink();
    let cs = cs.into_assembly::<std::alloc::Global>();

    cs
}
