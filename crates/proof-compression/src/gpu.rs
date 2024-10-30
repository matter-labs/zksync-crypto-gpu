use super::*;
use boojum::config::{ProvingCSConfig, SetupCSConfig};
use boojum::cs::implementations::prover::ProofConfig;
use boojum::worker::Worker;
use shivini::circuit_definitions::boojum;
use shivini::circuit_definitions::circuit_definitions::{
    aux_layer::{
        compression::{CompressionLayerCircuit, ProofCompressionFunction},
        compression_modes::{CompressionTranscriptForWrapper, CompressionTreeHasherForWrapper},
        CompressionProofsTreeHasher, ZkSyncCompressionForWrapperCircuit,
        ZkSyncCompressionLayerCircuit, ZkSyncCompressionProof, ZkSyncCompressionProofForWrapper,
        ZkSyncCompressionVerificationKey, ZkSyncCompressionVerificationKeyForWrapper,
    },
    recursion_layer::{ZkSyncRecursionProof, ZkSyncRecursionVerificationKey},
};
use shivini::gpu_proof_config::GpuProofConfig;
use shivini::synthesis_utils::synthesize_compression_circuit;
use shivini::{
    gpu_prove_from_external_witness_data_with_cache_strategy, CacheStrategy,
    CommitmentCacheStrategy, PolynomialsCacheStrategy, ProverContext, ProverContextConfig,
};
use std::alloc::Global;

use ::fflonk::*;

#[test]
fn run_proof_compression_by_schedule() {
    let path = if let Ok(path) = std::env::var("BLOB_PATH") {
        path.to_string()
    } else {
        "./data".to_string()
    };
    let (scheduler_proof, scheduler_vk) = load_scheduler_proof_and_vk(&path);
    process_steps(
        scheduler_proof,
        scheduler_vk,
        CompressionSchedule::hard(),
        &path,
    );
}

pub fn process_steps(
    proof: ZkSyncRecursionProof,
    vk: ZkSyncRecursionVerificationKey,
    schedule: CompressionSchedule,
    path: &str,
) {
    let worker = Worker::new();
    let mut input = CompressionInput::Recursion(Some(proof), vk, CompressionMode::One);

    dbg!(&schedule);
    let CompressionSchedule {
        name: compression_schedule_name,
        compression_steps,
    } = schedule;

    let last_compression_wrapping_mode =
        CompressionMode::from_compression_mode(*compression_steps.last().unwrap() as u8 + 1);
    dbg!(&last_compression_wrapping_mode);

    /*
        This illustrates how compression enforced for the "hardest" strategy

           input                       compression     verifier          output        compression wrapper
       _____________________________   ____________    ___________     __________      ___________________
       scheduler       proof   vk          1           scheduler   ->  compressed1         compressed2
       compressed1     proof   vk          2           compressed1 ->  compressed2         compressed3
       compressed2     proof   vk          3           compressed2 ->  compressed3         compressed4
       compressed3     proof   vk          4           compressed3 ->  compressed4         compressed5


       compressed5     proof   vk          -       compression wrapper5       ->  fflonk proof
    */

    let num_compression_steps = compression_steps.len();
    let mut compression_modes_iter = compression_steps.into_iter();
    for step_idx in 0..num_compression_steps {
        let compression_mode = compression_modes_iter.next().unwrap();
        let proof_file_path = format!("{}/compression_{}_proof.json", path, compression_mode as u8);
        let proof_file_path = std::path::Path::new(&proof_file_path);
        let vk_file_path = format!("{}/compression_{}_vk.json", path, compression_mode as u8);
        let vk_file_path = std::path::Path::new(&vk_file_path);
        if proof_file_path.exists() && vk_file_path.exists() {
            println!(
                "Compression {compression_schedule_name}/{} proof and vk already exist ignoring",
                compression_mode as u8
            );
            let proof_file = std::fs::File::open(proof_file_path).unwrap();
            let input_proof = serde_json::from_reader(&proof_file).unwrap();
            let vk_file = std::fs::File::open(vk_file_path).unwrap();
            let input_vk = serde_json::from_reader(&vk_file).unwrap();
            if step_idx + 1 == num_compression_steps {
                input = CompressionInput::CompressionWrapper(
                    input_proof,
                    input_vk,
                    last_compression_wrapping_mode,
                )
            } else {
                input = CompressionInput::Compression(
                    input_proof,
                    input_vk,
                    CompressionMode::from_compression_mode(compression_mode as u8 + 1),
                )
            }

            continue;
        }

        let compression_circuit = input.into_compression_circuit();
        let circuit_type = compression_circuit.numeric_circuit_type();
        println!(
            "Proving compression {compression_schedule_name}/{}",
            compression_mode as u8
        );
        let (output_proof, output_vk) =
            prove_compression_layer_circuit(compression_circuit.clone(), &worker);
        println!(
            "Proof for compression {compression_schedule_name}/{} is generated!",
            compression_mode as u8
        );

        save_compression_proof_and_vk_into_file(&output_proof, &output_vk, circuit_type, path);

        if step_idx + 1 == num_compression_steps {
            input = CompressionInput::CompressionWrapper(
                Some(output_proof),
                output_vk,
                last_compression_wrapping_mode,
            );
        } else {
            input = CompressionInput::Compression(
                Some(output_proof),
                output_vk,
                CompressionMode::from_compression_mode(compression_mode as u8 + 1),
            );
        }
    }

    // last wrapping step
    let proof_file_path = format!(
        "{}/compression_wrapper_{}_proof.json",
        path, last_compression_wrapping_mode as u8
    );
    let compression_wrapper_proof_file_path = std::path::Path::new(&proof_file_path);
    let vk_file_path = format!(
        "{}/compression_wrapper_{}_vk.json",
        path, last_compression_wrapping_mode as u8
    );
    let compression_wrapper_vk_file_path = std::path::Path::new(&vk_file_path);
    println!(
        "Compression for wrapper level {}",
        last_compression_wrapping_mode as u8
    );
    if compression_wrapper_proof_file_path.exists() && compression_wrapper_vk_file_path.exists() {
        println!(
            "Compression {compression_schedule_name}/{} for wrapper proof and vk already exist ignoring",
            last_compression_wrapping_mode as u8
        );
    } else {
        println!(
            "Proving compression {compression_schedule_name}/{} for wrapper",
            last_compression_wrapping_mode as u8
        );
        let compression_circuit = input.into_compression_wrapper_circuit();
        let (compression_wrapper_output_proof, compression_wrapper_output_vk) =
            prove_compression_wrapper_circuit(compression_circuit, &worker);
        println!(
            "Proof for compression wrapper {compression_schedule_name}/{} is generated!",
            last_compression_wrapping_mode as u8
        );
        save_compression_wrapper_proof_and_vk_into_file(
            &compression_wrapper_output_proof,
            &compression_wrapper_output_vk,
            last_compression_wrapping_mode as u8,
            path,
        );
        println!(
            "Compression wrapper proof and vk for {compression_schedule_name}/{} saved",
            last_compression_wrapping_mode as u8
        );
    }
    // final wrapping step
    let final_proof_file_path = format!("{}/final_proof.json", path);
    let final_proof_file_path = std::path::Path::new(&final_proof_file_path);
    let final_vk_file_path = format!("{}/final_vk.json", path,);
    let final_vk_file_path = std::path::Path::new(&final_vk_file_path);

    if final_proof_file_path.exists() == false || final_vk_file_path.exists() == false {
        let (compression_wrapper_proof, compression_wrapper_vk) =
            load_compression_wrapper_proof_and_vk_from_file(
                path,
                last_compression_wrapping_mode as u8,
            );
        let wrapper_circuit = init_snark_wrapper_circuit_from_inputs(
            last_compression_wrapping_mode as u8,
            compression_wrapper_proof,
            compression_wrapper_vk,
        );

        let (final_proof, final_vk) =
            ::fflonk::gpu_prove_fflonk_snark_verifier_circuit_single_shot(&wrapper_circuit);
        let final_proof_file = std::fs::File::create(final_proof_file_path).unwrap();
        serde_json::to_writer(&final_proof_file, &final_proof).unwrap();
        println!(
            "final snark proof saved into {}",
            final_proof_file_path.to_string_lossy()
        );
        save_fflonk_proof_and_vk_into_file(&final_proof, &final_vk, &path);
    } else {
        println!(
            "final proof already exists {}",
            final_proof_file_path.to_string_lossy()
        );
    }
}

pub fn prove_compression_layer_circuit(
    circuit: ZkSyncCompressionLayerCircuit,
    worker: &Worker,
) -> (ZkSyncCompressionProof, ZkSyncCompressionVerificationKey) {
    let proof_config = circuit.proof_config_for_compression_step();
    let verifier_builder = circuit.into_dyn_verifier_builder();
    let verifier = verifier_builder.create_verifier();
    let gpu_proof_config = GpuProofConfig::from_compression_layer_circuit(&circuit);

    let (proof, vk, is_proof_valid) = match circuit {
        ZkSyncCompressionLayerCircuit::CompressionMode1Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode2Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode3Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode4Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode5Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_layer_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
    };
    if !is_proof_valid {
        panic!("Proof is invalid");
    }

    (proof, vk)
}

pub fn prove_compression_wrapper_circuit(
    circuit: ZkSyncCompressionForWrapperCircuit,
    worker: &Worker,
) -> (
    ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
) {
    let proof_config = circuit.proof_config_for_compression_step();
    let verifier_builder = circuit.into_dyn_verifier_builder();
    let verifier = verifier_builder.create_verifier();
    let gpu_proof_config = GpuProofConfig::from_compression_wrapper_circuit(&circuit);

    let (proof, vk, is_proof_valid) = match circuit {
        ZkSyncCompressionForWrapperCircuit::CompressionMode1Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode2Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode3Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode4Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode5Circuit(inner) => {
            let (proof, vk) = inner_prove_compression_wrapper_circuit(
                inner.clone(),
                proof_config,
                gpu_proof_config,
                worker,
            );
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
    };
    if !is_proof_valid {
        panic!("Proof is invalid");
    }

    (proof, vk)
}

pub fn inner_prove_compression_layer_circuit<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
    proof_cfg: ProofConfig,
    gpu_cfg: GpuProofConfig,
    worker: &Worker,
) -> (ZkSyncCompressionProof, ZkSyncCompressionVerificationKey) {
    let (setup_cs, finalization_hint) =
        synthesize_compression_circuit::<_, SetupCSConfig, Global>(circuit.clone(), true, None);
    let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
        &worker,
        proof_cfg.fri_lde_factor,
        proof_cfg.merkle_tree_cap_size,
    );
    let (gpu_setup, vk) = shivini::cs::gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
        CompressionProofsTreeHasher,
        _,
    >(setup_base, vk_params, vars_hint, wits_hint, worker)
    .unwrap();
    let (proving_cs, _) = synthesize_compression_circuit::<_, ProvingCSConfig, Global>(
        circuit.clone(),
        true,
        finalization_hint.as_ref(),
    );
    let witness = proving_cs.witness.as_ref().unwrap();
    let domain_size = vk.fixed_parameters.domain_size as usize;
    let config = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
    let cache_strategy = CacheStrategy {
        setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        commitment: CommitmentCacheStrategy::CacheCosetCaps,
    };
    let gpu_proof = gpu_prove_from_external_witness_data_with_cache_strategy::<
        CompressionProofsTranscript,
        CompressionProofsTreeHasher,
        CF::ThisLayerPoW,
        Global,
    >(
        &gpu_cfg,
        witness,
        proof_cfg.clone(),
        &gpu_setup,
        &vk,
        (),
        worker,
        cache_strategy,
    )
    .expect("gpu proof");
    drop(ctx);
    let proof = gpu_proof.into();
    (proof, vk)
}

pub fn inner_prove_compression_wrapper_circuit<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
    proof_cfg: ProofConfig,
    gpu_cfg: GpuProofConfig,
    worker: &Worker,
) -> (
    ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
) {
    let (setup_cs, finalization_hint) =
        synthesize_compression_circuit::<_, SetupCSConfig, Global>(circuit.clone(), true, None);
    let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
        &worker,
        proof_cfg.fri_lde_factor,
        proof_cfg.merkle_tree_cap_size,
    );
    let (gpu_setup, vk) = shivini::cs::gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
        CompressionTreeHasherForWrapper,
        _,
    >(setup_base, vk_params, vars_hint, wits_hint, worker)
    .unwrap();
    let (proving_cs, _) = synthesize_compression_circuit::<_, ProvingCSConfig, Global>(
        circuit,
        true,
        finalization_hint.as_ref(),
    );
    let witness = proving_cs.witness.as_ref().unwrap();
    let domain_size = vk.fixed_parameters.domain_size as usize;
    let config = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
    let cache_strategy = CacheStrategy {
        setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
        commitment: CommitmentCacheStrategy::CacheCosetCaps,
    };
    let gpu_proof = gpu_prove_from_external_witness_data_with_cache_strategy::<
        CompressionTranscriptForWrapper,
        CompressionTreeHasherForWrapper,
        CF::ThisLayerPoW,
        Global,
    >(
        &gpu_cfg,
        witness,
        proof_cfg.clone(),
        &gpu_setup,
        &vk,
        (),
        worker,
        cache_strategy,
    )
    .expect("gpu proof");
    drop(ctx);
    (gpu_proof.into(), vk)
}
