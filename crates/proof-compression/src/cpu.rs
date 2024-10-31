use super::*;
use bellman::plonk::better_better_cs::cs::ProvingAssembly;
use bellman::plonk::better_better_cs::{
    cs::{Circuit, PlonkCsWidth3Params, SetupAssembly, TrivialAssembly},
    gates::naive_main_gate::NaiveMainGate,
};
use circuit_definitions::circuit_definitions::recursion_layer::{
    ZkSyncRecursionProof, ZkSyncRecursionVerificationKey,
};
use circuit_definitions::{
    circuit_definitions::aux_layer::{
        compression::{CompressionLayerCircuit, ProofCompressionFunction},
        ZkSyncCompressionForWrapperCircuit, ZkSyncCompressionLayerCircuit, ZkSyncCompressionProof,
        ZkSyncCompressionProofForWrapper, ZkSyncCompressionVerificationKey,
        ZkSyncCompressionVerificationKeyForWrapper,
    },
    snark_wrapper::franklin_crypto::bellman::plonk::commitments::transcript::keccak_transcript::RollingKeccakTranscript,
};

use franklin_crypto::boojum::{
    config::{CSConfig, DevCSConfig},
    cs::{
        cs_builder::new_builder,
        cs_builder_reference::CsReferenceImplementationBuilder,
        implementations::{
            prover::ProofConfig, reference_cs::CSReferenceAssembly,
            setup::FinalizationHintsForProver,
        },
    },
    field::goldilocks::{GoldilocksExt2, GoldilocksField},
};

use franklin_crypto::boojum::cs::implementations::proof::Proof as BoojumProof;
use franklin_crypto::boojum::cs::implementations::verifier::VerificationKey as BoojumVK;
use franklin_crypto::boojum::worker::Worker as BoojumWorker;

pub const L1_VERIFIER_DOMAIN_SIZE_LOG: usize = 23;
pub const MAX_COMBINED_DEGREE_FACTOR: usize = 9;
type F = GoldilocksField;
type EXT = GoldilocksExt2;

pub fn precompute_and_save_setup_for_fflonk_snark_circuit(
    circuit: &FflonkSnarkVerifierCircuit,
    worker: &Worker,
    output_blob_path: &str,
) {
    let compression_wrapper_mode = circuit.wrapper_function.numeric_circuit_type();
    println!("Compression mode: {compression_wrapper_mode}");
    let mut setup_assembly = SetupAssembly::<Bn256, PlonkCsWidth3Params, NaiveMainGate>::new();
    circuit.synthesize(&mut setup_assembly).expect("must work");
    assert!(setup_assembly.is_satisfied());
    setup_assembly.finalize();
    println!("Finalized assembly contains {} gates", setup_assembly.n());

    let domain_size = setup_assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);

    let mon_crs = fflonk::init_crs(&worker, domain_size);
    let setup: FflonkSnarkVerifierCircuitSetup =
        FflonkSetup::create_setup(&setup_assembly, &worker, &mon_crs).expect("fflonk setup");
    let vk = FflonkVerificationKey::from_setup(&setup, &mon_crs).unwrap();

    save_fflonk_setup_and_vk_into_file(&setup, &vk, output_blob_path);
}

pub fn prove_fflonk_snark_verifier_circuit_with_precomputation(
    circuit: &FflonkSnarkVerifierCircuit,
    precomputed_setup: &FflonkSnarkVerifierCircuitSetup,
    vk: &FflonkSnarkVerifierCircuitVK,
    worker: &Worker,
) -> FflonkSnarkVerifierCircuitProof {
    let compression_wrapper_mode = circuit.wrapper_function.numeric_circuit_type();
    let mut assembly = ProvingAssembly::<Bn256, PlonkCsWidth3Params, NaiveMainGate>::new();
    circuit.synthesize(&mut assembly).expect("must work");
    assert!(assembly.is_satisfied());
    assembly.finalize();

    let domain_size = assembly.n() + 1;
    println!(
        "Trace log length {} for compression mode {}",
        domain_size.trailing_zeros(),
        compression_wrapper_mode
    );
    assert!(domain_size.is_power_of_two());

    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);

    let mon_crs = fflonk::init_crs(&worker, domain_size);

    let proof = fflonk::fflonk_cpu::prover::create_proof::<
        _,
        FflonkSnarkVerifierCircuit,
        _,
        _,
        _,
        RollingKeccakTranscript<Fr>,
    >(&assembly, &worker, &precomputed_setup, &mon_crs, None)
    .expect("proof");
    let valid =
        fflonk::fflonk_cpu::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");

    proof
}

pub fn prove_fflonk_snark_verifier_circuit_single_shot(
    circuit: &FflonkSnarkVerifierCircuit,
    worker: &Worker,
) -> (
    FflonkSnarkVerifierCircuitProof,
    FflonkSnarkVerifierCircuitVK,
) {
    let compression_wrapper_mode = circuit.wrapper_function.numeric_circuit_type();
    let mut assembly = TrivialAssembly::<Bn256, PlonkCsWidth3Params, NaiveMainGate>::new();
    circuit.synthesize(&mut assembly).expect("must work");
    assert!(assembly.is_satisfied());
    assembly.finalize();
    let domain_size = assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);
    println!(
        "Trace log length {} for compression mode {}",
        domain_size.trailing_zeros(),
        compression_wrapper_mode
    );

    let max_combined_degree = fflonk::fflonk::compute_max_combined_degree_from_assembly::<
        _,
        _,
        _,
        _,
        FflonkSnarkVerifierCircuit,
    >(&assembly);
    println!("Max degree is {}", max_combined_degree);
    let mon_crs = fflonk::init_crs(&worker, domain_size);
    let setup = FflonkSetup::create_setup(&assembly, &worker, &mon_crs).expect("setup");
    let vk = FflonkVerificationKey::from_setup(&setup, &mon_crs).unwrap();

    let proof = fflonk::fflonk_cpu::prover::create_proof::<
        _,
        FflonkSnarkVerifierCircuit,
        _,
        _,
        _,
        RollingKeccakTranscript<Fr>,
    >(&assembly, &worker, &setup, &mon_crs, None)
    .expect("proof");
    let valid =
        fflonk::fflonk_cpu::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");

    (proof, vk)
}

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
    let worker = BoojumWorker::new();
    let mut input = CompressionInput::Recursion(Some(proof), vk, CompressionMode::One);

    dbg!(&schedule);
    let CompressionSchedule {
        name: compression_schedule_name,
        compression_steps,
    } = schedule;

    let last_compression_wrapping_mode =
        CompressionMode::from_compression_mode(compression_steps.last().unwrap().clone() as u8 + 1);
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
        let proof_file_path = format!("{path}/compression_{}_proof.json", compression_mode as u8);
        let proof_file_path = std::path::Path::new(&proof_file_path);
        let vk_file_path = format!("{path}/compression_{}_vk.json", compression_mode as u8);
        let vk_file_path = std::path::Path::new(&vk_file_path);
        if proof_file_path.exists() && vk_file_path.exists() {
            println!(
                "Compression {compression_schedule_name}/{} proof and vk already exist ignoring",
                compression_mode as u8
            );
            let proof_file = std::fs::File::open(proof_file_path).unwrap();
            let proof = serde_json::from_reader(&proof_file).unwrap();
            let vk_file = std::fs::File::open(vk_file_path).unwrap();
            let vk = serde_json::from_reader(&vk_file).unwrap();
            if step_idx + 1 == num_compression_steps {
                input =
                    CompressionInput::CompressionWrapper(proof, vk, last_compression_wrapping_mode)
            } else {
                input = CompressionInput::Compression(
                    proof,
                    vk,
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
        let (proof, vk) = inner_prove_compression_layer_circuit(compression_circuit, &worker);
        println!(
            "Proof for compression {compression_schedule_name}/{} is generated!",
            compression_mode as u8
        );

        save_compression_proof_and_vk_into_file(&proof, &vk, compression_mode as u8, path);
        if step_idx + 1 == num_compression_steps {
            input = CompressionInput::CompressionWrapper(
                Some(proof),
                vk,
                last_compression_wrapping_mode,
            );
        } else {
            input = CompressionInput::Compression(
                Some(proof),
                vk,
                CompressionMode::from_compression_mode(compression_mode as u8 + 1),
            );
        }
    }

    // last wrapping step
    let proof_file_path = format!(
        "{path}/compression_wrapper_{}_proof.json",
        last_compression_wrapping_mode as u8
    );
    let proof_file_path = std::path::Path::new(&proof_file_path);
    let vk_file_path = format!(
        "{path}/compression_wrapper_{}_vk.json",
        last_compression_wrapping_mode as u8
    );
    let vk_file_path = std::path::Path::new(&vk_file_path);
    println!(
        "Compression for wrapper level {}",
        last_compression_wrapping_mode as u8
    );
    if proof_file_path.exists() && vk_file_path.exists() {
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
        let (proof, vk) = inner_prove_compression_wrapper_circuit(compression_circuit, &worker);
        println!(
            "Proof for compression wrapper {compression_schedule_name}/{} is generated!",
            last_compression_wrapping_mode as u8
        );
        save_compression_wrapper_proof_and_vk_into_file(
            &proof,
            &vk,
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
        let wrapper_circuit = fflonk::init_snark_wrapper_circuit_from_inputs(
            last_compression_wrapping_mode as u8,
            compression_wrapper_proof,
            compression_wrapper_vk,
        );

        let (final_proof, final_vk) =
            prove_fflonk_snark_verifier_circuit_single_shot(&wrapper_circuit, &Worker::new());
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

pub fn inner_prove_compression_layer_circuit(
    circuit: ZkSyncCompressionLayerCircuit,
    worker: &BoojumWorker,
) -> (ZkSyncCompressionProof, ZkSyncCompressionVerificationKey) {
    let proof_config = circuit.proof_config_for_compression_step();
    let verifier_builder = circuit.into_dyn_verifier_builder();
    let verifier = verifier_builder.create_verifier();

    let (proof, vk, is_proof_valid) = match circuit {
        ZkSyncCompressionLayerCircuit::CompressionMode1Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode2Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode3Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode4Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionLayerCircuit::CompressionMode5Circuit(_inner) => {
            unreachable!("Only 4 modes of compression is allowed")
        }
    };
    if is_proof_valid == false {
        println!("Proof is invalid");
    }
    (proof, vk)
}

pub fn inner_prove_compression_wrapper_circuit(
    circuit: ZkSyncCompressionForWrapperCircuit,
    worker: &BoojumWorker,
) -> (
    ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
) {
    let proof_config = circuit.proof_config_for_compression_step();
    let verifier_builder = circuit.into_dyn_verifier_builder();
    let verifier = verifier_builder.create_verifier();

    let (proof, vk, is_proof_valid) = match circuit {
        ZkSyncCompressionForWrapperCircuit::CompressionMode1Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode2Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode3Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode4Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
        ZkSyncCompressionForWrapperCircuit::CompressionMode5Circuit(inner) => {
            let (proof, vk) = prove_compression_circuit(inner.clone(), proof_config, worker);
            let is_proof_valid = verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
            (proof, vk, is_proof_valid)
        }
    };
    if is_proof_valid == false {
        println!("Proof is invalid");
    }

    (proof, vk)
}

pub fn synthesize_circuit<CF: ProofCompressionFunction, CS: CSConfig>(
    circuit: CompressionLayerCircuit<CF>,
) -> (FinalizationHintsForProver, CSReferenceAssembly<F, F, CS>) {
    let geometry = circuit.geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl = CsReferenceImplementationBuilder::<GoldilocksField, F, CS>::new(
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

pub fn prove_compression_circuit<CF: ProofCompressionFunction>(
    circuit: CompressionLayerCircuit<CF>,
    proof_config: ProofConfig,
    worker: &BoojumWorker,
) -> (
    BoojumProof<F, CF::ThisLayerHasher, EXT>,
    BoojumVK<GoldilocksField, <CF as ProofCompressionFunction>::ThisLayerHasher>,
) {
    let (_, cs) = synthesize_circuit::<_, DevCSConfig>(circuit);
    cs.prove_one_shot::<_, CF::ThisLayerTranscript, CF::ThisLayerHasher, CF::ThisLayerPoW>(
        worker,
        proof_config,
        CF::this_layer_transcript_parameters(),
    )
}
