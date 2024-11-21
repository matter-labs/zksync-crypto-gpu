use std::alloc::Global;
use std::fs::File;
use std::path::Path;
use bellman::{
    bn256::{Bn256, Fr},
    kate_commitment::{Crs, CrsForMonomialForm},
    plonk::{
        better_better_cs::cs::{
            Circuit, SynthesisModeGenerateSetup, SynthesisModeProve, SynthesisModeTesting,
        },
        commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    },
};
use circuit_definitions::circuit_definitions::aux_layer::{
    wrapper::ZkSyncCompressionWrapper, ZkSyncCompressionProofForWrapper,
    ZkSyncCompressionVerificationKeyForWrapper,
};
use fflonk::{FflonkAssembly, L1_VERIFIER_DOMAIN_SIZE_LOG};
use nvtx::{range_pop, range_push};

pub type FflonkSnarkVerifierCircuitDeviceSetup =
    FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit>;

use super::*;

pub fn init_crs(
    worker: &bellman::worker::Worker,
    domain_size: usize,
) -> Crs<Bn256, CrsForMonomialForm> {
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);
    let num_points = MAX_COMBINED_DEGREE_FACTOR * domain_size;
    let mon_crs = if let Ok(crs_file_path) = std::env::var("CRS_FILE") {
        println!("using crs file at {crs_file_path}");
        let crs_file =
            std::fs::File::open(&crs_file_path).expect(&format!("crs file at {}", crs_file_path));
        let mon_crs = Crs::<Bn256, CrsForMonomialForm>::read(crs_file)
            .expect(&format!("read crs file at {}", crs_file_path));
        assert!(num_points <= mon_crs.g1_bases.len());

        mon_crs
    } else {
        Crs::<Bn256, CrsForMonomialForm>::non_power_of_two_crs_42(num_points, &worker)
    };

    mon_crs
}

pub fn init_snark_wrapper_circuit(path: &str) -> FflonkSnarkVerifierCircuit {
    let compression_wrapper_mode =
        if let Ok(compression_wrapper_mode) = std::env::var("COMPRESSION_WRAPPER_MODE") {
            compression_wrapper_mode.parse::<u8>().unwrap()
        } else {
            5u8
        };
    println!("Compression mode {}", compression_wrapper_mode);
    let compression_proof_file_path = if let Ok(file_path) = std::env::var("COMPRESSION_PROOF_FILE")
    {
        file_path
    } else {
        format!(
            "{}/compression_wrapper_{compression_wrapper_mode}_proof.json",
            path
        )
    };
    println!("Reading proof file at {compression_proof_file_path}");
    let compression_vk_file_path = if let Ok(file_path) = std::env::var("COMPRESSION_VK_FILE") {
        file_path
    } else {
        format!(
            "{}/compression_wrapper_{compression_wrapper_mode}_vk.json",
            path
        )
    };
    println!("Reading vk file at {compression_vk_file_path}");

    let compression_proof_file = std::fs::File::open(compression_proof_file_path).unwrap();
    let compression_proof: ZkSyncCompressionProofForWrapper =
        serde_json::from_reader(&compression_proof_file).unwrap();

    let compression_vk_file = std::fs::File::open(compression_vk_file_path).unwrap();
    let compression_vk: ZkSyncCompressionVerificationKeyForWrapper =
        serde_json::from_reader(&compression_vk_file).unwrap();

    init_snark_wrapper_circuit_from_inputs(
        compression_wrapper_mode,
        compression_proof,
        compression_vk,
    )
}

pub fn init_snark_wrapper_circuit_from_inputs(
    compression_wrapper_mode: u8,
    input_proof: ZkSyncCompressionProofForWrapper,
    input_vk: ZkSyncCompressionVerificationKeyForWrapper,
) -> FflonkSnarkVerifierCircuit {
    let wrapper_function =
        ZkSyncCompressionWrapper::from_numeric_circuit_type(compression_wrapper_mode);
    let fixed_parameters = input_vk.fixed_parameters.clone();

    FflonkSnarkVerifierCircuit {
        witness: Some(input_proof),
        vk: input_vk,
        fixed_parameters,
        transcript_params: (),
        wrapper_function,
    }
}

pub fn gpu_prove_fflonk_snark_verifier_circuit_single_shot(
    circuit: &FflonkSnarkVerifierCircuit,
) -> (
    FflonkSnarkVerifierCircuitProof,
    FflonkSnarkVerifierCircuitVK,
) {
    range_push!("prove");
    range_push!("setup");
    let setup_path = Path::new("./data");
    let setup = if FflonkDeviceSetup::<Bn256, FflonkSnarkVerifierCircuit>::is_saved(setup_path) {
        println!("loading setup from {}", setup_path.display());
        FflonkDeviceSetup::load(setup_path).unwrap()
    } else {
        let setup = FflonkDeviceSetup::create_setup_on_device(circuit)
            .unwrap();
        println!("saving setup to {}", setup_path.display());
        setup.save(setup_path).unwrap();
        setup
    };
    range_pop!();
    range_push!("vk");
    println!("generating vk");
    let vk = setup.get_verification_key();
    range_pop!();
    range_push!("assembly");
    let mut assembly = FflonkAssembly::<Bn256, SynthesisModeTesting, Global>::new();
    range_pop!();
    range_push!("synthesize");
    println!("synthesizing");
    circuit.synthesize(&mut assembly).expect("must work");
    // dbg!(assembly.n());
    range_pop!();
    // range_push!("satisfied");
    // println!("checking satisfied");
    // assert!(assembly.is_satisfied());
    // range_pop!();
    let raw_trace_len = assembly.n();
    let domain_size = (raw_trace_len + 1).next_power_of_two();
    range_push!("finalize");
    println!("finalizing");
    assembly.finalize();
    dbg!(assembly.n());
    range_pop!();
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);
    assert_eq!(assembly.n(), vk.n);
    assert_eq!(assembly.n() + 1, domain_size);
    range_push!("context");
    let _context = DeviceContextWithSingleDevice::init(domain_size)
        .expect("Couldn't create fflonk GPU Context");
    range_pop!();
    range_push!("create_proof");
    let start = std::time::Instant::now();
    let proof = create_proof::<
        _,
        FflonkSnarkVerifierCircuit,
        _,
        RollingKeccakTranscript<_>,
        CombinedMonomialDeviceStorage<Fr>,
    >(&assembly, &setup, raw_trace_len)
    .unwrap();
    println!("proof generation takes {} ms", start.elapsed().as_millis());
    range_pop!();

    let valid = fflonk::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");
    range_pop!();
    (proof, vk)
}

pub fn gpu_prove_fflonk_snark_verifier_circuit_with_precomputation(
    circuit: &FflonkSnarkVerifierCircuit,
    setup: &FflonkSnarkVerifierCircuitDeviceSetup,
    vk: &FflonkSnarkVerifierCircuitVK,
) -> FflonkSnarkVerifierCircuitProof {
    println!("Synthesizing for fflonk proving");
    let mut proving_assembly = FflonkAssembly::<Bn256, SynthesisModeProve, Global>::new();
    circuit
        .synthesize(&mut proving_assembly)
        .expect("must work");
    assert!(proving_assembly.is_satisfied());
    let raw_trace_len = proving_assembly.n();
    proving_assembly.finalize();
    let domain_size = proving_assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);

    let start = std::time::Instant::now();
    let proof = create_proof::<
        _,
        FflonkSnarkVerifierCircuit,
        _,
        RollingKeccakTranscript<_>,
        CombinedMonomialDeviceStorage<Fr>,
    >(&proving_assembly, setup, raw_trace_len)
    .unwrap();
    println!("proof generation takes {} ms", start.elapsed().as_millis());

    let valid = fflonk::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");

    proof
}

pub fn precompute_and_save_setup_and_vk_for_fflonk_snark_circuit(
    circuit: &FflonkSnarkVerifierCircuit,
    path: &str,
) {
    let compression_wrapper_mode = circuit.wrapper_function.numeric_circuit_type();
    println!("Compression mode: {compression_wrapper_mode}");
    println!("Generating fflonk setup data on the device");
    let device_setup =
        FflonkSnarkVerifierCircuitDeviceSetup::create_setup_on_device(&circuit).unwrap();
    println!("Saving fflonk setup into {path}");
    device_setup.save(path).unwrap();
    println!("Saved fflonk setup into {path}");
    let vk_file_path = Path::new(path).join("fflonk_vk.json");
    println!("Saving fflonk vk into {path}");
    let vk_file = File::create(&vk_file_path).unwrap();
    serde_json::to_writer(&vk_file, &device_setup.get_verification_key()).unwrap();
    println!("Saved fflonk vk into {path}");
}

pub fn load_device_setup_and_vk_of_fflonk_snark_circuit(
    path: &str,
) -> (
    FflonkSnarkVerifierCircuitDeviceSetup,
    FflonkSnarkVerifierCircuitVK,
) {
    println!("Loading fflonk setup from {path}");
    let setup = FflonkDeviceSetup::load(path).unwrap();
    println!("Loaded fflonk setup from {path}");
    println!("Loading fflonk vk from {path}");
    let vk_file_path = format!("{}/final_vk.json", path);
    let vk_file_path = Path::new(&vk_file_path);
    let vk_file = File::open(&vk_file_path).unwrap();
    let vk = serde_json::from_reader(&vk_file).unwrap();
    println!("Loaded fflonk vk from {path}");
    (setup, vk)
}
