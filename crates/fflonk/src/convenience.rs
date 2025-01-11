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

pub type FflonkSnarkVerifierCircuitDeviceSetup<A> =
    FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, A>;

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
    let mut assembly = FflonkAssembly::<Bn256, SynthesisModeTesting, GlobalHost>::new();
    circuit.synthesize(&mut assembly).expect("must work");
    assert!(assembly.is_satisfied());
    let raw_trace_len = assembly.n();
    let domain_size = (raw_trace_len + 1).next_power_of_two();
    let _context = DeviceContextWithSingleDevice::init(domain_size)
        .expect("Couldn't create fflonk GPU Context");

    let setup =
        FflonkDeviceSetup::<_, FflonkSnarkVerifierCircuit, GlobalHost>::create_setup_from_assembly_on_device(
            &assembly,
        )
        .unwrap();
    let vk = setup.get_verification_key();

    assembly.finalize();
    assert_eq!(assembly.n(), vk.n);
    assert_eq!(assembly.n() + 1, domain_size);
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);

    let start = std::time::Instant::now();
    let proof = create_proof::<
        _,
        FflonkSnarkVerifierCircuit,
        _,
        RollingKeccakTranscript<_>,
        CombinedMonomialDeviceStorage<Fr>,
        _,
    >(&assembly, &setup, raw_trace_len)
    .unwrap();
    println!("proof generation takes {} ms", start.elapsed().as_millis());

    let valid = fflonk::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");

    (proof, vk)
}

pub fn gpu_prove_fflonk_snark_verifier_circuit_with_precomputation(
    circuit: &FflonkSnarkVerifierCircuit,
    setup: &FflonkSnarkVerifierCircuitDeviceSetup,
    vk: &FflonkSnarkVerifierCircuitVK,
) -> FflonkSnarkVerifierCircuitProof {
    println!("Synthesizing for fflonk proving");
    let mut proving_assembly = FflonkAssembly::<Bn256, SynthesisModeProve, GlobalHost>::new();
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
        _,
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
    let setup_file_path = format!("{}/final_snark_device_setup.bin", path);
    println!("Saving setup into file {setup_file_path}");
    let device_setup_file = std::fs::File::create(&setup_file_path).unwrap();
    device_setup.write(&device_setup_file).unwrap();
    println!("fflonk device setup saved into {}", setup_file_path);

    let vk_file_path = format!("{}/final_vk.json", path);
    let vk_file = std::fs::File::create(&vk_file_path).unwrap();
    serde_json::to_writer(&vk_file, &device_setup.get_verification_key()).unwrap();
    println!("fflonk vk saved into {}", vk_file_path);
}

pub fn load_device_setup_and_vk_of_fflonk_snark_circuit(
    path: &str,
) -> (
    FflonkSnarkVerifierCircuitDeviceSetup,
    FflonkSnarkVerifierCircuitVK,
) {
    println!("Loading fflonk setup for snark circuit");
    let setup_file_path = format!("{}/final_snark_device_setup.bin", path);
    let setup_file = std::fs::File::open(setup_file_path).unwrap();
    let device_setup = FflonkDeviceSetup::read(&setup_file).unwrap();

    let vk_file_path = format!("{}/final_vk.json", path);
    let vk_file_path = std::path::Path::new(&vk_file_path);
    let vk_file = std::fs::File::open(&vk_file_path).unwrap();
    let vk = serde_json::from_reader(&vk_file).unwrap();

    (device_setup, vk)
}
