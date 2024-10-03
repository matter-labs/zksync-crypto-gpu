use std::alloc::Global;

use bellman::{
    bn256::{Bn256, Fr},
    plonk::{
        better_better_cs::cs::{Circuit, SynthesisModeTesting},
        commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    },
    worker::Worker,
};
use fflonk::*;

use super::*;

pub fn load_fflonk_test_vk() -> FflonkVerificationKey<Bn256, FflonkTestCircuit> {
    let vk_file_path = if let Ok(vk_file_path) = std::env::var("VK_FILE") {
        vk_file_path
    } else {
        "./data/test_vk.json".to_string()
    };
    println!("reading fflonk test proof from file at {vk_file_path}");
    let vk_file = std::fs::File::open(&vk_file_path).unwrap();
    let vk = serde_json::from_reader(&vk_file).unwrap();

    vk
}

#[test]
#[ignore]
fn test_fflonk_proof_verification() {
    let vk_file_path = std::env::var("FFLONK_VK_FILE").expect("fflonk vk file path");
    let vk_file =
        std::fs::File::open(&vk_file_path).expect(&format!("vk file at {}", vk_file_path));
    let vk: FflonkSnarkVerifierCircuitVK =
        serde_json::from_reader(&vk_file).expect("deserialize vk");

    let proof_file_path = std::env::var("FFLONK_PROOF_FILE").expect("fflonk proof file path");
    let proof_file =
        std::fs::File::open(&proof_file_path).expect(&format!("proof file at {}", proof_file_path));
    let proof: FflonkSnarkVerifierCircuitProof =
        serde_json::from_reader(&proof_file).expect("deserialize proof");

    let is_valid = fflonk::verifier::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None)
        .expect("verify proof");
    assert!(is_valid, "fflonk proof is not corrects");
}

#[test]
#[ignore]
fn test_snark_circuit_with_naive_main_gate() {
    let path = if let Ok(path) = std::env::var("BLOB_PATH") {
        path.to_string()
    } else {
        format!("./data")
    };
    let worker = Worker::new();
    let circuit = init_snark_wrapper_circuit(&path);
    let (proof, vk) =
        crate::convenience::gpu_prove_fflonk_snark_verifier_circuit_single_shot(&circuit, &worker);

    save_fflonk_proof_and_vk_into_file(&proof, &vk, &path);
}

#[test]
fn test_test_circuit_with_naive_main_gate() {
    use bellman::bn256::Bn256;
    let circuit = FflonkTestCircuit {};
    let worker = Worker::new();

    let mut assembly = FflonkAssembly::<Bn256, SynthesisModeTesting>::new();
    circuit.synthesize(&mut assembly).expect("must work");
    assert!(assembly.is_satisfied());
    assembly.finalize();
    let domain_size = assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);
    println!("Trace log length {}", domain_size.trailing_zeros());

    let vk = load_fflonk_test_vk();
    dbg!(&vk.c0);

    println!("Creating device setup");

    let domain_size = vk.n + 1;
    assert!(domain_size.is_power_of_two());

    let _context = unsafe { DeviceContextWithSingleDevice::init(domain_size).unwrap() };

    let setup = FflonkDeviceSetup::create_setup_on_host(&circuit, &worker);

    let proof = create_proof::<
        _,
        FflonkTestCircuit,
        _,
        RollingKeccakTranscript<Fr>,
        CombinedMonomialDeviceStorage<Fr>,
        Global,
    >(&assembly, &setup, &worker)
    .expect("proof");

    let valid = fflonk::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");
}

#[test]
fn test_device_setup() {
    let circuit = FflonkTestCircuit {};
    let worker = Worker::new();

    let start = std::time::Instant::now();
    let device_setup_on_device =
        FflonkDeviceSetup::<_, FflonkTestCircuit, std::alloc::Global>::create_setup_on_device(
            &circuit, &worker,
        )
        .unwrap();
    println!(
        "Setup creation on device takes {} s",
        start.elapsed().as_secs()
    );

    let start = std::time::Instant::now();
    let device_setup_on_host =
        FflonkDeviceSetup::<_, FflonkTestCircuit, std::alloc::Global>::create_setup_on_host(
            &circuit, &worker,
        );
    println!(
        "Setup creation on host takes {} s",
        start.elapsed().as_secs()
    );

    assert_eq!(
        device_setup_on_host.c0_commitment,
        device_setup_on_device.c0_commitment
    );
    assert_eq!(
        device_setup_on_host.main_gate_selector_monomials,
        device_setup_on_device.main_gate_selector_monomials
    );

    assert_eq!(
        device_setup_on_host.permutation_monomials,
        device_setup_on_device.permutation_monomials
    );
}
