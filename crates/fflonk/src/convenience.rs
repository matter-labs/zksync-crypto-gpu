use bellman::{
    bn256::{Bn256, Fr},
    plonk::{
        better_better_cs::cs::{
            Circuit, SynthesisModeGenerateSetup, SynthesisModeProve, SynthesisModeTesting,
        },
        commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    },
    worker::Worker,
};
use fflonk::{
    init_crs, FflonkAssembly, FflonkSetup, FflonkSnarkVerifierCircuit,
    FflonkSnarkVerifierCircuitProof, FflonkSnarkVerifierCircuitVK, FflonkVerificationKey,
    L1_VERIFIER_DOMAIN_SIZE_LOG,
};

pub type FflonkSnarkVerifierCircuitDeviceSetup =
    FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit>;

use super::*;

pub fn prove_fflonk_snark_verifier_circuit_single_shot(
    circuit: &FflonkSnarkVerifierCircuit,
    worker: &Worker,
) -> (
    FflonkSnarkVerifierCircuitProof,
    FflonkSnarkVerifierCircuitVK,
) {
    let mut assembly = FflonkAssembly::<Bn256, SynthesisModeTesting>::new();
    circuit.synthesize(&mut assembly).expect("must work");
    assert!(assembly.is_satisfied());
    assembly.finalize();
    let domain_size = assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);

    let mon_crs = init_crs(&worker, domain_size);
    let host_setup = FflonkSetup::create_setup(&assembly, &worker, &mon_crs).expect("setup");
    let vk = FflonkVerificationKey::from_setup(&host_setup, &mon_crs).unwrap();

    let context = unsafe { DeviceContextWithSingleDevice::init(domain_size).unwrap() };

    let setup = FflonkDeviceSetup::<_, FflonkSnarkVerifierCircuit>::from_host_setup(host_setup);

    let start = std::time::Instant::now();
    let proof = create_proof::<_, FflonkSnarkVerifierCircuit, _, RollingKeccakTranscript<_>, _>(
        &assembly, &setup, &worker,
    )
    .unwrap();
    println!("proof generation takes {} ms", start.elapsed().as_millis());

    let valid = fflonk::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");

    (proof, vk)
}

pub fn prove_fflonk_snark_verifier_circuit_with_precomputation(
    circuit: &FflonkSnarkVerifierCircuit,
    setup: &FflonkSnarkVerifierCircuitDeviceSetup,
    vk: &FflonkSnarkVerifierCircuitVK,
    worker: &Worker,
) -> FflonkSnarkVerifierCircuitProof {
    let mut proving_assembly = FflonkAssembly::<Bn256, SynthesisModeProve>::new();
    circuit
        .synthesize(&mut proving_assembly)
        .expect("must work");
    assert!(proving_assembly.is_satisfied());
    proving_assembly.finalize();
    let domain_size = proving_assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);

    let context = unsafe { DeviceContextWithSingleDevice::init(domain_size).unwrap() };

    let start = std::time::Instant::now();
    let proof = create_proof::<_, FflonkSnarkVerifierCircuit, _, RollingKeccakTranscript<_>, _>(
        &proving_assembly,
        &setup,
        &worker,
    )
    .unwrap();
    println!("proof generation takes {} ms", start.elapsed().as_millis());

    let valid = fflonk::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");

    proof
}

pub fn precompute_and_save_setup_for_fflonk_snark_circuit(
    circuit: &FflonkSnarkVerifierCircuit,
    worker: &Worker,
    output_blob_path: &str,
) {
    let compression_wrapper_mode = circuit.wrapper_function.numeric_circuit_type();
    println!("Compression mode: {compression_wrapper_mode}");
    let mut setup_assembly = FflonkAssembly::<Bn256, SynthesisModeGenerateSetup>::new();
    circuit.synthesize(&mut setup_assembly).expect("must work");
    assert!(setup_assembly.is_satisfied());
    setup_assembly.finalize();

    let domain_size = setup_assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);

    let mon_crs = init_crs(&worker, domain_size);
    todo!()
}
