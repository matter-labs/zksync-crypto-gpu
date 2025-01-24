use bellman::{
    bn256::Fr,
    plonk::{
        better_better_cs::cs::{Circuit, SynthesisModeTesting},
        commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    },
};
use fflonk::*;

use super::*;

#[test]
#[ignore]
fn test_snark_circuit_with_naive_main_gate() {
    let path = if let Ok(path) = std::env::var("BLOB_PATH") {
        path.to_string()
    } else {
        format!("./data")
    };
    let circuit = init_snark_wrapper_circuit(&path);
    let (proof, vk) =
        crate::convenience::gpu_prove_fflonk_snark_verifier_circuit_single_shot(&circuit);
    assert!(
        fflonk_cpu::verify::<Bn256, FflonkSnarkVerifierCircuit, RollingKeccakTranscript<Fr>>(
            &vk, &proof, None
        )
        .unwrap()
    );
}

#[test]
#[ignore]
fn test_simple_circuit_with_naive_main_gate() {
    use bellman::bn256::Bn256;
    type C = FflonkTestCircuit;
    let circuit = C {};

    let mut assembly = FflonkAssembly::<Bn256, SynthesisModeTesting>::new();
    circuit.synthesize(&mut assembly).expect("must work");
    assert!(assembly.is_satisfied());
    let raw_trace_len = assembly.n();
    assembly.finalize();
    let domain_size = assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);
    println!("Trace log length {}", domain_size.trailing_zeros());

    let _context = DeviceContextWithSingleDevice::init(domain_size).unwrap();

    let setup = FflonkDeviceSetup::create_setup_on_device(&circuit).unwrap();
    let vk = setup.get_verification_key();
    assert_eq!(vk.n + 1, domain_size);
    #[cfg(feature = "allocator")]
    let proof = create_proof::<_, C, _, RollingKeccakTranscript<Fr>, std::alloc::Global>(
        &assembly,
        &setup,
        raw_trace_len,
    )
    .expect("proof");
    #[cfg(not(feature = "allocator"))]
    let proof =
        create_proof::<_, C, _, RollingKeccakTranscript<Fr>>(&assembly, &setup, raw_trace_len)
            .expect("proof");

    let valid = fflonk::verify::<_, _, RollingKeccakTranscript<Fr>>(&vk, &proof, None).unwrap();
    assert!(valid, "proof verification fails");
}
