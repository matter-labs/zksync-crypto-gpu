use bellman::{
    bn256::{Bn256, Fr},
    plonk::{
        better_better_cs::cs::{
            Assembly, Circuit, GateInternal, MainGate, PlonkConstraintSystemParams,
            PlonkCsWidth4WithNextStepParams, SynthesisModeGenerateSetup, SynthesisModeTesting,
            Width4MainGateWithDNext,
        },
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

    let mut assembly = FflonkAssembly::<Bn256, SynthesisModeTesting, GlobalHost>::new();
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

    let _context = DeviceContextWithSingleDevice::init(domain_size).unwrap();

    let setup = FflonkDeviceSetup::create_setup_on_host(&circuit, &worker);

    let proof = create_proof::<
        _,
        FflonkTestCircuit,
        _,
        RollingKeccakTranscript<Fr>,
        CombinedMonomialDeviceStorage<Fr>,
        GlobalHost,
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

type SimpleCircuitWidth4 = FflonkTestCircuit;
struct SimpleCircuitWidth3;

impl Circuit<Bn256> for SimpleCircuitWidth3 {
    type MainGate = Width4MainGateWithDNext;

    fn synthesize<CS: bellman::plonk::better_better_cs::cs::ConstraintSystem<Bn256> + 'static>(
        &self,
        cs: &mut CS,
    ) -> Result<(), bellman::SynthesisError> {
        let rng = &mut rand::thread_rng();
        use rand::Rand;

        for _ in 0..8 {
            use circuit_definitions::snark_wrapper::franklin_crypto::bellman::Field;
            use circuit_definitions::snark_wrapper::franklin_crypto::plonk::circuit::allocated_num::Num;
            use circuit_definitions::snark_wrapper::franklin_crypto::plonk::circuit::linear_combination::LinearCombination;
            let a = Fr::rand(rng);
            let b = Fr::rand(rng);
            let mut c = a;
            c.add_assign(&b);

            let a_var = Num::alloc(cs, Some(a))?;
            let b_var = Num::alloc(cs, Some(b))?;
            let c_var = Num::alloc(cs, Some(c))?;

            let mut lc = LinearCombination::zero();
            lc.add_assign_number_with_coeff(&a_var, Fr::one());
            lc.add_assign_number_with_coeff(&b_var, Fr::one());
            let mut minus_one = Fr::one();
            minus_one.negate();
            lc.add_assign_number_with_coeff(&c_var, minus_one);

            let _ = lc.into_num(cs)?;
        }

        let _input = cs.alloc_input(|| Ok(Fr::one()))?;

        Ok(())
    }
}

#[test]
fn test_permutation_materalization() {
    let mut assembly = Assembly::<
        Bn256,
        PlonkCsWidth4WithNextStepParams,
        Width4MainGateWithDNext,
        SynthesisModeGenerateSetup,
        GlobalHost,
    >::new();
    let circuit = SimpleCircuitWidth4 {};
    circuit.synthesize(&mut assembly).unwrap();
    run_permutation_materalization(assembly);

    // width 3 assembly
    let mut assembly_3cols = FflonkAssembly::<Bn256, SynthesisModeGenerateSetup, GlobalHost>::new();
    let circuit = SimpleCircuitWidth3 {};
    circuit.synthesize(&mut assembly_3cols).unwrap();
    run_permutation_materalization(assembly_3cols);
}

fn run_permutation_materalization<P: PlonkConstraintSystemParams<Bn256>, M: MainGate<Bn256>>(
    mut assembly: Assembly<Bn256, P, M, SynthesisModeGenerateSetup, GlobalHost>,
) {
    let worker = Worker::new();
    let raw_trace_len = assembly.n();
    let num_input_variables = assembly.num_inputs;
    assert_eq!(num_input_variables, 1);

    let num_input_gates = assembly.num_input_gates;
    assert_eq!(num_input_gates, 1);
    let num_aux_gates = assembly.num_aux_gates;
    assert_eq!(num_input_gates + num_aux_gates, raw_trace_len);

    assert!(assembly.is_satisfied());
    println!("Raw Trace length {}", assembly.n());
    assembly.finalize();

    let domain_size = assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(domain_size <= 1 << L1_VERIFIER_DOMAIN_SIZE_LOG);
    println!("Trace log length {}", domain_size.trailing_zeros());

    let expected_permutation_monomials = assembly.make_permutations(&worker).unwrap();

    let num_cols = GateInternal::<Bn256>::variable_polynomials(&assembly.main_gate).len();
    let mut h_transformed_variables = Vec::with_capacity(num_cols * raw_trace_len);

    let _context = DeviceContextWithSingleDevice::init_no_msm(domain_size).unwrap();
    let pool = bc_mem_pool::new(DEFAULT_DEVICE_ID).unwrap();
    let stream = bc_stream::new().unwrap();
    let mut transformed_variables = DVec::allocate_zeroed_on(num_cols * domain_size, pool, stream);

    for ((((_, src_aux), (_, src_input)), h_dst), dst) in assembly
        .aux_storage
        .state_map
        .iter()
        .zip(assembly.inputs_storage.state_map.iter())
        .zip(h_transformed_variables.chunks_mut(raw_trace_len))
        .zip(transformed_variables.chunks_mut(domain_size))
    {
        assert_eq!(src_input.len(), num_input_gates);
        assert_eq!(src_aux.len(), num_aux_gates);
        _transform_variables(
            &src_input,
            &mut h_dst[..num_input_gates],
            num_input_variables,
            &worker,
        );
        _transform_variables(
            &src_aux,
            &mut h_dst[num_input_gates..],
            num_input_variables,
            &worker,
        );
        mem::h2d_on(h_dst, dst, stream).unwrap();
    }

    let h_actual_permutations = match num_cols {
        3 => compute_actual_permutations::<3>(&transformed_variables, domain_size),
        4 => compute_actual_permutations::<4>(&transformed_variables, domain_size),
        _ => unimplemented!(),
    };

    assert_eq!(
        expected_permutation_monomials.len(),
        h_actual_permutations.len()
    );
    for (this, other) in expected_permutation_monomials
        .iter()
        .zip(h_actual_permutations.iter())
    {
        assert_eq!(this.size(), other.len());
        assert_eq!(this.as_ref(), other);
    }
}

fn compute_actual_permutations<const N: usize>(
    transformed_variables: &DSlice<u32>,
    domain_size: usize,
) -> Vec<Vec<Fr>> {
    let stream = bc_stream::new().unwrap();
    let pool = bc_mem_pool::new(DEFAULT_DEVICE_ID).unwrap();
    let actual_permutations =
        materialize_permutation_polys::<Fr, 3>(&transformed_variables, domain_size, pool, stream)
            .unwrap();

    let h_actual_permutations: Vec<_> = actual_permutations
        .iter()
        .map(|p| p.as_ref().to_vec(stream).unwrap())
        .collect();
    stream.sync().unwrap();

    h_actual_permutations
}
