use super::*;
use crate::cs::{
    gpu_setup_and_vk_from_base_setup_vk_params_and_hints,
    materialize_permutation_cols_from_indexes_into,
};
use crate::gpu_proof_config::GpuProofConfig;
use boojum::cs::implementations::transcript::GoldilocksPoisedon2Transcript;
use boojum::cs::oracle::TreeHasher;
use boojum::cs::{implementations::polynomial_storage::SetupBaseStorage, Variable};
use boojum::field::traits::field_like::PrimeFieldLikeVectorized;
use boojum::sha2::{Digest, Sha256};
use boojum::{
    config::{CSConfig, CSSetupConfig, DevCSConfig, ProvingCSConfig, SetupCSConfig},
    cs::{
        gates::{
            ConstantAllocatableCS, ConstantsAllocatorGate, FmaGateInBaseFieldWithoutConstant,
            NopGate, PublicInputGate, ReductionGate,
        },
        implementations::{
            pow::NoPow, proof::Proof, prover::ProofConfig, reference_cs::CSReferenceAssembly,
            setup::FinalizationHintsForProver,
        },
        traits::{cs::ConstraintSystem, gate::GatePlacementStrategy},
        CSGeometry,
    },
    field::{goldilocks::GoldilocksExt2, U64Representable},
    gadgets::{
        sha256::sha256,
        tables::{
            ch4::{create_ch4_table, Ch4Table},
            chunk4bits::{create_4bit_chunk_split_table, Split4BitChunkTable},
            maj4::{create_maj4_table, Maj4Table},
            trixor4::{create_tri_xor_table, TriXor4Table},
        },
        traits::{
            round_function::{BuildableCircuitRoundFunction, CircuitRoundFunction},
            witnessable::WitnessHookable,
        },
        u8::UInt8,
    },
    implementations::poseidon2::Poseidon2Goldilocks,
    worker::Worker,
};
use boojum_cuda::poseidon2::GLHasher;
use serial_test::serial;
use std::{path::Path, sync::Arc};

#[cfg(test)]
type DefaultTranscript = GoldilocksPoisedon2Transcript;

#[cfg(test)]
type DefaultTreeHasher = GLHasher;

#[allow(dead_code)]
pub type DefaultDevCS = CSReferenceAssembly<F, F, DevCSConfig>;

#[serial]
#[test]
#[ignore]
fn test_proof_comparison_for_poseidon_gate_with_private_witnesses() {
    let (setup_cs, finalization_hint) =
        init_or_synth_cs_with_poseidon2_and_private_witnesses::<SetupCSConfig, true>(None);
    let worker = Worker::new();
    let prover_config = init_proof_cfg();
    let (setup_base, setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::create_with_config(
        ProverContextConfig::default().with_smallest_supported_domain_size(domain_size),
    )
    .expect("init gpu prover context");
    let gpu_setup = {
        let (base_setup, vk_params, variables_hint, witness_hint) = setup_cs.get_light_setup(
            &worker,
            prover_config.fri_lde_factor,
            prover_config.merkle_tree_cap_size,
        );
        let (gpu_setup, gpu_vk) = gpu_setup_and_vk_from_base_setup_vk_params_and_hints(
            base_setup,
            vk_params,
            variables_hint,
            witness_hint,
            &worker,
        )
        .unwrap();
        assert_eq!(vk, gpu_vk);
        gpu_setup
    };
    assert!(domain_size.is_power_of_two());
    let actual_proof = {
        let (proving_cs, _) = init_or_synth_cs_with_poseidon2_and_private_witnesses::<
            ProvingCSConfig,
            true,
        >(finalization_hint.as_ref());
        let witness = proving_cs.witness.unwrap();
        let (reusable_cs, _) = init_or_synth_cs_with_poseidon2_and_private_witnesses::<
            ProvingCSConfig,
            false,
        >(finalization_hint.as_ref());
        let config = GpuProofConfig::from_assembly(&reusable_cs);

        gpu_prove_from_external_witness_data::<DefaultTranscript, DefaultTreeHasher, NoPow, Global>(
            &config,
            &witness,
            prover_config.clone(),
            &gpu_setup,
            &vk,
            (),
            &worker,
        )
        .expect("gpu proof")
    };

    let expected_proof = {
        let (proving_cs, _) = init_or_synth_cs_with_poseidon2_and_private_witnesses::<
            ProvingCSConfig,
            true,
        >(finalization_hint.as_ref());
        let worker = Worker::new();
        let prover_config = init_proof_cfg();

        proving_cs.prove_from_precomputations::<GoldilocksExt2, DefaultTranscript, DefaultTreeHasher, NoPow>(
            prover_config,
            &setup_base,
            &setup,
            &setup_tree,
            &vk,
            &vars_hint,
            &wits_hint,
            (),
            &worker,
        )
    };
    let actual_proof = actual_proof.into();
    compare_proofs(&expected_proof, &actual_proof);
}

fn init_or_synth_cs_with_poseidon2_and_private_witnesses<CFG: CSConfig, const DO_SYNTH: bool>(
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG>,
    Option<FinalizationHintsForProver>,
) {
    let geometry = CSGeometry {
        num_columns_under_copy_permutation: 100,
        num_witness_columns: 30,
        num_constant_columns: 4,
        max_allowed_constraint_degree: 4,
    };

    use boojum::cs::cs_builder_reference::*;
    let builder_impl = CsReferenceImplementationBuilder::<F, F, CFG>::new(geometry, 1 << 20);
    use boojum::cs::cs_builder::new_builder;
    let builder = new_builder::<_, F>(builder_impl);

    let builder = Poseidon2Goldilocks::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder = ConstantsAllocatorGate::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder =
        NopGate::configure_builder(builder, GatePlacementStrategy::UseGeneralPurposeColumns);

    let mut owned_cs = builder.build(1 << 25);
    // quick and dirty way of testing with private witnesses
    fn synthesize<CS: ConstraintSystem<F>>(cs: &mut CS) -> [Variable; 8] {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        type R = Poseidon2Goldilocks;
        let num_gates = 1 << 16;
        let mut prev_state = [cs.allocate_constant(F::ZERO); 12];
        let _to_keep = [cs.allocate_constant(F::ZERO); 4];
        for _ in 0..num_gates {
            let to_absorb =
                cs.alloc_multiple_variables_from_witnesses([F::from_u64_unchecked(rng.gen()); 8]);
            let to_keep = R::split_capacity_elements(&prev_state);
            prev_state = R::absorb_with_replacement(cs, to_absorb, to_keep);
            prev_state = R::compute_round_function(cs, prev_state);
        }

        Poseidon2Goldilocks::state_into_commitment::<8>(&prev_state)
    }

    if DO_SYNTH {
        let output = synthesize(&mut owned_cs);
        let next_available_row = owned_cs.next_available_row();
        for (column, var) in output.into_iter().enumerate() {
            // TODO: Ask Sait
            // I'm not sure it's ok to add a gate only if we synthesized the witness.
            // This may yield inconsistencies between a fully synthesized cs created
            // with DO_SYNTH=true and a reusable cs created with DO_SYNTH=false.
            // On the other hand, in the "ordinary" zksync circuits I'm fairly sure
            // place_gate, place_variable, and set_public are called during synthesis.
            let gate = PublicInputGate::new(var);
            owned_cs.place_gate(&gate, next_available_row);
            owned_cs.place_variable(var, next_available_row, column);
            owned_cs.set_public(column, next_available_row);
        }
    }

    // imitates control flow of synthesis_utils::init_or_synthesize_assembly
    if <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP {
        let (_, finalization_hint) = owned_cs.pad_and_shrink();
        (owned_cs.into_assembly(), Some(finalization_hint))
    } else {
        let hint = finalization_hint.unwrap();
        if DO_SYNTH {
            owned_cs.pad_and_shrink_using_hint(hint);
            (owned_cs.into_assembly(), None)
        } else {
            (owned_cs.into_assembly_for_repeated_proving(hint), None)
        }
    }
}

#[serial]
#[test]
#[ignore]
fn test_permutation_polys() {
    let (setup_cs, _finalization_hint) =
        init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (setup_base, vk_params, variables_hint, witnesses_hint) = setup_cs.get_light_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let expected_permutation_polys = setup_base.copy_permutation_polys.clone();

    let domain_size = setup_cs.max_trace_len;
    let cfg = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let _ctx = ProverContext::create_with_config(cfg).expect("init gpu prover context");

    let num_copy_permutation_polys = variables_hint.maps.len();
    let (gpu_setup, _) =
        gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
            setup_base,
            vk_params,
            variables_hint,
            witnesses_hint,
            &worker,
        )
        .expect("gpu setup");
    println!("Gpu setup is made");

    let mut actual_copy_permutation_polys =
        GenericStorage::allocate(num_copy_permutation_polys, domain_size);
    let copy_permutation_polys_as_slice_view = actual_copy_permutation_polys.as_single_slice_mut();
    println!("GenericSetupStorage is allocated");
    let variable_indexes =
        construct_indexes_from_hint(&gpu_setup.variables_hint, domain_size, &worker).unwrap();
    materialize_permutation_cols_from_indexes_into(
        copy_permutation_polys_as_slice_view,
        &variable_indexes,
        num_copy_permutation_polys,
        domain_size,
    )
    .unwrap();
    println!("Permutation polynomials are constructed");

    for (expected, actual) in expected_permutation_polys
        .into_iter()
        .map(|p| Arc::try_unwrap(p).unwrap())
        .map(|p| p.storage)
        .map(F::vec_into_base_vec)
        .zip(
            actual_copy_permutation_polys
                .into_poly_storage()
                .polynomials
                .into_iter()
                .map(|p| p.storage.into_inner())
                .map(|p| p.to_vec().unwrap()),
        )
    {
        assert_eq!(expected, actual);
    }
}

#[serial]
#[test]
#[ignore]
fn test_setup_comparison() {
    let (setup_cs, _) = init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );

    let _expected_permutation_polys = setup_base.copy_permutation_polys.clone();

    let domain_size = setup_cs.max_trace_len;
    let cfg = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let _ctx = ProverContext::create_with_config(cfg).expect("init gpu prover context");

    let expected_setup = GenericSetupStorage::from_host_values(&setup_base).unwrap();

    let (gpu_setup, _vk) = gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
        DefaultTreeHasher,
        _,
    >(setup_base, vk_params, vars_hint, wits_hint, &worker)
    .expect("gpu setup");

    let actual_setup = GenericSetupStorage::from_gpu_setup(&gpu_setup, &worker).unwrap();

    assert_eq!(
        expected_setup.inner.to_vec().unwrap(),
        actual_setup.inner.to_vec().unwrap(),
    );

    let expected_monomial = expected_setup.into_monomials().unwrap();
    let actual_monomial = actual_setup.into_monomials().unwrap();

    assert_eq!(
        expected_monomial.inner.to_vec().unwrap(),
        actual_monomial.inner.to_vec().unwrap(),
    );
}

#[cfg(feature = "allocator_stats")]
#[serial]
#[test]
#[ignore]
fn test_dry_runs() {
    let (setup_cs, finalization_hint) =
        init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);
    let (proving_cs, _) =
        init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(finalization_hint.as_ref());
    let witness = proving_cs.witness.unwrap();
    let (reusable_cs, _) =
        init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, false>(finalization_hint.as_ref());
    let config = GpuProofConfig::from_assembly(&reusable_cs);
    let worker = Worker::new();
    let prover_config = init_proof_cfg();
    let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let domain_size = setup_cs.max_trace_len;
    let cfg = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let _ctx = ProverContext::create_with_config(cfg).expect("init gpu prover context");
    let (gpu_setup, vk) =
        gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
            setup_base.clone(),
            vk_params,
            vars_hint.clone(),
            wits_hint.clone(),
            &worker,
        )
        .unwrap();

    assert!(domain_size.is_power_of_two());
    let candidates = CacheStrategy::get_strategy_candidates(
        &config,
        &prover_config,
        &gpu_setup,
        &vk.fixed_parameters,
    );
    for (_, strategy) in candidates.iter().copied() {
        let proof = || {
            let _ = gpu_prove_from_external_witness_data_with_cache_strategy::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                &witness,
                prover_config.clone(),
                &gpu_setup,
                &vk,
                (),
                &worker,
                strategy,
            )
            .expect("gpu proof");
        };
        dry_run_start();
        proof();
        dry_run_stop().unwrap();
        let dry = _alloc()
            .stats
            .lock()
            .unwrap()
            .allocations_at_maximum_block_count_at_maximum_tail_index
            .clone();
        let dry_tail_index = dry.tail_index();
        _setup_cache_reset();
        _alloc().stats.lock().unwrap().reset();
        assert_eq!(_alloc().stats.lock().unwrap().allocations.tail_index(), 0);
        proof();
        let wet = _alloc()
            .stats
            .lock()
            .unwrap()
            .allocations_at_maximum_block_count_at_maximum_tail_index
            .clone();
        let wet_tail_index = wet.tail_index();
        _setup_cache_reset();
        _alloc().stats.lock().unwrap().reset();
        assert_eq!(_alloc().stats.lock().unwrap().allocations.tail_index(), 0);
        assert_eq!(dry_tail_index, wet_tail_index);
    }
}

#[serial]
#[test]
#[ignore]
fn test_proof_comparison_for_sha256() {
    let (setup_cs, finalization_hint) =
        init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();
    let (setup_base, setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let domain_size = setup_cs.max_trace_len;
    let cfg = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
    let _ctx = ProverContext::create_with_config(cfg).expect("init gpu prover context");
    let (gpu_setup, gpu_vk) =
        gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
            setup_base.clone(),
            vk.fixed_parameters.clone(),
            vars_hint.clone(),
            wits_hint.clone(),
            &worker,
        )
        .unwrap();
    assert_eq!(vk, gpu_vk);
    assert!(domain_size.is_power_of_two());
    let actual_proof = {
        let (proving_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(
            finalization_hint.as_ref(),
        );
        let witness = proving_cs.witness.unwrap();
        let (reusable_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, false>(
            finalization_hint.as_ref(),
        );
        let config = GpuProofConfig::from_assembly(&reusable_cs);

        gpu_prove_from_external_witness_data::<DefaultTranscript, DefaultTreeHasher, NoPow, Global>(
            &config,
            &witness,
            prover_config.clone(),
            &gpu_setup,
            &vk,
            (),
            &worker,
        )
        .expect("gpu proof")
    };

    let expected_proof = {
        let (proving_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(
            finalization_hint.as_ref(),
        );
        let worker = Worker::new();
        let prover_config = init_proof_cfg();

        proving_cs.prove_from_precomputations::<GoldilocksExt2, DefaultTranscript, DefaultTreeHasher, NoPow>(
            prover_config,
            &setup_base,
            &setup,
            &setup_tree,
            &vk,
            &vars_hint,
            &wits_hint,
            (),
            &worker,
        )
    };
    let actual_proof = actual_proof.into();
    compare_proofs(&expected_proof, &actual_proof);
}

fn init_or_synth_cs_for_sha256<CFG: CSConfig, A: GoodAllocator, const DO_SYNTH: bool>(
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG, A>,
    Option<FinalizationHintsForProver>,
) {
    // let len = 10 * 64 + 64 - 9;
    // let len = 2 * (1 << 10);
    let len = 2 * (1 << 2);
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut input = vec![];
    for _ in 0..len {
        let byte: u8 = rng.gen();
        input.push(byte);
    }

    let mut hasher = Sha256::new();
    hasher.update(&input);
    let reference_output = hasher.finalize();

    let geometry = CSGeometry {
        num_columns_under_copy_permutation: 32,
        num_witness_columns: 0,
        num_constant_columns: 4,
        max_allowed_constraint_degree: 4,
    };

    use boojum::cs::cs_builder_reference::*;
    let builder_impl = CsReferenceImplementationBuilder::<F, F, CFG>::new(geometry, 1 << 19);
    use boojum::cs::cs_builder::new_builder;
    let builder = new_builder::<_, F>(builder_impl);

    let builder = builder.allow_lookup(
        boojum::cs::LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
            width: 4,
            num_repetitions: 8,
            share_table_id: true,
        },
    );

    let builder = ConstantsAllocatorGate::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder = FmaGateInBaseFieldWithoutConstant::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder = ReductionGate::<F, 4>::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    // not present in boojum/src/gadgets/sha256
    // let builder = PublicInputGate::configure_builder(
    //     builder,
    //     GatePlacementStrategy::UseGeneralPurposeColumns,
    // );
    let builder =
        NopGate::configure_builder(builder, GatePlacementStrategy::UseGeneralPurposeColumns);

    let mut owned_cs = builder.build(1 << 25);

    // add tables
    let table = create_tri_xor_table();
    owned_cs.add_lookup_table::<TriXor4Table, 4>(table);

    let table = create_ch4_table();
    owned_cs.add_lookup_table::<Ch4Table, 4>(table);

    let table = create_maj4_table();
    owned_cs.add_lookup_table::<Maj4Table, 4>(table);

    let table = create_4bit_chunk_split_table::<F, 1>();
    owned_cs.add_lookup_table::<Split4BitChunkTable<1>, 4>(table);

    let table = create_4bit_chunk_split_table::<F, 2>();
    owned_cs.add_lookup_table::<Split4BitChunkTable<2>, 4>(table);

    if DO_SYNTH {
        let mut circuit_input = vec![];

        let cs = &mut owned_cs;

        for el in input.iter() {
            let el = UInt8::allocate_checked(cs, *el);
            circuit_input.push(el);
        }

        let output = sha256(cs, &circuit_input);
        dbg!(output.len());

        // not present in boojum/src/gadgets/sha256
        // let mut next_available_row = cs.next_available_row();
        // for (column, var) in output.iter().enumerate() {
        //     let gate = PublicInputGate::new(var.get_variable());
        //     cs.place_gate(&gate, next_available_row);
        //     cs.place_variable(var.get_variable(), next_available_row, column);
        //     cs.set_public(column, next_available_row);
        // }
        let output = hex::encode(output.witness_hook(&*cs)().unwrap());
        let reference_output = hex::encode(reference_output.as_slice());
        assert_eq!(output, reference_output);
    }

    // imitates control flow of synthesis_utils::init_or_synthesize_assembly
    if <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP {
        let (_, finalization_hint) = owned_cs.pad_and_shrink();
        let owned_cs = owned_cs.into_assembly();
        (owned_cs, Some(finalization_hint))
    } else {
        let hint = finalization_hint.unwrap();
        if DO_SYNTH {
            owned_cs.pad_and_shrink_using_hint(hint);
            let owned_cs = owned_cs.into_assembly();
            (owned_cs, None)
        } else {
            (owned_cs.into_assembly_for_repeated_proving(hint), None)
        }
    }
}

fn compare_proofs<H: TreeHasher<F>>(
    expected_proof: &Proof<F, H, EXT>,
    actual_proof: &Proof<F, H, EXT>,
) {
    assert_eq!(expected_proof.public_inputs, actual_proof.public_inputs);
    assert_eq!(
        expected_proof.witness_oracle_cap,
        actual_proof.witness_oracle_cap
    );
    assert_eq!(
        expected_proof.stage_2_oracle_cap,
        actual_proof.stage_2_oracle_cap
    );
    assert_eq!(
        expected_proof.quotient_oracle_cap,
        actual_proof.quotient_oracle_cap
    );

    assert_eq!(expected_proof.values_at_z, actual_proof.values_at_z);
    assert_eq!(
        expected_proof.values_at_z_omega,
        actual_proof.values_at_z_omega
    );
    assert_eq!(expected_proof.values_at_0, actual_proof.values_at_0);
    assert_eq!(
        expected_proof.fri_base_oracle_cap,
        actual_proof.fri_base_oracle_cap
    );
    assert_eq!(
        expected_proof.fri_intermediate_oracles_caps,
        actual_proof.fri_intermediate_oracles_caps
    );
    assert_eq!(
        expected_proof.final_fri_monomials,
        actual_proof.final_fri_monomials
    );
    assert_eq!(expected_proof.pow_challenge, actual_proof.pow_challenge);
    assert_eq!(
        expected_proof.queries_per_fri_repetition.len(),
        actual_proof.queries_per_fri_repetition.len(),
    );

    for (expected_fri_query, actual_fri_query) in expected_proof
        .queries_per_fri_repetition
        .iter()
        .zip(actual_proof.queries_per_fri_repetition.iter())
    {
        // leaf elems
        assert_eq!(
            expected_fri_query.witness_query.leaf_elements.len(),
            actual_fri_query.witness_query.leaf_elements.len()
        );
        assert_eq!(
            expected_fri_query.witness_query.leaf_elements,
            actual_fri_query.witness_query.leaf_elements
        );

        assert_eq!(
            expected_fri_query.stage_2_query.leaf_elements.len(),
            actual_fri_query.stage_2_query.leaf_elements.len(),
        );
        assert_eq!(
            expected_fri_query.stage_2_query.leaf_elements,
            actual_fri_query.stage_2_query.leaf_elements
        );

        assert_eq!(
            expected_fri_query.quotient_query.leaf_elements.len(),
            actual_fri_query.quotient_query.leaf_elements.len()
        );
        assert_eq!(
            expected_fri_query.quotient_query.leaf_elements,
            actual_fri_query.quotient_query.leaf_elements
        );
        assert_eq!(
            expected_fri_query.setup_query.leaf_elements.len(),
            actual_fri_query.setup_query.leaf_elements.len()
        );
        assert_eq!(
            expected_fri_query.setup_query.leaf_elements,
            actual_fri_query.setup_query.leaf_elements
        );

        assert_eq!(
            expected_fri_query.fri_queries.len(),
            actual_fri_query.fri_queries.len(),
        );

        for (expected, actual) in expected_fri_query
            .fri_queries
            .iter()
            .zip(actual_fri_query.fri_queries.iter())
        {
            assert_eq!(expected.leaf_elements.len(), actual.leaf_elements.len());
            assert_eq!(expected.leaf_elements, actual.leaf_elements);
        }

        // merkle paths
        assert_eq!(
            expected_fri_query.witness_query.proof.len(),
            actual_fri_query.witness_query.proof.len(),
        );
        assert_eq!(
            expected_fri_query.witness_query.proof,
            actual_fri_query.witness_query.proof
        );
        assert_eq!(
            expected_fri_query.stage_2_query.proof.len(),
            actual_fri_query.stage_2_query.proof.len()
        );
        assert_eq!(
            expected_fri_query.quotient_query.proof,
            actual_fri_query.quotient_query.proof
        );

        assert_eq!(
            expected_fri_query.setup_query.proof.len(),
            actual_fri_query.setup_query.proof.len(),
        );

        assert_eq!(
            expected_fri_query.setup_query.proof,
            actual_fri_query.setup_query.proof,
        );

        for (expected, actual) in expected_fri_query
            .fri_queries
            .iter()
            .zip(actual_fri_query.fri_queries.iter())
        {
            assert_eq!(expected.proof.len(), actual.proof.len());
            assert_eq!(expected.proof, actual.proof);
        }
    }
}

#[serial]
#[test]
#[ignore]
fn test_reference_proof_for_sha256() {
    let (mut cs, _) = init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (base_setup, setup, vk, setup_tree, vars_hint, wits_hint) = cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let witness_set = cs.take_witness_using_hints(&worker, &vars_hint, &wits_hint);
    let _proof = cs.prove_cpu_basic::<GoldilocksExt2, DefaultTranscript, DefaultTreeHasher, NoPow>(
        &worker,
        witness_set,
        &base_setup,
        &setup,
        &setup_tree,
        &vk,
        prover_config,
        (),
    );
}

pub fn init_proof_cfg() -> ProofConfig {
    ProofConfig {
        fri_lde_factor: 2,
        pow_bits: 0,
        merkle_tree_cap_size: 32,
        ..Default::default()
    }
}

#[cfg(test)]
#[cfg(feature = "zksync")]
mod zksync {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    use crate::cs::{GpuSetup, PACKED_PLACEHOLDER_BITMASK};
    use crate::prover::gpu_prove_from_external_witness_data_with_cache_strategy;
    use boojum::cs::implementations::verifier::VerificationKey;
    use boojum::cs::implementations::{
        fast_serialization::MemcopySerializable, transcript::GoldilocksPoisedon2Transcript,
        verifier::Verifier,
    };
    use circuit_definitions::circuit_definitions::{
        aux_layer::{
            compression::{CompressionLayerCircuit, ProofCompressionFunction},
            compression_modes::{CompressionTranscriptForWrapper, CompressionTreeHasherForWrapper},
            CompressionProofsTreeHasher, CompressionProofsTreeHasherForWrapper,
            ZkSyncCompressionForWrapperCircuit, ZkSyncCompressionLayerCircuit,
            ZkSyncCompressionProof, ZkSyncCompressionProofForWrapper,
            ZkSyncCompressionVerificationKey, ZkSyncCompressionVerificationKeyForWrapper,
        },
        base_layer::ZkSyncBaseLayerCircuit,
        recursion_layer::{
            ZkSyncRecursionLayerProof, ZkSyncRecursionLayerVerificationKey, ZkSyncRecursionProof,
            ZkSyncRecursionVerificationKey,
        },
    };
    use era_cudart::memory::memory_get_info;
    use era_cudart_sys::CudaError;
    use prover::gpu_prove_from_external_witness_data_with_cache_strategy_inner;
    use serde::{Deserialize, Serialize};
    use synthesis_utils::synthesize_compression_circuit;

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(bound = "F: serde::Serialize + serde::de::DeserializeOwned")]
    pub struct GpuProverSetupData<H: GpuTreeHasher> {
        pub setup: GpuSetup<H>,
        #[serde(bound(
            serialize = "H::Output: serde::Serialize",
            deserialize = "H::Output: serde::de::DeserializeOwned"
        ))]
        pub vk: VerificationKey<F, H>,
        pub finalization_hint: FinalizationHintsForProver,
    }

    pub type ZksyncProof = Proof<F, DefaultTreeHasher, GoldilocksExt2>;
    type CompressionProofsTranscript = GoldilocksPoisedon2Transcript;

    const TEST_DATA_ROOT_DIR: &str = "./crates/shivini/test_data";
    const DEFAULT_CIRCUIT_INPUT: &str = "default.circuit";

    use crate::synthesis_utils::{
        init_cs_for_external_proving, init_or_synthesize_assembly, synth_circuit_for_proving,
        synth_circuit_for_setup, CircuitWrapper,
    };

    #[allow(dead_code)]
    pub type BaseLayerCircuit = ZkSyncBaseLayerCircuit;

    fn scan_directory<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
        let mut file_paths = vec![];
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                file_paths.push(path);
            }
        }
        file_paths.sort();

        file_paths
    }

    fn scan_directory_for_circuits<P: AsRef<Path>>(dir: P) -> Vec<CircuitWrapper> {
        let mut circuits = vec![];
        let file_paths = scan_directory(dir);
        for path in file_paths {
            let file_extension = path.extension().unwrap().to_string_lossy().to_string();
            if file_extension.contains("circuit") {
                let file = fs::File::open(path).unwrap();
                let circuit: CircuitWrapper = bincode::deserialize_from(file).expect("deserialize");
                circuits.push(circuit);
            }
        }

        circuits
    }

    #[allow(dead_code)]
    fn scan_directory_for_setups<P: AsRef<Path>>(dir: P) -> Vec<SetupBaseStorage<F, F>> {
        let mut circuits = vec![];
        let file_paths = scan_directory(dir);
        for path in file_paths {
            let file_extension = path.extension().unwrap().to_string_lossy().to_string();
            if file_extension.contains("setup") {
                let file = fs::File::open(path).unwrap();
                let circuit: SetupBaseStorage<F, F> =
                    bincode::deserialize_from(file).expect("deserialize");
                circuits.push(circuit);
            }
        }

        circuits
    }

    fn scan_directory_for_proofs<P: AsRef<Path>>(dir: P) -> Vec<ZksyncProof> {
        let mut proofs = vec![];
        let file_paths = scan_directory(dir);
        for path in file_paths {
            let file_extension = path.extension().unwrap().to_string_lossy().to_string();
            if file_extension.contains("proof") {
                let file = fs::File::open(path).unwrap();
                let proof: ZksyncProof = bincode::deserialize_from(file).expect("deserialize");
                proofs.push(proof);
            }
        }

        proofs
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_single_shot_zksync_setup_comparison() {
        let circuit = get_circuit_from_env();
        let _ctx = ProverContext::create().expect("gpu prover context");

        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = Worker::new();
        let (setup_cs, _) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();

        let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
            &worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );

        let _expected_permutation_polys = setup_base.copy_permutation_polys.clone();

        let _domain_size = setup_cs.max_trace_len;

        let expected_setup = GenericSetupStorage::from_host_values(&setup_base).unwrap();

        let (gpu_setup, _) = gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
            DefaultTreeHasher,
            _,
        >(setup_base, vk_params, vars_hint, wits_hint, &worker)
        .expect("gpu setup");

        let actual_setup = GenericSetupStorage::from_gpu_setup(&gpu_setup, &worker).unwrap();

        assert_eq!(
            expected_setup.inner.to_vec().unwrap(),
            actual_setup.inner.to_vec().unwrap(),
        );

        let expected_monomial = expected_setup.into_monomials().unwrap();
        let actual_monomial = actual_setup.into_monomials().unwrap();

        assert_eq!(
            expected_monomial.inner.to_vec().unwrap(),
            actual_monomial.inner.to_vec().unwrap(),
        );
    }

    #[serial]
    #[test]
    fn compare_proofs_for_all_zksync_circuits() -> CudaResult<()> {
        let worker = &Worker::new();
        let _ctx = ProverContext::create().expect("gpu prover context");

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("./crates/shivini/test_data/{}", main_dir);
            dbg!(&data_dir);
            let circuits = scan_directory_for_circuits(&data_dir);
            let reference_proofs = scan_directory_for_proofs(&data_dir);

            for (circuit, _reference_proof) in
                circuits.into_iter().zip(reference_proofs.into_iter())
            {
                let reference_proof_path =
                    format!("{}/{}.cpu.proof", data_dir, circuit.numeric_circuit_type());

                let reference_proof_path = Path::new(&reference_proof_path);

                let gpu_proof_path =
                    format!("{}/{}.gpu.proof", data_dir, circuit.numeric_circuit_type());

                let gpu_proof_path = Path::new(&gpu_proof_path);

                if reference_proof_path.exists() && gpu_proof_path.exists() {
                    continue;
                }

                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let proof_config = circuit.proof_config();

                let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
                let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
                    worker,
                    proof_config.fri_lde_factor,
                    proof_config.merkle_tree_cap_size,
                );

                let (gpu_setup, vk) =
                    gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
                        setup_base.clone(),
                        vk_params,
                        vars_hint.clone(),
                        wits_hint.clone(),
                        worker,
                    )?;

                println!("gpu proving");

                let gpu_proof = {
                    let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
                    let witness = proving_cs.witness.unwrap();
                    let config = GpuProofConfig::from_circuit_wrapper(&circuit);
                    gpu_prove_from_external_witness_data::<
                        DefaultTranscript,
                        DefaultTreeHasher,
                        NoPow,
                        Global,
                    >(
                        &config,
                        &witness,
                        proof_config.clone(),
                        &gpu_setup,
                        &vk,
                        (),
                        worker,
                    )
                    .expect("gpu proof")
                };

                let reference_proof_file = fs::File::open(reference_proof_path).unwrap();
                let reference_proof = bincode::deserialize_from(&reference_proof_file).unwrap();
                let actual_proof = gpu_proof.into();
                compare_proofs(&reference_proof, &actual_proof);
                assert!(
                    circuit.verify_proof::<DefaultTranscript, DefaultTreeHasher>(
                        (),
                        &vk,
                        &actual_proof
                    )
                );
                let proof_file = fs::File::create(gpu_proof_path).unwrap();

                bincode::serialize_into(proof_file, &actual_proof).expect("write proof into file");
            }
        }

        Ok(())
    }

    #[serial]
    #[test]
    #[ignore]
    fn generate_reference_proofs_for_all_zksync_circuits() {
        let worker = &Worker::new();

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("./crates/shivini/test_data/{}", main_dir);
            dbg!(&data_dir);
            let circuits = scan_directory_for_circuits(&data_dir);

            for circuit in circuits.into_iter() {
                if Path::new(&format!(
                    "{}/{}.cpu.proof",
                    data_dir,
                    circuit.numeric_circuit_type(),
                ))
                .exists()
                {
                    continue;
                }
                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let proof_config = circuit.proof_config();

                let (setup_cs, finalization_hint) =
                    init_or_synthesize_assembly::<SetupCSConfig, true>(circuit.clone(), None);
                let finalization_hint = finalization_hint.unwrap();
                let (setup_base, setup, vk, setup_tree, vars_hint, witness_hints) = setup_cs
                    .get_full_setup(
                        worker,
                        proof_config.fri_lde_factor,
                        proof_config.merkle_tree_cap_size,
                    );

                println!("reference proving");
                let reference_proof = {
                    let (proving_cs, _finalization_hint) =
                        init_or_synthesize_assembly::<ProvingCSConfig, true>(
                            circuit.clone(),
                            Some(&finalization_hint),
                        );
                    proving_cs.prove_from_precomputations::<EXT, DefaultTranscript, DefaultTreeHasher, NoPow>(
                        proof_config,
                        &setup_base,
                        &setup,
                        &setup_tree,
                        &vk,
                        &vars_hint,
                        &witness_hints,
                        (),
                        worker,
                    )
                };
                assert!(
                    circuit.verify_proof::<DefaultTranscript, DefaultTreeHasher>(
                        (),
                        &vk,
                        &reference_proof
                    )
                );
                let proof_file = fs::File::create(format!(
                    "{}/{}.cpu.proof",
                    data_dir,
                    circuit.numeric_circuit_type()
                ))
                .unwrap();

                bincode::serialize_into(proof_file, &reference_proof)
                    .expect("write proof into file");
            }
        }
    }

    fn load_scheduler_proof_and_vk() -> (ZkSyncRecursionProof, ZkSyncRecursionVerificationKey) {
        let scheduler_vk_file =
            fs::File::open("./test_data/compression/scheduler_recursive_vk.json").unwrap();
        let scheduler_vk: ZkSyncRecursionLayerVerificationKey =
            serde_json::from_reader(&scheduler_vk_file).unwrap();
        let scheduler_proof_file =
            fs::File::open("./test_data/compression/scheduler_recursive_proof.json").unwrap();
        let scheduler_proof: ZkSyncRecursionLayerProof =
            serde_json::from_reader(&scheduler_proof_file).unwrap();

        (scheduler_proof.into_inner(), scheduler_vk.into_inner())
    }

    #[test]
    #[ignore]
    fn run_make_compression_circuit_input() {
        let compression_wrapper_mode = 1;
        let (scheduler_proof, scheduler_vk) = load_scheduler_proof_and_vk();
        let circuit = CircuitWrapper::CompressionWrapper(
            ZkSyncCompressionForWrapperCircuit::from_witness_and_vk(
                Some(scheduler_proof),
                scheduler_vk,
                compression_wrapper_mode,
            ),
        );

        let circuit_file_path = format!(
            "./test_data/compression/compression_{}_wrapper.circuit",
            compression_wrapper_mode
        );
        let circuit_file = fs::File::create(&circuit_file_path).unwrap();
        bincode::serialize_into(&circuit_file, &circuit).unwrap();
        println!(
            "Compression wrapper {} circuit saved into {}",
            compression_wrapper_mode, circuit_file_path
        );
    }

    #[test]
    #[ignore]
    fn run_prove_compression_wrapper_circuit() {
        // Some Compression wrapper modes benefit from PoW
        // and underlying PoW function is defined as assoc type of the trait

        // type H = BNHasher;
        type H = CompressionTreeHasherForWrapper;
        type T = CompressionTranscriptForWrapper;

        let circuit = get_circuit_from_env();

        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();
        let proof_cfg = circuit.proof_config();
        println!("gpu proving");
        let (actual_proof, _) = prove_compression_wrapper_circuit(
            circuit.clone().into_compression_wrapper(),
            &mut None,
            worker,
        );

        println!("cpu proving");
        let reference_proof = {
            let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
            let (setup_base, setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs
                .get_full_setup(
                    worker,
                    proof_cfg.fri_lde_factor,
                    proof_cfg.merkle_tree_cap_size,
                );
            let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
            let proof = proving_cs.prove_from_precomputations::<EXT, T, H, NoPow>(
                proof_cfg.clone(),
                &setup_base,
                &setup,
                &setup_tree,
                &vk,
                &vars_hint,
                &wits_hint,
                (),
                worker,
            );
            let is_valid = circuit.verify_proof::<T, H>((), &vk, &proof);
            assert!(is_valid, "proof is invalid");
            proof
        };
        compare_proofs(&reference_proof, &actual_proof);
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
        name: &'static str,
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

    #[test]
    #[ignore]
    fn run_proof_compression_by_schedule() {
        let (scheduler_proof, scheduler_vk) = load_scheduler_proof_and_vk();
        compress_proof(scheduler_proof, scheduler_vk, CompressionSchedule::hard());
    }

    pub fn compress_proof(
        proof: ZkSyncRecursionProof,
        vk: ZkSyncRecursionVerificationKey,
        schedule: CompressionSchedule,
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
            let proof_file_path = format!(
                "./test_data/compression/compression_{}_proof.json",
                compression_mode as u8
            );
            let proof_file_path = Path::new(&proof_file_path);
            let vk_file_path = format!(
                "./test_data/compression/compression_{}_vk.json",
                compression_mode as u8
            );
            let vk_file_path = Path::new(&vk_file_path);
            let setup_data_file_path = format!(
                "./test_data/compression/compression_{}_setup_data.bin",
                compression_mode as u8
            );
            let setup_data_file_path = Path::new(&setup_data_file_path);
            if proof_file_path.exists() && vk_file_path.exists() {
                println!("Compression {compression_schedule_name}/{} proof and vk already exist ignoring", compression_mode as u8);
                let proof_file = fs::File::open(proof_file_path).unwrap();
                let input_proof = serde_json::from_reader(&proof_file).unwrap();
                let vk_file = fs::File::open(vk_file_path).unwrap();
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
            let mut setup_data = if setup_data_file_path.exists() {
                let bytes = fs::read(setup_data_file_path).unwrap();
                println!(
                    "Compression wrapper setup data for {compression_schedule_name}/{} loaded",
                    compression_mode as u8
                );
                Some(bincode::deserialize(&bytes).unwrap())
            } else {
                None
            };

            let compression_circuit = input.into_compression_circuit();
            let circuit_type = compression_circuit.numeric_circuit_type();
            println!(
                "Proving compression {compression_schedule_name}/{}",
                compression_mode as u8
            );
            let (output_proof, output_vk) = prove_compression_layer_circuit(
                compression_circuit.clone(),
                &mut setup_data,
                &worker,
            );
            println!(
                "Proof for compression {compression_schedule_name}/{} is generated!",
                compression_mode as u8
            );

            save_compression_proof_and_vk_into_file(&output_proof, &output_vk, circuit_type);

            if setup_data.is_some() {
                let bytes = bincode::serialize(&setup_data.unwrap()).unwrap();
                fs::write(setup_data_file_path, bytes).unwrap();
                println!(
                    "Compression wrapper setup data for {compression_schedule_name}/{} saved",
                    compression_mode as u8
                );
            }

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
            "./test_data/compression/compression_wrapper_{}_proof.json",
            last_compression_wrapping_mode as u8
        );
        let proof_file_path = Path::new(&proof_file_path);
        let vk_file_path = format!(
            "./test_data/compression/compression_wrapper_{}_vk.json",
            last_compression_wrapping_mode as u8
        );
        let vk_file_path = Path::new(&vk_file_path);
        let setup_data_file_path = format!(
            "./test_data/compression/compression_wrapper_{}_setup_data.bin",
            last_compression_wrapping_mode as u8
        );
        let setup_data_file_path = Path::new(&setup_data_file_path);
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
            let mut setup_data = if setup_data_file_path.exists() {
                let bytes = fs::read(setup_data_file_path).unwrap();
                println!(
                    "Compression wrapper setup data for {compression_schedule_name}/{} loaded",
                    last_compression_wrapping_mode as u8
                );
                Some(bincode::deserialize(&bytes).unwrap())
            } else {
                None
            };
            let compression_circuit = input.into_compression_wrapper_circuit();
            let (output_proof, output_vk) =
                prove_compression_wrapper_circuit(compression_circuit, &mut setup_data, &worker);
            println!(
                "Proof for compression wrapper {compression_schedule_name}/{} is generated!",
                last_compression_wrapping_mode as u8
            );
            save_compression_wrapper_proof_and_vk_into_file(
                &output_proof,
                &output_vk,
                last_compression_wrapping_mode as u8,
            );
            println!(
                "Compression wrapper proof and vk for {compression_schedule_name}/{} saved",
                last_compression_wrapping_mode as u8
            );
            if setup_data.is_some() {
                let bytes = bincode::serialize(&setup_data.unwrap()).unwrap();
                fs::write(setup_data_file_path, bytes).unwrap();
                println!(
                    "Compression wrapper setup data for {compression_schedule_name}/{} saved",
                    last_compression_wrapping_mode as u8
                );
            }
        }
    }

    pub fn save_compression_proof_and_vk_into_file(
        proof: &ZkSyncCompressionProof,
        vk: &ZkSyncCompressionVerificationKey,
        compression_mode: u8,
    ) {
        let proof_file = fs::File::create(format!(
            "./test_data/compression/compression_{}_proof.json",
            compression_mode
        ))
        .unwrap();
        serde_json::to_writer(proof_file, &proof).unwrap();
        let vk_file = fs::File::create(format!(
            "./test_data/compression/compression_{}_vk.json",
            compression_mode
        ))
        .unwrap();
        serde_json::to_writer(vk_file, &vk).unwrap();
    }

    pub fn save_compression_wrapper_proof_and_vk_into_file(
        proof: &ZkSyncCompressionProofForWrapper,
        vk: &ZkSyncCompressionVerificationKeyForWrapper,
        compression_mode: u8,
    ) {
        let proof_file = fs::File::create(format!(
            "./test_data/compression/compression_wrapper_{}_proof.json",
            compression_mode
        ))
        .unwrap();
        serde_json::to_writer(proof_file, &proof).unwrap();
        let vk_file = fs::File::create(format!(
            "./test_data/compression/compression_wrapper_{}_vk.json",
            compression_mode
        ))
        .unwrap();
        serde_json::to_writer(vk_file, &vk).unwrap();
    }

    pub fn prove_compression_layer_circuit(
        circuit: ZkSyncCompressionLayerCircuit,
        setup_data: &mut Option<GpuProverSetupData<CompressionProofsTreeHasher>>,
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
                    setup_data,
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
                    setup_data,
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
                    setup_data,
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
                    setup_data,
                    worker,
                );
                let is_proof_valid = verify_compression_layer_circuit(inner, &proof, &vk, verifier);
                (proof, vk, is_proof_valid)
            }
            ZkSyncCompressionLayerCircuit::CompressionMode5Circuit(_inner) => {
                unimplemented!()
            }
        };
        if !is_proof_valid {
            panic!("Proof is invalid");
        }

        (proof, vk)
    }

    pub fn prove_compression_wrapper_circuit(
        circuit: ZkSyncCompressionForWrapperCircuit,
        setup_data: &mut Option<GpuProverSetupData<CompressionTreeHasherForWrapper>>,
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
                    setup_data,
                    worker,
                );
                let is_proof_valid =
                    verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
                (proof, vk, is_proof_valid)
            }
            ZkSyncCompressionForWrapperCircuit::CompressionMode2Circuit(inner) => {
                let (proof, vk) = inner_prove_compression_wrapper_circuit(
                    inner.clone(),
                    proof_config,
                    gpu_proof_config,
                    setup_data,
                    worker,
                );
                let is_proof_valid =
                    verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
                (proof, vk, is_proof_valid)
            }
            ZkSyncCompressionForWrapperCircuit::CompressionMode3Circuit(inner) => {
                let (proof, vk) = inner_prove_compression_wrapper_circuit(
                    inner.clone(),
                    proof_config,
                    gpu_proof_config,
                    setup_data,
                    worker,
                );
                let is_proof_valid =
                    verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
                (proof, vk, is_proof_valid)
            }
            ZkSyncCompressionForWrapperCircuit::CompressionMode4Circuit(inner) => {
                let (proof, vk) = inner_prove_compression_wrapper_circuit(
                    inner.clone(),
                    proof_config,
                    gpu_proof_config,
                    setup_data,
                    worker,
                );
                let is_proof_valid =
                    verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
                (proof, vk, is_proof_valid)
            }
            ZkSyncCompressionForWrapperCircuit::CompressionMode5Circuit(inner) => {
                let (proof, vk) = inner_prove_compression_wrapper_circuit(
                    inner.clone(),
                    proof_config,
                    gpu_proof_config,
                    setup_data,
                    worker,
                );
                let is_proof_valid =
                    verify_compression_wrapper_circuit(inner, &proof, &vk, verifier);
                (proof, vk, is_proof_valid)
            }
        };
        if !is_proof_valid {
            panic!("Proof is invalid");
        }

        (proof, vk)
    }

    pub fn inner_prove_compression_layer_circuit<
        CF: ProofCompressionFunction<ThisLayerPoW: GPUPoWRunner>,
    >(
        circuit: CompressionLayerCircuit<CF>,
        proof_cfg: ProofConfig,
        gpu_cfg: GpuProofConfig,
        setup_data: &mut Option<GpuProverSetupData<CompressionProofsTreeHasher>>,
        worker: &Worker,
    ) -> (ZkSyncCompressionProof, ZkSyncCompressionVerificationKey) {
        let local_setup_data = match setup_data.take() {
            Some(setup_data) => setup_data,
            None => {
                let (setup_cs, finalization_hint) = synthesize_compression_circuit::<
                    _,
                    SetupCSConfig,
                    Global,
                >(circuit.clone(), true, None);
                let (base_setup, vk_params, variables_hint, witnesses_hint) = setup_cs
                    .get_light_setup(
                        worker,
                        proof_cfg.fri_lde_factor,
                        proof_cfg.merkle_tree_cap_size,
                    );
                let domain_size = vk_params.domain_size as usize;
                let config =
                    ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
                let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
                let (setup, vk) = gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
                    CompressionProofsTreeHasher,
                    _,
                >(
                    base_setup,
                    vk_params,
                    variables_hint,
                    witnesses_hint,
                    worker,
                )
                .unwrap();
                drop(ctx);
                let finalization_hint = finalization_hint.unwrap();
                GpuProverSetupData {
                    setup,
                    vk,
                    finalization_hint,
                }
            }
        };
        let (proving_cs, _) = synthesize_compression_circuit::<_, ProvingCSConfig, Global>(
            circuit.clone(),
            true,
            Some(&local_setup_data.finalization_hint),
        );
        let witness = proving_cs.witness.as_ref().unwrap();
        let cache_strategy = CacheStrategy {
            setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            commitment: CommitmentCacheStrategy::CacheCosetCaps,
        };
        let domain_size = local_setup_data.vk.fixed_parameters.domain_size as usize;
        let config =
            ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
        let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
        let gpu_proof = gpu_prove_from_external_witness_data_with_cache_strategy_inner::<
            CompressionProofsTranscript,
            CompressionProofsTreeHasher,
            CF::ThisLayerPoW,
            std::alloc::Global,
        >(
            &gpu_cfg,
            witness,
            proof_cfg.clone(),
            &local_setup_data.setup,
            &local_setup_data.vk,
            (),
            worker,
            cache_strategy,
        )
        .expect("gpu proof");
        drop(ctx);
        let proof = gpu_proof.into();
        let vk = local_setup_data.vk.clone();
        setup_data.replace(local_setup_data);
        (proof, vk)
    }

    pub fn inner_prove_compression_wrapper_circuit<
        CF: ProofCompressionFunction<ThisLayerPoW: GPUPoWRunner>,
    >(
        circuit: CompressionLayerCircuit<CF>,
        proof_cfg: ProofConfig,
        gpu_cfg: GpuProofConfig,
        setup_data: &mut Option<GpuProverSetupData<CompressionTreeHasherForWrapper>>,
        worker: &Worker,
    ) -> (
        ZkSyncCompressionProofForWrapper,
        ZkSyncCompressionVerificationKeyForWrapper,
    ) {
        let local_setup_data = match setup_data.take() {
            Some(setup_data) => setup_data,
            None => {
                let (setup_cs, finalization_hint) = synthesize_compression_circuit::<
                    _,
                    SetupCSConfig,
                    Global,
                >(circuit.clone(), true, None);
                let (base_setup, vk_params, variables_hint, witnesses_hint) = setup_cs
                    .get_light_setup(
                        worker,
                        proof_cfg.fri_lde_factor,
                        proof_cfg.merkle_tree_cap_size,
                    );
                let domain_size = vk_params.domain_size as usize;
                let config =
                    ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
                let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
                let (setup, vk) = gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
                    CompressionProofsTreeHasherForWrapper,
                    _,
                >(
                    base_setup,
                    vk_params,
                    variables_hint,
                    witnesses_hint,
                    worker,
                )
                .unwrap();
                drop(ctx);
                let finalization_hint = finalization_hint.unwrap();
                GpuProverSetupData {
                    setup,
                    vk,
                    finalization_hint,
                }
            }
        };
        let (proving_cs, _) = synthesize_compression_circuit::<_, ProvingCSConfig, Global>(
            circuit,
            true,
            Some(&local_setup_data.finalization_hint),
        );
        let witness = proving_cs.witness.as_ref().unwrap();
        let cache_strategy = CacheStrategy {
            setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            commitment: CommitmentCacheStrategy::CacheCosetCaps,
        };
        let domain_size = local_setup_data.vk.fixed_parameters.domain_size as usize;
        let config =
            ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
        let ctx = ProverContext::create_with_config(config).expect("gpu prover context");
        let gpu_proof = gpu_prove_from_external_witness_data_with_cache_strategy_inner::<
            CompressionTranscriptForWrapper,
            CompressionTreeHasherForWrapper,
            CF::ThisLayerPoW,
            std::alloc::Global
        >(
            &gpu_cfg,
            witness,
            proof_cfg.clone(),
            &local_setup_data.setup,
            &local_setup_data.vk,
            (),
            worker,
            cache_strategy,
        )
        .expect("gpu proof");
        drop(ctx);
        let vk = local_setup_data.vk.clone();
        setup_data.replace(local_setup_data);
        (gpu_proof.into(), vk)
    }

    pub fn verify_compression_layer_circuit<CF: ProofCompressionFunction>(
        _circuit: CompressionLayerCircuit<CF>,
        proof: &ZkSyncCompressionProof,
        vk: &ZkSyncCompressionVerificationKey,
        verifier: Verifier<F, EXT>,
    ) -> bool {
        verifier
            .verify::<CompressionProofsTreeHasher, CompressionProofsTranscript, CF::ThisLayerPoW>(
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

    #[serial]
    #[test]
    #[ignore]
    fn compare_proofs_for_single_zksync_circuit() {
        let circuit = get_circuit_from_env();
        let _ctx = ProverContext::create().expect("gpu prover context");

        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();

        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
            worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );

        println!(
            "trace length size 2^{}",
            setup_base.copy_permutation_polys[0]
                .domain_size()
                .trailing_zeros()
        );

        let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);

        let (gpu_setup, gpu_vk) =
            gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
                setup_base.clone(),
                vk.fixed_parameters.clone(),
                vars_hint.clone(),
                wits_hint.clone(),
                worker,
            )
            .expect("gpu setup");
        assert_eq!(vk, gpu_vk);

        println!("gpu proving");
        let gpu_proof = {
            let witness = proving_cs.witness.as_ref().unwrap();
            let config = GpuProofConfig::from_circuit_wrapper(&circuit);
            gpu_prove_from_external_witness_data::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                witness,
                proof_cfg.clone(),
                &gpu_setup,
                &vk,
                (),
                worker,
            )
            .expect("gpu proof")
        };

        println!("cpu proving");
        let reference_proof = {
            // we can't clone assembly lets synth it again
            let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
            proving_cs
                .prove_from_precomputations::<EXT, DefaultTranscript, DefaultTreeHasher, NoPow>(
                    proof_cfg.clone(),
                    &setup_base,
                    &setup,
                    &setup_tree,
                    &vk,
                    &vars_hint,
                    &wits_hint,
                    (),
                    worker,
                )
        };
        let start = std::time::Instant::now();
        let actual_proof = gpu_proof.into();
        println!("proof transformation takes {:?}", start.elapsed());
        circuit.verify_proof::<DefaultTranscript, DefaultTreeHasher>((), &vk, &actual_proof);
        compare_proofs(&reference_proof, &actual_proof);
    }

    #[serial]
    #[test]
    #[ignore]
    fn benchmark_single_circuit() {
        let circuit = get_circuit_from_env();
        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();
        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
            worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );
        let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
        let witness = proving_cs.witness.unwrap();
        let config = GpuProofConfig::from_circuit_wrapper(&circuit);
        let (gpu_setup, vk) = {
            let _ctx = ProverContext::create().expect("gpu prover context");
            gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
                setup_base.clone(),
                vk_params,
                vars_hint.clone(),
                wits_hint.clone(),
                worker,
            )
            .expect("gpu setup")
        };
        let proof_fn = || {
            let _ = gpu_prove_from_external_witness_data::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                &witness,
                proof_cfg.clone(),
                &gpu_setup,
                &vk,
                (),
                worker,
            )
            .expect("gpu proof");
        };
        loop {
            for i in 0..40 {
                let num_blocks = 2560 - i * 64;
                println!("num_blocks = {num_blocks}");
                let max_device_allocation =
                    (num_blocks * size_of::<F>()) << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
                let cfg = ProverContextConfig::default()
                    .with_maximum_device_allocation(max_device_allocation);
                let ctx = ProverContext::create_with_config(cfg).expect("gpu prover context");
                // technically not needed because CacheStrategy::get calls it internally,
                // but nice for peace of mind
                _setup_cache_reset();
                let strategy =
                    CacheStrategy::get::<DefaultTranscript, DefaultTreeHasher, NoPow, Global>(
                        &config,
                        &witness,
                        proof_cfg.clone(),
                        &gpu_setup,
                        &vk,
                        (),
                        worker,
                    );
                // technically not needed because CacheStrategy::get calls it internally,
                // but nice for peace of mind
                _setup_cache_reset();
                let strategy = match strategy {
                    Ok(s) => s,
                    Err(CudaError::ErrorMemoryAllocation) => {
                        println!("no cache strategy for {num_blocks}  found");
                        return;
                    }
                    Err(e) => panic!("unexpected error: {e}"),
                };
                println!("strategy: {:?}", strategy);
                println!("warmup with determined strategy");
                proof_fn();
                _setup_cache_reset();
                println!("first run");
                let start = std::time::Instant::now();
                proof_fn();
                println!("◆ total: {:?}", start.elapsed());
                println!("second run");
                let start = std::time::Instant::now();
                proof_fn();
                println!("◆ total: {:?}", start.elapsed());
                drop(ctx);
            }
        }
    }

    #[serial]
    #[test]
    #[ignore]
    fn profile_single_circuit() {
        let circuit = get_circuit_from_env();
        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();
        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
            worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );
        let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
        let witness = proving_cs.witness.unwrap();
        let reusable_cs = init_cs_for_external_proving(circuit.clone(), &finalization_hint);
        let config = GpuProofConfig::from_assembly(&reusable_cs);
        let (gpu_setup, vk) = {
            let _ctx = ProverContext::create().expect("gpu prover context");
            gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
                setup_base.clone(),
                vk_params,
                vars_hint.clone(),
                wits_hint.clone(),
                worker,
            )
            .expect("gpu setup")
        };
        let proof_fn = || {
            let _ = gpu_prove_from_external_witness_data::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                &witness,
                proof_cfg.clone(),
                &gpu_setup,
                &vk,
                (),
                worker,
            )
            .expect("gpu proof");
        };
        let ctx = ProverContext::create().expect("gpu prover context");
        println!("warmup");
        proof_fn();
        _setup_cache_reset();
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("test");
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("first run");
        println!("first run");
        let start = std::time::Instant::now();
        proof_fn();
        println!("◆ total: {:?}", start.elapsed());
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("second run");
        println!("second run");
        let start = std::time::Instant::now();
        proof_fn();
        println!("◆ total: {:?}", start.elapsed());
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("third run");
        println!("third run");
        let start = std::time::Instant::now();
        proof_fn();
        println!("◆ total: {:?}", start.elapsed());
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        drop(ctx);
        return;
    }

    #[serial]
    #[test]
    #[ignore]
    #[should_panic(expected = "placeholder found in a public input location")]
    fn test_public_input_placeholder_fail() {
        let (setup_cs, finalization_hint) =
            init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);
        let worker = Worker::new();
        let proof_config = init_proof_cfg();
        let (setup_base, vk_params, vars_hint, wits_hint) = setup_cs.get_light_setup(
            &worker,
            proof_config.fri_lde_factor,
            proof_config.merkle_tree_cap_size,
        );
        let domain_size = setup_cs.max_trace_len;
        let cfg = ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
        let _ctx = ProverContext::create_with_config(cfg).expect("init gpu prover context");
        let (proving_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(
            finalization_hint.as_ref(),
        );
        let mut witness = proving_cs.witness.as_ref().unwrap().clone();
        let (reusable_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, false>(
            finalization_hint.as_ref(),
        );
        let config = GpuProofConfig::from_assembly(&reusable_cs);
        let (mut gpu_setup, vk) =
            gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
                setup_base.clone(),
                vk_params,
                vars_hint.clone(),
                wits_hint.clone(),
                &worker,
            )
            .expect("gpu setup");
        witness.public_inputs_locations = vec![(0, 0)];
        gpu_setup.variables_hint[0][0] = PACKED_PLACEHOLDER_BITMASK;
        let _ = gpu_prove_from_external_witness_data::<
            DefaultTranscript,
            DefaultTreeHasher,
            NoPow,
            Global,
        >(
            &config,
            &witness,
            proof_config.clone(),
            &gpu_setup,
            &vk,
            (),
            &worker,
        )
        .expect("gpu proof");
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_reference_proof_for_circuit() {
        let circuit = get_circuit_from_env();
        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();

        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, setup, vk, setup_tree, vars_hint, witness_hints) = setup_cs
            .get_full_setup(
                worker,
                proof_cfg.fri_lde_factor,
                proof_cfg.merkle_tree_cap_size,
            );

        println!(
            "trace length size 2^{}",
            setup_base.copy_permutation_polys[0]
                .domain_size()
                .trailing_zeros()
        );

        let _proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
        let reference_proof = {
            // we can't clone assembly lets synth it again
            let mut proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
            let _witness_set =
                proving_cs.take_witness_using_hints(worker, &vars_hint, &witness_hints);
            proving_cs
                .prove_from_precomputations::<EXT, DefaultTranscript, DefaultTreeHasher, NoPow>(
                    proof_cfg.clone(),
                    &setup_base,
                    &setup,
                    &setup_tree,
                    &vk,
                    &vars_hint,
                    &witness_hints,
                    (),
                    worker,
                )
        };
        circuit.verify_proof::<DefaultTranscript, DefaultTreeHasher>((), &vk, &reference_proof);
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_generate_reference_setups_for_all_zksync_circuits() {
        let _worker = Worker::new();

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("./crates/shivini/test_data/{}", main_dir);
            let circuits = scan_directory_for_circuits(&data_dir);

            let worker = &Worker::new();
            for circuit in circuits {
                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let setup_file_path = format!(
                    "{}/{}.reference.setup",
                    data_dir,
                    circuit.numeric_circuit_type()
                );

                let (setup_cs, _finalization_hint) = synth_circuit_for_setup(circuit);
                let reference_base_setup = setup_cs.create_base_setup(worker, &mut ());

                let setup_file = fs::File::create(&setup_file_path).unwrap();
                reference_base_setup
                    .write_into_buffer(&setup_file)
                    .expect("write gpu setup into file");
                println!("Setup written into file {}", setup_file_path);
            }
        }
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_generate_gpu_setups_for_all_zksync_circuits() {
        let _worker = Worker::new();
        let _ctx = ProverContext::create().expect("gpu context");
        let worker = &Worker::new();

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("{}/{}", TEST_DATA_ROOT_DIR, main_dir);
            let circuits = scan_directory_for_circuits(&data_dir);

            for circuit in circuits {
                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let setup_file_path =
                    format!("{}/{}.gpu.setup", data_dir, circuit.numeric_circuit_type());

                let proof_cfg = circuit.proof_config();
                let (setup_cs, _finalization_hint) = synth_circuit_for_setup(circuit);
                let (setup_base, vk_params, variables_hint, witnesses_hint) = setup_cs
                    .get_light_setup(
                        worker,
                        proof_cfg.fri_lde_factor,
                        proof_cfg.merkle_tree_cap_size,
                    );

                let (gpu_setup, _) =
                    gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<DefaultTreeHasher, _>(
                        setup_base,
                        vk_params,
                        variables_hint,
                        witnesses_hint,
                        worker,
                    )
                    .expect("gpu setup");

                let setup_file = fs::File::create(&setup_file_path).unwrap();
                bincode::serialize_into(&setup_file, &gpu_setup).unwrap();
                println!("Setup written into file {}", setup_file_path);
            }
        }
    }

    fn get_circuit_from_env() -> CircuitWrapper {
        let circuit_file_path = if let Ok(circuit_file) = std::env::var("CIRCUIT_FILE") {
            circuit_file
        } else {
            std::env::args()
                // --circuit=/path/to/circuit prevents rebuilds
                .filter(|arg| arg.contains("--circuit"))
                .map(|arg| {
                    let parts: Vec<&str> = arg.splitn(2, '=').collect();
                    assert_eq!(parts.len(), 2);
                    let circuit_file_path = parts[1];
                    dbg!(circuit_file_path);
                    circuit_file_path.to_string()
                })
                .collect::<Vec<String>>()
                .pop()
                .unwrap_or(format!(
                    "./crates/shivini/test_data/{}",
                    DEFAULT_CIRCUIT_INPUT
                ))
        };

        let data = fs::read(circuit_file_path).expect("circuit file");
        bincode::deserialize(&data).expect("circuit")
    }

    #[serial]
    #[test]
    #[ignore]
    fn context_config_default() -> CudaResult<()> {
        const SLACK: usize = 1 << 26; // 64MB
        let (free_before, _) = memory_get_info()?;
        dbg!(free_before);
        let cfg = ProverContextConfig::default();
        let _ctx = ProverContext::create_with_config(cfg)?;
        let (free_after, _) = memory_get_info()?;
        dbg!(free_after);
        assert!(free_after < SLACK);
        Ok(())
    }

    #[serial]
    #[test]
    #[ignore]
    fn context_config_with_maximum_device_allocation() -> CudaResult<()> {
        const MAX: usize = 1 << 32; // 4GB
        const SLACK: usize = 1 << 26; // 64MB
        let (free_before, _) = memory_get_info()?;
        dbg!(free_before);
        let cfg = ProverContextConfig::default().with_maximum_device_allocation(MAX);
        let _ctx = ProverContext::create_with_config(cfg)?;
        let (free_after, _) = memory_get_info()?;
        dbg!(free_after);
        assert!(free_before - free_after > MAX);
        assert!(free_before - free_after < MAX + SLACK);
        Ok(())
    }

    #[serial]
    #[test]
    #[should_panic]
    #[ignore]
    fn context_config_with_minimum_device_allocation() {
        const SLACK: usize = 1 << 28; // 256MB
        let (free_before, _) = memory_get_info().unwrap();
        dbg!(free_before);
        let min = free_before + SLACK;
        let cfg = ProverContextConfig::default().with_minimum_device_allocation(min);
        let _ctx = ProverContext::create_with_config(cfg).unwrap();
    }
}
