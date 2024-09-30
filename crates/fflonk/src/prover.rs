use super::*;

use bellman::plonk::{
    better_better_cs::cs::{Circuit, SynthesisMode},
    commitments::transcript::{keccak_transcript::RollingKeccakTranscript, Transcript},
};
use fflonk::{
    commit_point_as_xy, compute_generators, horner_evaluation, FflonkAssembly, FflonkProof,
};

pub fn create_proof<E, C, S, T, CM, A>(
    assembly: &FflonkAssembly<E, S>,
    setup: &FflonkDeviceSetup<E, C, A>,
    worker: &bellman::worker::Worker,
) -> CudaResult<FflonkProof<E, C>>
where
    E: Engine,
    C: Circuit<E>,
    S: SynthesisMode,
    T: Transcript<E::Fr>,
    A: HostAllocator,
{
    assert!(S::PRODUCE_WITNESS);
    assert!(assembly.is_finalized);

    let domain_size = assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(
        domain_size.trailing_zeros() <= 23,
        "Only trace length up to 2^23 is allowed"
    );
    dbg!(domain_size);
    let mut context = None;
    if is_context_initialized() == false {
        context = Some(unsafe { DeviceContextWithSingleDevice::init(domain_size)? })
    }
    assert!(is_context_initialized());

    let mut transcript = T::new();

    // commit initial values
    let h_public_inputs = assembly.input_assingments.clone();
    assert!(h_public_inputs.is_empty() == false);
    for inp in h_public_inputs.iter() {
        transcript.commit_field_element(inp);
    }
    commit_point_as_xy::<E, _>(&mut transcript, &setup.c0_commitment);
    dbg!(setup.c0_commitment);

    let stream = bc_stream::new().unwrap();

    // Allocate same length buffer to prevent fragmentation
    let common_combined_degree = MAX_COMBINED_DEGREE_FACTOR * domain_size;
    let device = Device::model();
    let mut combined_monomial_storage =
        GenericCombinedStorage::<E::Fr, _>::allocate_on(&device, domain_size)?;
    let start = std::time::Instant::now();
    let ([c1_commitment, c2_commitment], h_all_evaluations, h_aux_evaluations, challenges) =
        prove_statements(
            &assembly,
            &setup,
            &mut combined_monomial_storage,
            &mut transcript,
            domain_size,
            common_combined_degree,
            stream,
            worker,
        )?;
    println!("Statement are proven in {} s ", start.elapsed().as_secs());
    let start = std::time::Instant::now();
    let ([w_commitment, w_prime_commitment], h_lagrange_basis_inverses) = prove_openings::<E, _, _>(
        &mut combined_monomial_storage,
        &mut transcript,
        h_all_evaluations.clone(),
        h_aux_evaluations.clone(),
        challenges,
        domain_size,
        common_combined_degree,
        stream,
    )?;
    println!("Openings are proven in {} s ", start.elapsed().as_secs());
    // make proof
    let mut proof = FflonkProof::empty();
    proof.inputs = h_public_inputs;
    proof.commitments = vec![
        c1_commitment,
        c2_commitment,
        w_commitment,
        w_prime_commitment,
    ];
    proof.evaluations = h_all_evaluations;
    proof.lagrange_basis_inverses = h_lagrange_basis_inverses;

    stream.sync().unwrap();
    Ok(proof)
}

pub fn prove_statements<E, C, S, T, CM, A>(
    assembly: &FflonkAssembly<E, S>,
    setup: &FflonkDeviceSetup<E, C, A>,
    combined_monomial_stoarge: &mut CM,
    transcript: &mut T,
    domain_size: usize,
    common_combined_degree: usize,
    stream: bc_stream,
    worker: &bellman::worker::Worker,
) -> CudaResult<([E::G1Affine; 2], Vec<E::Fr>, Vec<E::Fr>, [E::Fr; 3])>
where
    E: Engine,
    C: Circuit<E>,
    S: SynthesisMode,
    T: Transcript<E::Fr>,
    CM: CombinedMonomialStorage<Poly = Poly<E::Fr, MonomialBasis>>,
    A: HostAllocator,
{
    // take witnesses
    let trace_monomials = unsafe {
        load_trace_from_precomputations(assembly, worker, domain_size, stream).expect("load trace")
    };
    // load main gate selectors
    let main_gate_selectors_monomial = unsafe {
        load_main_gate_selectors(&setup.main_gate_selector_monomials, domain_size, stream)?
    };

    // compute main gate quotient chunk by chunk
    let main_gate_quotient_degree = 2;
    let mut main_gate_quotient_lde_bitreversed =
        Poly::<_, LDE>::zero(main_gate_quotient_degree * domain_size);

    let public_inputs = DScalars::from_host_scalars_on(&assembly.input_assingments, stream)?;
    for coset_idx in 0..main_gate_quotient_degree {
        let quotient_sum = evaluate_main_gate_constraints(
            coset_idx,
            domain_size,
            trace_monomials.iter(),
            main_gate_selectors_monomial.iter(),
            &public_inputs,
            stream,
        )?;
        let start = coset_idx * domain_size;
        let end = start + domain_size;
        mem::d2d_on(
            quotient_sum.as_ref(),
            &mut main_gate_quotient_lde_bitreversed.as_mut()[start..end],
            stream,
        )?;
    }
    let mut main_gate_quotient_lde = main_gate_quotient_lde_bitreversed;
    main_gate_quotient_lde.bitreverse(stream)?;
    let main_gate_quotient_monomial = main_gate_quotient_lde.coset_ifft_on(stream)?;
    if SANITY_CHECK {
        let leading_coeffs = main_gate_quotient_monomial.as_ref()
            [main_gate_quotient_degree * (domain_size - 1)..]
            .to_vec_on(stream)?;
        leading_coeffs
            .iter()
            .rev()
            .enumerate()
            .for_each(|(idx, coeff)| assert!(coeff.is_zero(), "{idx}-th coeff should be zero"));
    }
    // combine monomials into c1 combined monomial
    // max degree combined poly is c2 where deg(c2) = 9(n-1) + 12
    let num_first_round_polys = 4;
    let mut c1_monomial = Poly::<_, MonomialBasis>::zero(common_combined_degree);
    combine_monomials(
        trace_monomials
            .iter()
            .chain(std::iter::once(&main_gate_quotient_monomial)),
        &mut c1_monomial,
        num_first_round_polys,
        stream,
    )?;

    // commit to the c1(x)
    let c1_commitment = commit_monomial::<E>(&c1_monomial, domain_size, stream)?;
    stream.sync().unwrap();
    dbg!(c1_commitment);

    // commit commitment into transcript
    commit_point_as_xy::<E, T>(transcript, &c1_commitment);
    combined_monomial_stoarge.write(1, c1_monomial, stream)?;

    let h_beta = transcript.get_challenge();
    let h_gamma = transcript.get_challenge();
    dbg!(h_beta);
    dbg!(h_gamma);

    let beta = DScalar::from_host_value_on(&h_beta, stream)?;
    let gamma = DScalar::from_host_value_on(&h_gamma, stream)?;

    let h_non_residues = bellman::plonk::better_cs::generator::make_non_residues::<E::Fr>(2);
    // let non_residues = DVec::from_host_slice_on(&h_non_residues, this_stage_mempool, stream)?;
    // compute permutation polynomials
    // let permutation_monomials = unsafe {
    //     materialize_permutation_polys(&variable_indexes, &non_residues, domain_size, stream)?
    // };
    let permutation_monomials =
        unsafe { load_permutation_monomials(&setup.permutation_monomials, domain_size, stream)? };
    // compute product of scalars
    let mut h_non_residues_by_beta = vec![h_beta];
    for mut non_residue in h_non_residues.iter().cloned() {
        non_residue.mul_assign(&h_beta);
        h_non_residues_by_beta.push(non_residue);
    }
    let non_residues_by_beta = DScalars::from_host_scalars_on(&h_non_residues_by_beta, stream)?;

    // compute copy-permutation grand product
    let copy_perm_grand_prod_monomial = compute_copy_perm_grand_product(
        domain_size,
        trace_monomials.iter(),
        permutation_monomials.iter(),
        &non_residues_by_beta,
        &beta,
        &gamma,
        stream,
    )?;

    // compute quotients of copy-permutation
    // first quotient is the grand product relationship
    let copy_perm_quotient_degree = 3;
    // underlying ntt requires problem size to be power of two
    let padded_copy_perm_quotient_degree = 4;
    let mut copy_perm_first_quotient =
        Poly::<_, LDE>::zero(padded_copy_perm_quotient_degree * domain_size);

    for coset_idx in 0..padded_copy_perm_quotient_degree {
        let first_quotient_sum = evaluate_copy_permutation_constraint(
            domain_size,
            coset_idx,
            trace_monomials.iter(),
            permutation_monomials.iter(),
            &copy_perm_grand_prod_monomial,
            &non_residues_by_beta,
            &beta,
            &gamma,
            stream,
        )?;
        let start = coset_idx * domain_size;
        let end = start + domain_size;
        mem::d2d_on(
            first_quotient_sum.as_ref(),
            &mut copy_perm_first_quotient.as_mut()[start..end],
            stream,
        )?;
    }
    copy_perm_first_quotient.bitreverse(stream)?;
    let copy_perm_first_quotient_monomial = copy_perm_first_quotient.coset_ifft_on(stream)?;
    let copy_perm_first_quotient_monomial = copy_perm_first_quotient_monomial
        .trim_to_degree(copy_perm_quotient_degree * domain_size, stream)?;
    if SANITY_CHECK {
        let leading_coeffs = copy_perm_first_quotient_monomial.as_ref()
            [copy_perm_quotient_degree * (domain_size - 1)..]
            .to_vec_on(stream)?;
        leading_coeffs
            .iter()
            .enumerate()
            .for_each(|(idx, coeff)| assert!(coeff.is_zero(), "{idx}-th coeff should be zero"));
    }
    // second quotient is the first element of the grand product
    let copy_perm_second_quotient_monomial =
        prove_copy_perm_second_constraint(&copy_perm_grand_prod_monomial, domain_size, stream)?;
    if SANITY_CHECK {
        let leading_coeffs =
            copy_perm_second_quotient_monomial.as_ref()[domain_size - 1..].to_vec_on(stream)?;
        leading_coeffs
            .iter()
            .enumerate()
            .for_each(|(idx, coeff)| assert!(coeff.is_zero(), "{idx}-th coeff should be zero"));
    }
    // combine monomials into c2 combined monomial
    let mut c2_monomial = Poly::<_, MonomialBasis>::zero(common_combined_degree);
    combine_monomials(
        [
            &copy_perm_grand_prod_monomial,
            &copy_perm_first_quotient_monomial,
            &copy_perm_second_quotient_monomial,
        ]
        .into_iter(),
        &mut c2_monomial,
        3,
        stream,
    )?;

    // commit to the c2(x)
    let c2_commitment = commit_monomial::<E>(&c2_monomial, domain_size, stream)?;
    dbg!(c2_commitment);
    // commit commitment into transcript
    commit_point_as_xy::<E, T>(transcript, &c2_commitment);
    combined_monomial_stoarge.write(2, c2_monomial, stream)?;

    // get evaluation challenge
    let power = 24;
    let h_r = transcript.get_challenge();
    dbg!(h_r);

    let h_z = h_r.pow(&[power as u64]);
    dbg!(h_z);
    let z = DScalar::from_host_value_on(&h_z, stream)?;

    let mut h_z_omega = h_z.clone();
    let omega = bellman::plonk::domains::Domain::new_for_size(domain_size as u64)
        .unwrap()
        .generator;
    h_z_omega.mul_assign(&omega);
    let z_omega = DScalar::from_host_value_on(&h_z_omega, stream)?;
    dbg!(h_z_omega);

    // compute all evaluations
    // the verifier will re-compute evaluations of the quotients
    // by herself from existing evaluations, however we still need to
    // evaluate quotients of the copy-perm at z*w

    let num_evals_at_z = 8 + 3 + 1;
    let num_evals_at_z_omega = 1 + 2;
    let num_all_evaluations = num_evals_at_z + num_evals_at_z_omega;
    let mut all_evaluations = DScalars::allocate_zeroed_on(num_all_evaluations, stream)?;
    let (evaluations_at_z, evaluations_at_z_omega) = all_evaluations.split_at_mut(num_evals_at_z);
    for (monomial, value) in main_gate_selectors_monomial
        .iter()
        .chain(permutation_monomials.iter())
        .chain(trace_monomials.iter())
        .chain([&copy_perm_grand_prod_monomial])
        .zip(evaluations_at_z.iter_mut())
    {
        stream.sync().unwrap();
        monomial.evaluate_at_into_on(&z, value, stream)?;
    }
    assert_eq!(evaluations_at_z_omega.len(), 3);
    for (monomial, value) in [
        &copy_perm_grand_prod_monomial,
        &copy_perm_first_quotient_monomial,
        &copy_perm_second_quotient_monomial,
    ]
    .into_iter()
    .zip(evaluations_at_z_omega.iter_mut())
    {
        monomial.evaluate_at_into_on(&z_omega, value, stream)?;
    }

    let num_aux_evaluations = 1 + 2;
    let mut aux_evaluations = DScalars::allocate_zeroed_on(num_aux_evaluations, stream)?;

    for (monomial, value) in [
        &main_gate_quotient_monomial,
        &copy_perm_first_quotient_monomial,
        &copy_perm_second_quotient_monomial,
    ]
    .into_iter()
    .zip(aux_evaluations.iter_mut())
    {
        monomial.evaluate_at_into_on(&z, value, stream)?;
    }

    // commit evaluations into transcript
    let h_all_evaluations = all_evaluations.to_host_scalars_on(stream)?;

    h_all_evaluations
        .iter()
        .for_each(|el| transcript.commit_field_element(el));

    let h_aux_evaluations = aux_evaluations.to_host_scalars_on(stream)?;

    let mut c0_monomial = Poly::<_, MonomialBasis>::zero(common_combined_degree);
    combine_monomials(
        main_gate_selectors_monomial
            .iter()
            .chain(permutation_monomials.iter()),
        &mut c0_monomial,
        8,
        stream,
    )?;

    combined_monomial_stoarge.write(0, c0_monomial, stream)?;

    stream.sync().unwrap();

    Ok((
        [c1_commitment, c2_commitment],
        h_all_evaluations,
        h_aux_evaluations,
        [h_r, h_z, h_z_omega],
    ))
}

// our fflonk implementation commits to the polynomials that each has
// degree of 9*2^23=2.42GB
// MSM of same size alloctes twice as memory ~5GB

// Cuda has following available devices:
// T4 has total 16GB memory, 11GB for the proving
// L4 has 24GB memory, 21GB for the proving
// A100 has both 40GB and 80GB models,

// If we decide to do divisons on the opening protocol on the device,
// then memory becomes bottleneck in case of T4 devices.
// Simplest obvious solution for such models is to transfer combined monomials to the host
// in a async way then load them back to the device sequantially each time
// Obviously it will affect proof generation time but it is still much better
// than the case where we do divison on the host.
pub fn prove_openings<E, T, CM>(
    combined_monomial_stoarge: &mut CM,
    transcript: &mut T,
    h_all_evaluations: Vec<E::Fr>,
    h_aux_evaluations: Vec<E::Fr>,
    previous_round_challenges: [E::Fr; 3],
    domain_size: usize,
    common_combined_degree: usize,
    stream: bc_stream,
) -> CudaResult<([E::G1Affine; 2], Vec<E::Fr>)>
where
    E: Engine,
    T: Transcript<E::Fr>,
    CM: CombinedMonomialStorage<Poly = Poly<E::Fr, MonomialBasis>>,
{
    // get linearization challenge
    let h_alpha = transcript.get_challenge();
    dbg!(h_alpha);
    let mut h_alpha_pows = [h_alpha, h_alpha];
    h_alpha_pows[1].mul_assign(&h_alpha);
    let alpha_pows = DScalars::from_host_scalars_on(&h_alpha_pows, stream)?;

    let [h_r, h_z, h_z_omega] = previous_round_challenges;
    let power = 24;
    let opening_points = compute_opening_points(h_r, h_z, h_z_omega, power, domain_size);

    dbg!(&opening_points);

    // construct r_i(x)  monomials from evaluations
    let [h_r0_monomial, h_r1_monomial, h_r2_monomial] = construct_r_monomials(
        h_all_evaluations.clone(),
        h_aux_evaluations.clone(),
        opening_points,
    );

    let mut r0_monomial = Poly::allocate_on(h_r0_monomial.len(), _small_scalar_mempool(), stream);
    mem::h2d_on(&h_r0_monomial, r0_monomial.as_mut(), stream)?;

    let mut r1_monomial = Poly::allocate_on(h_r1_monomial.len(), _small_scalar_mempool(), stream);
    mem::h2d_on(&h_r1_monomial, r1_monomial.as_mut(), stream)?;

    let mut r2_monomial = Poly::allocate_on(h_r2_monomial.len(), _small_scalar_mempool(), stream);
    mem::h2d_on(&h_r2_monomial, r2_monomial.as_mut(), stream)?;

    if SANITY_CHECK {
        let [setup_omega, trace_omega, copy_perm_omega] = compute_generators::<E::Fr>(8, 4, 3);
        let mut expected = DScalar::zero(stream)?;
        let mut opening_points_iter = opening_points.into_iter();
        let mut h_current = opening_points_iter.next().unwrap();
        let mut tmp = Poly::zero(common_combined_degree);
        combined_monomial_stoarge.read_into(0, &mut tmp, stream)?;
        for _ in 0..8 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            tmp.evaluate_at_into_on(&current, &mut expected, stream)?;
            stream.sync().unwrap();
            let h_actual = horner_evaluation(&h_r0_monomial, h_current);
            assert_eq!(expected.to_host_value_on(stream)?, h_actual,);
            h_current.mul_assign(&setup_omega);
        }
        let mut h_current = opening_points_iter.next().unwrap();
        combined_monomial_stoarge.read_into(1, &mut tmp, stream)?;
        for _ in 0..4 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            tmp.evaluate_at_into_on(&current, &mut expected, stream)?;
            stream.sync().unwrap();
            let h_actual = horner_evaluation(&h_r1_monomial, h_current);
            assert_eq!(expected.to_host_value_on(stream)?, h_actual,);
            h_current.mul_assign(&trace_omega);
        }

        let mut h_current = opening_points_iter.next().unwrap();
        combined_monomial_stoarge.read_into(2, &mut tmp, stream)?;
        for _ in 0..3 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            tmp.evaluate_at_into_on(&current, &mut expected, stream)?;
            stream.sync().unwrap();
            let h_actual = horner_evaluation(&h_r2_monomial, h_current);
            assert_eq!(expected.to_host_value_on(stream)?, h_actual,);
            h_current.mul_assign(&copy_perm_omega);
        }

        let mut h_current = opening_points_iter.next().unwrap();
        for _ in 0..3 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            tmp.evaluate_at_into_on(&current, &mut expected, stream)?;
            stream.sync().unwrap();
            let h_actual = horner_evaluation(&h_r2_monomial, h_current);
            assert_eq!(expected.to_host_value_on(stream)?, h_actual,);
            h_current.mul_assign(&copy_perm_omega);
        }
        stream.sync().unwrap();
    }
    // re-construct combined polynomial of the preprocessing parts
    let r_monomials = [r0_monomial, r1_monomial, r2_monomial];

    // compute opening proof W(x) in monomial  W(x) = f(x) / Z(x)
    let w_monomial = compute_w_monomial(
        combined_monomial_stoarge,
        domain_size,
        &r_monomials,
        h_z,
        h_z_omega,
        opening_points,
        &alpha_pows,
        stream,
    )?;
    // commit W(x) commitment into transcript
    let w_commitment =
        commit_padded_monomial::<E>(&w_monomial, common_combined_degree, domain_size, stream)?;
    commit_point_as_xy::<E, T>(transcript, &w_commitment);
    dbg!(w_commitment);

    // get challenge
    let h_y = transcript.get_challenge();
    dbg!(h_y);

    // evaluate r monomials at challenge point
    let mut r_evals_at_y = [
        DScalar::zero(stream)?,
        DScalar::zero(stream)?,
        DScalar::zero(stream)?,
    ];

    for (monomial, value) in [h_r0_monomial, h_r1_monomial, h_r2_monomial]
        .into_iter()
        .zip(r_evals_at_y.iter_mut())
    {
        let sum = fflonk::utils::horner_evaluation(&monomial, h_y);
        value.copy_from_host_value_on(&sum, stream)?;
    }

    // compute linearization W'(x) = L(x) / (x-y)
    let w_prime_monomial = compute_w_prime_monomial(
        w_monomial,
        combined_monomial_stoarge,
        domain_size,
        common_combined_degree,
        r_evals_at_y,
        h_z,
        h_z_omega,
        h_y,
        h_alpha_pows,
        stream,
    )?;

    let w_prime_commitment = commit_padded_monomial::<E>(
        &w_prime_monomial,
        common_combined_degree,
        domain_size,
        stream,
    )?;
    stream.sync().unwrap();
    // mempool.destroy().unwrap();

    dbg!(w_prime_commitment);

    let h_lagrange_basis_inverses = compute_flattened_lagrange_basis_inverses(opening_points, h_y);
    Ok((
        [w_commitment, w_prime_commitment],
        h_lagrange_basis_inverses,
    ))
}

// TODO: bit representation of the selectors
pub unsafe fn load_main_gate_selectors<F: PrimeField, A: HostAllocator>(
    h_main_gate_selector_monomials: &[Vec<F, A>; 5],
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<MainGateSelectors<F>> {
    let mut selectors_monomial = MainGateSelectors::allocate_zeroed(domain_size);

    for (src_col, dst_col) in h_main_gate_selector_monomials
        .iter()
        .zip(selectors_monomial.iter_mut())
    {
        // load host values first
        // TODO overlap
        assert_eq!(src_col.len(), dst_col.size());
        mem::h2d_on(src_col, dst_col.as_mut(), stream)?;
    }

    Ok(selectors_monomial)
}

// TODO load indexes and keep permutation polynomials in Lagrange basis
pub unsafe fn load_permutation_monomials<F: PrimeField, A: HostAllocator>(
    h_permutation_monomials: &[Vec<F, A>; 3],
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<Permutations<F>> {
    let mut permutation_monomials = Permutations::allocate_zeroed(domain_size);

    for (src_col, dst_col) in h_permutation_monomials
        .iter()
        .zip(permutation_monomials.iter_mut())
    {
        // load host values first
        // let event = bc_event::new().unwrap();
        assert_eq!(src_col.len(), dst_col.size());
        mem::h2d_on(src_col, dst_col.as_mut(), stream)?;
        // event.sync().unwrap();
        // stream.wait(event).unwrap();
    }

    Ok(permutation_monomials)
}

pub unsafe fn load_trace_from_precomputations<E: Engine, S: SynthesisMode>(
    assembly: &FflonkAssembly<E, S>,
    worker: &bellman::worker::Worker,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<Trace<E::Fr>> {
    let mut trace_monomial = Trace::allocate_zeroed(domain_size);

    // let (raw_trace_cols, _) = assembly
    //     .make_state_and_witness_polynomials(worker, true)
    //     .unwrap();
    let src_trace_cols = assembly.make_assembled_poly_storage(worker, true).unwrap();

    // let h2d_stream = bc_stream::new().unwrap();
    for (idx, dst) in trace_monomial.iter_mut().enumerate() {
        let idx = bellman::plonk::better_better_cs::cs::PolyIdentifier::VariablesPolynomial(idx);
        let src = src_trace_cols.get_poly(idx);
        // let event = bc_event::new().unwrap();
        // event.record(h2d_stream)?;
        // event.sync()?;
        mem::h2d_on(
            src.as_ref().as_ref(),
            &mut dst.as_mut()[..domain_size - 1],
            stream,
        )?;

        let h_dst = dst.as_ref().to_vec_on(stream).unwrap();
        stream.sync().unwrap();
        assert_eq!(src.as_ref().as_ref(), &h_dst[..domain_size - 1]);
        assert_eq!(h_dst.last().unwrap(), &E::Fr::zero());
        ntt::inplace_ifft_on(dst.as_mut(), stream)?;
    }

    Ok(trace_monomial)
}

pub unsafe fn load_witnesses_and_assign_values<F: PrimeField, A: Allocator>(
    h_input_assignments: &Vec<F>,
    h_aux_assignments: &Vec<F, A>,
    h_indexes: &[Vec<u32, A>; 3],
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<(Trace<F>, DVec<u32>)> {
    let mut trace_monomial = Trace::allocate_zeroed(domain_size);
    // same indexes will be used in materializing permutation polys and it requires
    // it to be same length with trace
    let mut indexes = DVec::allocate_zeroed(3 * domain_size);
    // transfer flattened witness values
    let num_aux_variables = h_aux_assignments.len();
    // assignments should have dummy value as first element
    let total_num_variables = num_aux_variables + h_input_assignments.len() + 1;
    let mut full_assignments = DVec::allocate_zeroed(total_num_variables);
    let (aux_assignments, input_assignments) =
        full_assignments[1..].split_at_mut(num_aux_variables);
    mem::h2d_on(&h_aux_assignments, aux_assignments, stream)?;
    // place input assignments right after aux assignments
    mem::h2d_on(&h_input_assignments, input_assignments, stream)?;
    // The setup already merges variables of both public input gates and main gate variables
    // assign values column by column and benefit from overlapping
    for ((h_index_col, index_col), trace_col) in h_indexes
        .iter()
        .zip(indexes.chunks_mut(domain_size))
        .zip(trace_monomial.iter_mut())
    {
        // load indexes first
        let event = bc_event::new().unwrap();
        let num_rows = h_index_col.len();
        assert!(h_index_col.len() <= index_col.len());
        mem::h2d_on(h_index_col, &mut index_col[..num_rows], stream)?;
        event.sync().unwrap();
        stream.wait(event).unwrap();
        variable_assignment_for_single_col(
            &full_assignments,
            index_col,
            trace_col.as_mut(),
            num_rows,
            stream,
        )?;
        // convert column into monomial basis
        ntt::inplace_ifft_on(trace_col.as_mut(), stream).expect("inplace ifft");
    }

    Ok((trace_monomial, indexes))
}

pub fn variable_assignment_for_single_col<F: PrimeField>(
    assignments: &DSlice<F>,
    indexes: &DSlice<u32>,
    trace_col: &mut DSlice<F>,
    num_rows: usize,
    stream: bc_stream,
) -> CudaResult<()> {
    unsafe {
        let src_ptr = assignments.as_ptr();
        let dst_ptr = trace_col.as_mut_ptr();
        let indexes_ptr = indexes.as_ptr();
        let result = gpu_ffi::ff_select(
            src_ptr.cast(),
            dst_ptr.cast(),
            indexes_ptr.cast(),
            num_rows as u32,
            stream,
        );
        if result != 0 {
            return Err(CudaError::VariableAssignmentError(result.to_string()));
        }

        Ok(())
    }
}

pub unsafe fn materialize_permutation_polys<F: PrimeField>(
    indexes: &DVec<u32>,
    non_residues: &DVec<F>,
    domain_size: usize,
    pool: bc_mem_pool,
    stream: bc_stream,
) -> CudaResult<Permutations<F>> {
    use gpu_ffi::generate_permutation_polynomials_configuration;
    let mut permutations_monomial = Permutations::<F>::allocate_zeroed(domain_size);
    let num_cols = permutations_monomial.num_polys();
    assert_eq!(num_cols * domain_size, indexes.len());
    assert_eq!(non_residues.len(), num_cols);

    let log_rows_count = domain_size.trailing_zeros();

    let indexes_ptr = indexes.as_ptr() as *mut usize;
    let non_residues_ptr = non_residues.as_ptr() as *mut F;

    let permutations_monomial_ptr = permutations_monomial.as_mut_ptr();
    let cfg = generate_permutation_polynomials_configuration {
        mem_pool: pool, // TODO: make sure it is tmp pool
        stream,
        indexes: indexes_ptr.cast(),
        scalars: non_residues_ptr.cast(),
        target: permutations_monomial_ptr.cast(),
        columns_count: num_cols as u32,
        log_rows_count,
    };

    unsafe {
        let result = gpu_ffi::pn_generate_permutation_polynomials(cfg);
        if result != 0 {
            return Err(CudaError::MaterializePermutationsError(result.to_string()));
        }
    }

    Ok(permutations_monomial)
}

pub fn compute_w_monomial<F, CM>(
    combined_monomial_storage: &mut CM,
    domain_size: usize,
    r_monomials: &[Poly<F, MonomialBasis, PoolAllocator>; 3],
    h_z: F,
    h_z_omega: F,
    opening_points: [F; 4],
    alpha_pows: &DScalars<F>,
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
    CM: CombinedMonomialStorage<Poly = Poly<F, MonomialBasis>>,
{
    assert_eq!(alpha_pows.len(), 2);
    let [
        sparse_polys_for_setup, // Z_{T\S0}(x) deg = 10
        sparse_polys_for_trace, // Z_{T\S1}(x) deg = 14
        sparse_polys_for_copy_perm,// Z_{T\S2}(x) deg = 12
        sparse_polys,// Z_T(x) = 18
    ] = construct_set_difference_monomials(
        h_z,
        h_z_omega,
    );

    let [r0, r1, r2] = r_monomials;

    let mut alpha_pows_iter = alpha_pows.iter();
    let alpha = &alpha_pows_iter.next().unwrap();
    let alpha_squared = &alpha_pows_iter.next().unwrap();

    // f(x) = Z_T\S0(x)(C0(x) - r0(x)) + alpha*Z_T\S1(x)(C1(x) - r1(x)) + alpha^2*Z_T\S2(x)(C2(x) - r2(x))
    // W(x) = f(x)/Z_T(x)

    // Aggregate parts

    // Loading all combined polys subsequently and keeping them
    // costs 30*N here
    let mut w_monomial_proto = Poly::zero(10 * domain_size);
    combined_monomial_storage.read_into(0, &mut w_monomial_proto, stream)?;

    let mut c1 = Poly::zero(10 * domain_size);
    combined_monomial_storage.read_into(1, &mut c1, stream)?;

    let mut c2 = Poly::zero(10 * domain_size);
    combined_monomial_storage.read_into(2, &mut c2, stream)?;

    arithmetic::sub_assign(
        &mut w_monomial_proto.as_mut()[..r0.size()],
        r0.as_ref(),
        stream,
    )?;
    let c0_degree = 8 * domain_size;
    multiply_monomial_with_multiple_sparse_polys_inplace(
        &mut w_monomial_proto,
        sparse_polys_for_setup,
        c0_degree,
        stream,
    )?;

    for (((mut c, r), sparse_poly), alpha) in [c1, c2]
        .into_iter()
        .zip([r1, r2].into_iter())
        .zip([sparse_polys_for_trace, sparse_polys_for_copy_perm])
        .zip([alpha, alpha_squared])
    {
        arithmetic::sub_assign(&mut c.as_mut()[..r.size()], r.as_ref(), stream)?;
        c.scale_on(alpha, stream)?; // TODO
        multiply_monomial_with_multiple_sparse_polys_inplace(
            &mut c,
            sparse_poly,
            9 * domain_size,
            stream,
        )?;
        w_monomial_proto.add_assign_on(&c, stream)?;
    }

    // do divison
    let h_sparse_product_coeffs = compute_product_of_multiple_sparse_monomials(&sparse_polys);
    if SANITY_CHECK {
        for (idx, point) in opening_points.into_iter().enumerate() {
            assert_eq!(
                F::zero(),
                horner_evaluation(&h_sparse_product_coeffs, point),
                "invalid sparse poly product: {idx}"
            );
        }
    }
    // Divison will happen in lagrange basis and fft requires 16 * N points
    let w_monomial = divide_in_values(
        w_monomial_proto,
        h_sparse_product_coeffs,
        domain_size,
        stream,
    )?;

    Ok(w_monomial)
}

pub fn compute_w_monomial_in_values<F, CM>(
    combined_monomial_stoarge: &mut CM,
    domain_size: usize,
    common_combined_degree: usize,
    r_monomials: &[Poly<F, MonomialBasis>; 3],
    h_z: F,
    h_z_omega: F,
    h_alpha_pows: [F; 2],
    opening_points: [F; 4],
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
    CM: CombinedMonomialStorage<Poly = Poly<F, MonomialBasis>>,
{
    let [alpha, alpha_squared] = h_alpha_pows;

    let all_sparse_polys = construct_set_difference_monomials(h_z, h_z_omega);
    let mut result_values = Poly::<F, LagrangeBasis>::zero(16 * domain_size);
    for (poly_idx, ((r, sparse_polys), alpha)) in r_monomials
        .iter()
        .zip(all_sparse_polys.iter().cloned())
        .zip([F::one(), alpha, alpha_squared])
        .enumerate()
    {
        let mut this = Poly::<F, MonomialBasis>::zero(16 * domain_size);
        combined_monomial_stoarge.read_into(poly_idx, &mut this, stream)?;
        arithmetic::sub_assign(&mut this.as_mut()[..r.size()], r.as_ref(), stream)?;

        let mut sparse_product = compute_product_of_multiple_sparse_monomials(&sparse_polys);
        if SANITY_CHECK {
            for (idx, h) in opening_points.iter().cloned().take(3).enumerate() {
                if poly_idx == idx {
                    continue;
                }
                assert_eq!(horner_evaluation(&sparse_product, h), F::zero(),);
            }
        }
        if poly_idx > 0 {
            sparse_product
                .iter_mut()
                .for_each(|el| el.mul_assign(&alpha));
        }
        let mut other = Poly::<F, MonomialBasis>::zero(16 * domain_size);
        mem::h2d_on(
            &sparse_product,
            &mut other.as_mut()[..sparse_product.len()],
            stream,
        )?;

        let mut this_values = this.fft_on(stream)?;
        let other_values = other.fft_on(stream)?;
        this_values.mul_assign_on(&other_values, stream)?;
        result_values.add_assign_on(&this_values, stream)?;
    }

    let union_sparse_product =
        compute_product_of_multiple_sparse_monomials(&all_sparse_polys.last().unwrap());
    if SANITY_CHECK {
        for h in opening_points {
            assert_eq!(horner_evaluation(&union_sparse_product, h), F::zero(),);
        }
    }
    let mut denum = Poly::<F, MonomialBasis>::zero(16 * domain_size);
    mem::h2d_on(
        &union_sparse_product,
        &mut denum.as_mut()[..union_sparse_product.len()],
        stream,
    )?;
    let mut denum_values = denum.fft_on(stream)?;
    denum_values.batch_inverse(stream)?;
    result_values.mul_assign_on(&denum_values, stream)?;

    result_values.bitreverse(stream)?;
    let result = result_values.ifft_on(stream)?;

    if SANITY_CHECK {
        let leading_coeffs = result.as_ref()[common_combined_degree..].to_vec_on(stream)?;
        leading_coeffs
            .iter()
            .rev()
            .enumerate()
            .for_each(|(idx, coeff)| assert!(coeff.is_zero(), "{idx}-th coeff should be zero"));
    }
    Ok(result)
}

pub fn compute_w_prime_monomial<F, CM>(
    w: Poly<F, MonomialBasis>,
    combined_monomial_storage: &mut CM,
    domain_size: usize,
    common_combined_degree: usize,
    r_evals_at_y: [DScalar<F>; 3],
    h_z: F,
    h_z_omega: F,
    h_y: F,
    h_alpha_pows: [F; 2],
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
    CM: CombinedMonomialStorage<Poly = Poly<F, MonomialBasis>>,
{
    // each product already cancels evaluation of Z_T{S0} and also
    // contains factor of corresponding power of the alpha
    let [
        sparse_polys_for_trace_at_y, // Z_{T\S1}(x) deg = 14
        sparse_polys_for_copy_perm_at_y,// Z_{T\S2}(x) deg = 12
        product_of_sparse_polys_at_y_negated,// Z_T(x) = 18
    ] = evaluate_set_difference_monomials_at_y(
        h_z,
        h_z_omega,
        h_y,
        h_alpha_pows,
        stream,
    )?;

    let [r0_at_y, r1_at_y, r2_at_y] = r_evals_at_y;

    // L(x) = (C0(x) - r0(y)) + (alpha*Z_T\S1(y)/Z_T\S0(y))(C1(x) - r1(y)) + (alpha^2*Z_T\S2(y)/Z_T\S0(y))(C2(x) - r2(y)) - (Z_t(y)/Z_T\S0(y))*W(x)
    // W'(x) = L(x)/(x-y)
    // combined_monomial_storage.load(0, stream)?.clo;
    let mut w_prime_monomial_proto = w;
    w_prime_monomial_proto.scale_on(&product_of_sparse_polys_at_y_negated, stream)?;

    let mut c0 = Poly::<F, MonomialBasis>::zero(common_combined_degree);
    combined_monomial_storage.read_into(0, &mut c0, stream)?;

    let mut c1 = Poly::<F, MonomialBasis>::zero(common_combined_degree);
    combined_monomial_storage.read_into(1, &mut c1, stream)?;

    let mut c2 = Poly::<F, MonomialBasis>::zero(common_combined_degree);
    combined_monomial_storage.read_into(2, &mut c2, stream)?;

    arithmetic::sub_constant(&mut c0.as_mut()[0..1], &r0_at_y, stream)?;
    w_prime_monomial_proto.add_assign_on(&c0, stream)?;
    drop(c0);

    arithmetic::sub_constant(&mut c1.as_mut()[0..1], &r1_at_y, stream)?;
    w_prime_monomial_proto.add_assign_scaled_on(&c1, &sparse_polys_for_trace_at_y, stream)?;
    drop(c1);

    arithmetic::sub_constant(&mut c2.as_mut()[0..1], &r2_at_y, stream)?;
    w_prime_monomial_proto.add_assign_scaled_on(&c2, &sparse_polys_for_copy_perm_at_y, stream)?;
    drop(c2);

    let mut h_y_negated = h_y;
    h_y_negated.negate();
    let h_divisor = vec![h_y_negated, F::one()];

    divide_in_values(w_prime_monomial_proto, h_divisor, domain_size, stream)
}

pub fn compute_w_prime_monomial_in_values<F, CM>(
    w: Poly<F, LagrangeBasis>,
    combined_monomial_storage: &mut CM,
    domain_size: usize,
    r_evals_at_y: [DScalar<F>; 3],
    h_z: F,
    h_z_omega: F,
    h_y: F,
    h_alpha_pows: [F; 2],
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
    CM: CombinedMonomialStorage<Poly = Poly<F, MonomialBasis>>,
{
    // each product already cancels evaluation of Z_T{S0} and also
    // contains factor of corresponding power of the alpha
    let [
        sparse_polys_for_trace_at_y, // Z_{T\S1}(x) deg = 14
        sparse_polys_for_copy_perm_at_y,// Z_{T\S2}(x) deg = 12
        product_of_sparse_polys_at_y_negated,// Z_T(x) = 18
    ] = evaluate_set_difference_monomials_at_y(
        h_z,
        h_z_omega,
        h_y,
        h_alpha_pows,
        stream,
    )?;

    // L(x) = (C0(x) - r0(y)) + (alpha*Z_T\S1(y)/Z_T\S0(y))(C1(x) - r1(y)) + (alpha^2*Z_T\S2(y)/Z_T\S0(y))(C2(x) - r2(y)) - (Z_t(y)/Z_T\S0(y))*W(x)
    // W'(x) = L(x)/(x-y)
    // combined_monomial_storage.load(0, stream)?.clo;
    let mut w_prime_values = w;
    w_prime_values.scale_on(&product_of_sparse_polys_at_y_negated, stream)?;

    let one = DScalar::one(stream)?;
    for (poly_idx, (r_at_y, sparse_contrib_at_y)) in r_evals_at_y
        .into_iter()
        .zip(
            [
                one,
                sparse_polys_for_trace_at_y,
                sparse_polys_for_copy_perm_at_y,
            ]
            .into_iter(),
        )
        .enumerate()
    {
        let mut c_monomial = Poly::<F, MonomialBasis>::zero(16 * domain_size);
        combined_monomial_storage.read_into(poly_idx, &mut c_monomial, stream)?;
        let mut c_values = c_monomial.fft_on(stream)?;
        c_values.sub_constant_on(&r_at_y, stream)?;
        c_values.scale_on(&sparse_contrib_at_y, stream)?;

        w_prime_values.add_assign_on(&c_values, stream)?;
    }

    let mut h_y_negated = h_y;
    h_y_negated.negate();
    let h_divisor = vec![h_y_negated, F::one()];

    let mut padded_denum = Poly::zero(16 * domain_size);
    mem::h2d_on(
        &h_divisor,
        &mut padded_denum.as_mut()[..h_divisor.len()],
        stream,
    )?;
    let mut padded_denum_values = padded_denum.fft_on(stream)?;
    padded_denum_values.batch_inverse(stream)?;

    w_prime_values.mul_assign_on(&padded_denum_values, stream)?;

    let w_prime_monomial = w_prime_values.ifft_on(stream)?;

    Ok(w_prime_monomial)
}
