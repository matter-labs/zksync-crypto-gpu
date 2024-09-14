use super::*;
use std::{
    alloc::Allocator,
    slice::{Iter, IterMut},
};

use bellman::{
    bn256::Bn256,
    plonk::{
        better_better_cs::cs::{Circuit, SynthesisMode},
        commitments::transcript::Transcript,
    },
};
use fflonk::{
    commit_point_as_xy, compute_generators, horner_evaluation, FflonkAssembly, FflonkProof,
};

pub(crate) trait PolyStorage<F, const N: usize>: Sized {
    unsafe fn allocate_zeroed_on(domain_size: usize, stream: bc_stream) -> Self;
    fn num_polys(&self) -> usize {
        N
    }
    fn as_mut_ptr(&mut self) -> *mut F;
}

pub struct MonomialStorage<F: PrimeField, const N: usize>([Poly<F, MonomialBasis>; N]);

pub(crate) type MainGateSelectors<F> = MonomialStorage<F, 5>;
pub(crate) type Permutations<F> = MonomialStorage<F, 3>;
pub(crate) type Trace<F> = MonomialStorage<F, 3>;

impl<F, const N: usize> PolyStorage<F, N> for MonomialStorage<F, N>
where
    F: PrimeField,
{
    unsafe fn allocate_zeroed_on(domain_size: usize, stream: bc_stream) -> Self {
        // constructing permutation polys require storage to be adjacent
        let mut chunks =
            DVec::allocate_zeroed_on(domain_size * N, stream).into_owned_chunks(domain_size);
        chunks.reverse();
        Self(std::array::from_fn(|_| {
            Poly::<F, MonomialBasis>::from_buffer(chunks.pop().unwrap())
        }))
    }

    fn as_mut_ptr(&mut self) -> *mut F {
        self.0[0].as_mut().as_mut_ptr()
    }
}

impl<F, const N: usize> MonomialStorage<F, N>
where
    F: PrimeField,
{
    pub fn iter(&self) -> Iter<Poly<F, MonomialBasis>> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<Poly<F, MonomialBasis>> {
        self.0.iter_mut()
    }
}

impl<F, const N: usize> Drop for MonomialStorage<F, N>
where
    F: PrimeField,
{
    fn drop(&mut self) {
        // TODO
        let mut polys = vec![];
        for poly in self.0.iter_mut() {
            let poly = std::mem::replace(
                poly,
                Poly::<F, MonomialBasis>::from_buffer(DVec::dangling()),
            );
            polys.push(poly);
        }
        let owner = polys.remove(0);
        for poly in polys.into_iter() {
            std::mem::forget(poly);
        }
    }
}

pub fn create_proof<
    E: Engine,
    C: Circuit<E>,
    S: SynthesisMode,
    T: Transcript<E::Fr>,
    A: HostAllocator,
>(
    assembly: &FflonkAssembly<E, S>,
    setup: &FflonkDeviceSetup<E, C, A>,
    worker: &bellman::worker::Worker,
) -> CudaResult<FflonkProof<E, C>> {
    assert!(S::PRODUCE_WITNESS);
    assert!(assembly.is_finalized);

    let domain_size = assembly.n() + 1;
    assert!(domain_size.is_power_of_two());
    assert!(
        domain_size.trailing_zeros() <= 23,
        "Only trace length up to 2^23 is allowed"
    );

    let mut transcript = T::new();

    // commit initial values
    let h_public_inputs = assembly.input_assingments.clone();
    assert!(h_public_inputs.is_empty() == false);
    for inp in h_public_inputs.iter() {
        transcript.commit_field_element(inp);
    }
    commit_point_as_xy::<E, _>(&mut transcript, &setup.c0_commitment);
    dbg!(setup.c0_commitment);

    // take witnesses
    let stream = bc_stream::new().unwrap();
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
        Poly::<_, LDE>::zero(main_gate_quotient_degree * domain_size, stream);

    let public_inputs = DVec::from_host_slice_on(&h_public_inputs, stream)?;
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
    // let common_combined_poly_degree = 10 * domain_size;
    let num_first_round_polys = 4;
    let c1_degree = 8 * (domain_size - 1);
    let mut c1_monomial = Poly::<_, MonomialBasis>::zero(c1_degree, stream);
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
    commit_point_as_xy::<E, _>(&mut transcript, &c1_commitment);

    let h_beta = transcript.get_challenge();
    let h_gamma = transcript.get_challenge();
    dbg!(h_beta);
    dbg!(h_gamma);

    let beta = DScalar::from_host_value_on(&h_beta, stream)?;
    let gamma = DScalar::from_host_value_on(&h_gamma, stream)?;

    let mut h_non_residues = bellman::plonk::better_cs::generator::make_non_residues::<E::Fr>(2);
    let non_residues = DVec::from_host_slice_on(&h_non_residues, stream)?;
    // compute permutation polynomials
    // let permutation_monomials = unsafe {
    //     materialize_permutation_polys(&variable_indexes, &non_residues, domain_size, stream)?
    // };
    let permutation_monomials =
        unsafe { load_permutation_monomials(&setup.permutation_monomials, domain_size, stream)? };
    println!("scheduling materializing permutation monomials ");
    // compute product of scalars
    let mut h_non_residues_by_beta = vec![h_beta];
    for mut non_residue in h_non_residues.iter().cloned() {
        non_residue.mul_assign(&h_beta);
        h_non_residues_by_beta.push(non_residue);
    }
    let non_residues_by_beta = DScalars::from_host_scalars_on(&h_non_residues_by_beta, stream)?;

    println!("copy perm grand product");
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
        Poly::<_, LDE>::zero(padded_copy_perm_quotient_degree * domain_size, stream);

    for coset_idx in 0..padded_copy_perm_quotient_degree {
        println!("evaluate copy perm grand product for coset {coset_idx}");
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
    println!("copy-perm first quotient");
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
    println!("copy-perm second quotient");
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
    println!("combined c2 monomial");
    // combine monomials into c2 combined monomial
    let c2_degree = 9 * (domain_size - 1);
    let mut c2_monomial = Poly::<_, MonomialBasis>::zero(c2_degree, stream);
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
    commit_point_as_xy::<E, _>(&mut transcript, &c2_commitment);

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
    println!("evaluating polynomials");
    // the verifier will re-compute evaluations of the quotients
    // by herself from existing evaluations, however we still need to
    // evaluate quotients of the copy-perm at z*w

    let num_evals_at_z = 8 + 3 + 1;
    let num_evals_at_z_omega = 1 + 2;
    let num_all_evaluations = num_evals_at_z + num_evals_at_z_omega;
    let mut all_evaluations = DScalars::allocate_zeroed_on(num_all_evaluations, stream)?;
    let (evaluations_at_z, evaluations_at_z_omega) = all_evaluations.split_at_mut(num_evals_at_z);
    println!("do evaluation");
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

    println!("aux evaluations");
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

    // get linearization challenge
    let h_alpha = transcript.get_challenge();
    dbg!(h_alpha);
    let mut h_alpha_pows = [h_alpha, h_alpha];
    h_alpha_pows[1].mul_assign(&h_alpha);
    let alpha_pows = DScalars::from_host_scalars_on(&h_alpha_pows, stream)?;

    let opening_points = compute_opening_points(h_r, h_z, h_z_omega, power, domain_size);

    dbg!(&opening_points);

    // construct r_i(x)  monomials from evaluations
    let [h_r0_monomial, h_r1_monomial, h_r2_monomial] = construct_r_monomials(
        h_all_evaluations.clone(),
        h_aux_evaluations.clone(),
        opening_points,
    );

    let mut r0_monomial = Poly::zero(h_r0_monomial.len(), stream);
    mem::h2d_on(&h_r0_monomial, r0_monomial.as_mut(), stream)?;

    let mut r1_monomial = Poly::zero(h_r1_monomial.len(), stream);
    mem::h2d_on(&h_r1_monomial, r1_monomial.as_mut(), stream)?;

    let mut r2_monomial = Poly::zero(h_r2_monomial.len(), stream);
    mem::h2d_on(&h_r2_monomial, r2_monomial.as_mut(), stream)?;

    // re-construct combined polynomial of the preprocessing parts
    let c0_degree = 8 * domain_size;
    let mut c0_monomial = Poly::<_, MonomialBasis>::zero(c0_degree, stream);
    combine_monomials(
        main_gate_selectors_monomial
            .iter()
            .chain(permutation_monomials.iter()),
        &mut c0_monomial,
        8,
        stream,
    )?;
    if SANITY_CHECK {
        let [setup_omega, trace_omega, copy_perm_omega] = compute_generators::<E::Fr>(8, 4, 3);
        let mut expected = DScalar::zero(stream)?;
        let mut actual = DScalar::zero(stream)?;

        let mut opening_points_iter = opening_points.into_iter();
        let mut h_current = opening_points_iter.next().unwrap();
        for _ in 0..8 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            c0_monomial.evaluate_at_into_on(&current, &mut expected, stream)?;
            r0_monomial.evaluate_at_into_on(&current, &mut actual, stream)?;
            assert_eq!(
                expected.to_host_value_on(stream)?,
                actual.to_host_value_on(stream)?
            );
            h_current.mul_assign(&setup_omega);
        }
        let mut h_current = opening_points_iter.next().unwrap();
        for _ in 0..4 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            c1_monomial.evaluate_at_into_on(&current, &mut expected, stream)?;
            r1_monomial.evaluate_at_into_on(&current, &mut actual, stream)?;
            assert_eq!(
                expected.to_host_value_on(stream)?,
                actual.to_host_value_on(stream)?
            );
            h_current.mul_assign(&trace_omega);
        }

        let mut h_current = opening_points_iter.next().unwrap();
        for _ in 0..3 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            c2_monomial.evaluate_at_into_on(&current, &mut expected, stream)?;
            r2_monomial.evaluate_at_into_on(&current, &mut actual, stream)?;
            assert_eq!(
                expected.to_host_value_on(stream)?,
                actual.to_host_value_on(stream)?
            );
            h_current.mul_assign(&copy_perm_omega);
        }

        let mut h_current = opening_points_iter.next().unwrap();
        for _ in 0..3 {
            let current = DScalar::from_host_value_on(&h_current, stream)?;
            c2_monomial.evaluate_at_into_on(&current, &mut expected, stream)?;
            r2_monomial.evaluate_at_into_on(&current, &mut actual, stream)?;
            assert_eq!(
                expected.to_host_value_on(stream)?,
                actual.to_host_value_on(stream)?
            );
            h_current.mul_assign(&copy_perm_omega);
        }
    }
    let combined_monomials = [c0_monomial, c1_monomial, c2_monomial];
    let r_monomials = [r0_monomial, r1_monomial, r2_monomial];

    // compute opening proof W(x) in monomial  W(x) = f(x) / Z(x)
    let w_monomial = compute_w_monomial(
        domain_size,
        &combined_monomials,
        &r_monomials,
        h_z,
        h_z_omega,
        opening_points,
        &alpha_pows,
        stream,
    )?;
    // commit W(x) commitment into transcript
    let w_commitment = commit_monomial::<E>(&w_monomial, domain_size, stream)?;
    commit_point_as_xy::<E, _>(&mut transcript, &w_commitment);
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
        domain_size,
        w_monomial,
        combined_monomials,
        r_evals_at_y,
        h_z,
        h_z_omega,
        h_y,
        h_alpha_pows,
        stream,
    )?;

    let w_prime_commitment = commit_monomial::<E>(&w_prime_monomial, domain_size, stream)?;
    dbg!(w_prime_commitment);

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
    proof.lagrange_basis_inverses = compute_flattened_lagrange_basis_inverses(opening_points, h_y);

    stream.sync().unwrap();
    Ok(proof)
}

// TODO: bit representation of the selectors
pub unsafe fn load_main_gate_selectors<F: PrimeField, A: HostAllocator>(
    h_main_gate_selector_monomials: &[Vec<F, A>; 5],
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<MainGateSelectors<F>> {
    let mut selectors_monomial = MainGateSelectors::allocate_zeroed_on(domain_size, stream);

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
    let mut permutation_monomials = Permutations::allocate_zeroed_on(domain_size, stream);

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
    let mut trace_monomial = Trace::allocate_zeroed_on(domain_size, stream);

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
    let mut trace_monomial = Trace::allocate_zeroed_on(domain_size, stream);
    // same indexes will be used in materializing permutation polys and it requires
    // it to be same length with trace
    let mut indexes = DVec::allocate_zeroed_on(3 * domain_size, stream);
    // transfer flattened witness values
    let num_aux_variables = h_aux_assignments.len();
    // assignments should have dummy value as first element
    let total_num_variables = num_aux_variables + h_input_assignments.len() + 1;
    let mut full_assignments = DVec::allocate_zeroed_on(total_num_variables, stream);
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
    assignments: &DVec<F>,
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
    stream: bc_stream,
) -> CudaResult<Permutations<F>> {
    use gpu_ffi::generate_permutation_polynomials_configuration;
    let mut permutations_monomial = Permutations::<F>::allocate_zeroed_on(domain_size, stream);
    let num_cols = permutations_monomial.num_polys();
    assert_eq!(num_cols * domain_size, indexes.len());
    assert_eq!(non_residues.len(), num_cols);

    let log_rows_count = domain_size.trailing_zeros();

    let indexes_ptr = indexes.as_ptr() as *mut usize;
    let non_residues_ptr = non_residues.as_ptr() as *mut F;

    let permutations_monomial_ptr = permutations_monomial.as_mut_ptr();
    let cfg = generate_permutation_polynomials_configuration {
        mem_pool: _mem_pool(),
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

pub fn compute_w_monomial<F>(
    domain_size: usize,
    combined_monomials: &[Poly<F, MonomialBasis>; 3],
    r_monomials: &[Poly<F, MonomialBasis>; 3],
    h_z: F,
    h_z_omega: F,
    opening_points: [F; 4],
    alpha_pows: &DScalars<F>,
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
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

    let [c0, c1, c2] = combined_monomials;
    let [r0, r1, r2] = r_monomials;

    let mut alpha_pows_iter = alpha_pows.iter();
    let alpha = &alpha_pows_iter.next().unwrap();
    let alpha_squared = &alpha_pows_iter.next().unwrap();

    // f(x) = Z_T\S0(x)(C0(x) - r0(x)) + alpha*Z_T\S1(x)(C1(x) - r1(x)) + alpha^2*Z_T\S2(x)(C2(x) - r2(x))
    // W(x) = f(x)/Z_T(x)

    // aggregate parts
    // c2 degree is the largest one, we can use it as accumulator
    assert!(c2.size() <= 9 * (domain_size - 1) + 12);
    let mut tmp = c2.clone_on(stream)?;
    tmp.sub_assign_on(&r2, stream)?;
    tmp.scale_on(&alpha_squared, stream)?;
    let mut w_monomial_proto =
        multiply_monomial_with_multiple_sparse_polys(tmp, sparse_polys_for_copy_perm, stream)?;

    for (c, r, sparse_poly, alpha) in [
        (c0, r0, sparse_polys_for_setup, None),
        (c1, r1, sparse_polys_for_trace, Some(alpha)),
    ]
    .into_iter()
    {
        let mut tmp = c.clone_on(stream)?;
        tmp.sub_assign_on(&r, stream)?;
        if let Some(alpha) = alpha {
            tmp.scale_on(alpha, stream)?;
        }
        let tmp = multiply_monomial_with_multiple_sparse_polys(tmp, sparse_poly, stream)?;
        w_monomial_proto.add_assign_on(&tmp, stream)?;
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
    // recall that max degree combined poly is 9(n-1) + 12 and
    // however deg(W(x))=9(n-1)-20 so we only need 9n points
    let w_monomial = divide_in_values(
        w_monomial_proto.clone_on(stream)?, // TOOD
        h_sparse_product_coeffs,
        domain_size,
        stream,
    )?;

    if SANITY_CHECK {
        let quotient = w_monomial.clone_on(stream)?;
        let product = multiply_monomial_with_multiple_sparse_polys(quotient, sparse_polys, stream)?;
        let point = DScalar::from_host_value_on(&h_z, stream)?;
        let mut expected = DScalar::zero(stream)?;
        let mut actual = DScalar::zero(stream)?;
        w_monomial_proto.evaluate_at_into_on(&point, &mut expected, stream)?;
        product.evaluate_at_into_on(&point, &mut actual, stream)?;
        let h_expected = expected.to_host_value_on(stream)?;
        let h_actual = actual.to_host_value_on(stream)?;
        stream.sync().unwrap();
        assert_eq!(h_expected, h_actual,);
    }

    Ok(w_monomial)
}

pub fn compute_w_prime_monomial<F>(
    domain_size: usize,
    w: Poly<F, MonomialBasis>,
    combined_monomials: [Poly<F, MonomialBasis>; 3],
    r_evals_at_y: [DScalar<F>; 3],
    h_z: F,
    h_z_omega: F,
    h_y: F,
    h_alpha_pows: [F; 2],
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
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

    let [mut c0, mut c1, mut c2] = combined_monomials;
    let [r0_at_y, r1_at_y, r2_at_y] = r_evals_at_y;

    // L(x) = (C0(x) - r0(y)) + (alpha*Z_T\S1(y)/Z_T\S0(y))(C1(x) - r1(y)) + (alpha^2*Z_T\S2(y)/Z_T\S0(y))(C2(x) - r2(y)) - (Z_t(y)/Z_T\S0(y))*W(x)
    // W'(x) = L(x)/(x-y)

    let mut w_prime_monomial_proto = w;
    w_prime_monomial_proto.scale_on(&product_of_sparse_polys_at_y_negated, stream)?;

    arithmetic::sub_constant(&mut c0.as_mut()[0..1], &r0_at_y, stream)?;
    w_prime_monomial_proto.add_assign_on(&c0, stream)?;

    arithmetic::sub_constant(&mut c1.as_mut()[0..1], &r1_at_y, stream)?;
    w_prime_monomial_proto.add_assign_scaled_on(&c1, &sparse_polys_for_trace_at_y, stream)?;

    arithmetic::sub_constant(&mut c2.as_mut()[0..1], &r2_at_y, stream)?;
    w_prime_monomial_proto.add_assign_scaled_on(&c2, &sparse_polys_for_copy_perm_at_y, stream)?;

    let mut h_y_negated = h_y;
    h_y_negated.negate();
    let h_divisor = vec![h_y_negated, F::one()];

    divide_in_values(w_prime_monomial_proto, h_divisor, domain_size, stream)
}
