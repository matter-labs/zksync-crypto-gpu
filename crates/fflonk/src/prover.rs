use super::*;
use std::alloc::{Allocator, Global};

use bellman::{
    bn256::Bn256,
    plonk::{
        better_better_cs::{
            cs::{Assembly, Circuit, PlonkCsWidth3Params, SynthesisMode},
            gates::naive_main_gate::NaiveMainGate,
        },
        commitments::transcript::Transcript,
    },
};
use fflonk::{commit_point_as_xy, FflonkProof, FflonkSetup, FflonkSnarkVerifierCircuit};

pub(crate) trait PolyStorage<F, const N: usize>: Sized {
    unsafe fn allocate_zeroed_on(domain_size: usize, stream: bc_stream) -> Self;
    fn num_polys(&self) -> usize {
        N
    }
    fn as_mut_ptr(&mut self) -> *mut F;
}
pub type MonomialStorage<F, const N: usize> = [Poly<F, MonomialBasis>; N];
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
            DVec::with_capacity_zeroed_on(domain_size * N, stream).into_owned_chunks(domain_size);
        std::array::from_fn(|_| Poly::<F, MonomialBasis>::from_buffer(chunks.pop().unwrap()))
    }

    fn as_mut_ptr(&mut self) -> *mut F {
        self[0].as_mut().as_mut_ptr()
    }
}

pub struct FflonkDeviceSetup<E: Engine, C: Circuit<E>, A: HostAllocator = GlobalHost> {
    main_gate_selectors: [Vec<E::Fr, A>; 5],
    column_indexes: [Vec<usize, A>; 3],
    c0_commitment: E::G1Affine,
    _c: std::marker::PhantomData<C>,
}

impl<E: Engine, C: Circuit<E>, A: HostAllocator> FflonkDeviceSetup<E, C, A> {
    pub fn from_host_setup(_host_setup: &FflonkSetup<E, C>) -> CudaResult<Self> {
        todo!()
    }

    pub fn read<R: std::io::Read>(mut _reader: R) -> std::io::Result<Self> {
        todo!()
    }

    pub fn write<W: std::io::Write>(&self, mut _writer: W) -> std::io::Result<()> {
        todo!()
    }
}

pub type FflonkSnarkVerifierCircuitDeviceSetup =
    FflonkDeviceSetup<Bn256, FflonkSnarkVerifierCircuit, Global>;

pub fn create_proof<
    E: Engine,
    C: Circuit<E>,
    S: SynthesisMode,
    T: Transcript<E::Fr>,
    A: HostAllocator,
>(
    assembly: Assembly<E, PlonkCsWidth3Params, NaiveMainGate, S, A>,
    setup: &FflonkDeviceSetup<E, C, A>,
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
    let h_input_values = assembly.input_assingments.clone();
    assert!(h_input_values.is_empty() == false);
    for inp in h_input_values.iter() {
        transcript.commit_field_element(inp);
    }
    commit_point_as_xy::<E, _>(&mut transcript, &setup.c0_commitment);

    let stream = bc_stream::new().unwrap();
    let input_values = DVec::from_host_slice_on(&h_input_values, stream)?;
    // take witnesses
    let (trace_monomials, variable_indexes) = unsafe {
        load_trace(
            &assembly.input_assingments,
            &assembly.aux_assingments,
            &setup.column_indexes,
            domain_size,
            stream,
        )?
    };
    // load main gate selectors
    let main_gate_selectors_monomial =
        unsafe { load_main_gate_selectors(&setup.main_gate_selectors, domain_size, stream)? };

    // compute main gate quotient chunk by chunk
    let mut main_gate_quotient_lde = Poly::<_, LDE>::with_capacity_on(2 * domain_size, stream);

    for coset_idx in 0..2 {
        let quotient_sum = evaluate_main_gate_constraints(
            coset_idx,
            domain_size,
            trace_monomials.iter(),
            main_gate_selectors_monomial.iter(),
            &input_values,
            stream,
        )?;
        let start = coset_idx * domain_size;
        let end = start + domain_size;
        mem::d2d_on_stream(
            quotient_sum.as_ref(),
            &mut main_gate_quotient_lde.as_mut()[start..end],
            stream,
        )?;
    }
    main_gate_quotient_lde.bitreverse(stream)?;

    let main_gate_quotient_monomial = main_gate_quotient_lde.icoset_fft_on(stream)?;

    // combine monomials into c1 combined monomial
    let common_combined_poly_degree = 9 * domain_size;
    let num_first_round_polys = 4;
    let mut c1_monomial =
        Poly::<_, MonomialBasis>::new_monomials_on(common_combined_poly_degree, stream);

    combine_monomials(
        trace_monomials
            .iter()
            .chain(std::iter::once(&main_gate_quotient_monomial)),
        &mut c1_monomial,
        num_first_round_polys,
        stream,
    )?;

    // commit to the c1(x)
    let c1_commitment = commit_monomial::<E>(&c1_monomial, stream)?;

    // commit commitment into transcript
    commit_point_as_xy::<E, _>(&mut transcript, &c1_commitment);

    let h_beta = transcript.get_challenge();
    let h_gamma = transcript.get_challenge();

    let beta = DScalar::from_host_value_on(&h_beta, stream)?;
    let gamma = DScalar::from_host_value_on(&h_gamma, stream)?;

    let h_non_residues = bellman::plonk::better_cs::generator::make_non_residues::<E::Fr>(2);
    let non_residues = DVec::from_host_scalars_on(&h_non_residues, stream)?;
    // compute permutation polynomials
    let permutation_monomials = unsafe {
        construct_permutation_polys(&variable_indexes, &non_residues, domain_size, stream)?
    };

    let mut non_residues_by_beta = vec![DScalar::from_host_value_on(&h_beta, stream).unwrap()];
    for mut non_residue in h_non_residues.clone().into_iter() {
        non_residue.mul_assign(&h_beta);
        let d_non_residue = DScalar::from_host_value_on(&non_residue, stream)?;
        non_residues_by_beta.push(d_non_residue);
    }

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
    // second quotient is the first element of the grand product
    let mut copy_perm_second_quotient_coset =
        Poly::<_, CosetEvals>::with_capacity_on(domain_size, stream);
    // compute product of scalars first
    for coset_idx in 0..copy_perm_quotient_degree {
        let (first_quotient_sum, second_quotient) = evaluate_copy_permutation_constraints(
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
        mem::d2d_on_stream(
            first_quotient_sum.as_ref(),
            &mut main_gate_quotient_lde.as_mut()[start..end],
            stream,
        )?;

        if let Some(second_quotient) = second_quotient {
            assert_eq!(coset_idx, 0);
            mem::d2d_on_stream(
                second_quotient.as_ref(),
                &mut copy_perm_second_quotient_coset.as_mut()[start..end],
                stream,
            )?;
        }
    }
    copy_perm_first_quotient.bitreverse(stream)?;
    let mut copy_perm_first_quotient_monomial = copy_perm_first_quotient.icoset_fft_on(stream)?;
    copy_perm_first_quotient_monomial.trim_to_degree(copy_perm_quotient_degree * domain_size);

    copy_perm_second_quotient_coset.bitreverse(stream)?;
    let copy_perm_second_quotient_monomial =
        copy_perm_second_quotient_coset.icoset_fft_on(stream)?;

    // combine monomials into c2 combined monomial

    let mut c2_monomial =
        Poly::<_, MonomialBasis>::new_monomials_on(common_combined_poly_degree, stream);

    combine_monomials(
        [
            &copy_perm_first_quotient_monomial,
            &copy_perm_first_quotient_monomial,
            &copy_perm_second_quotient_monomial,
        ]
        .into_iter(),
        &mut c2_monomial,
        permutation_monomials.num_polys(),
        stream,
    )?;

    // commit to the c2(x)
    let c2_commitment = commit_monomial::<E>(&c2_monomial, stream)?;
    // commit commitment into transcript
    commit_point_as_xy::<E, _>(&mut transcript, &c2_commitment);

    // get evaluation challenge
    let power = 24;
    let h_r = transcript.get_challenge();

    let h_z = h_r.pow(&[power as u64]);
    let z = DScalar::from_host_value_on(&h_z, stream)?;
    let mut h_z_omega = h_z.clone();
    let omega = bellman::plonk::domains::Domain::new_for_size(domain_size as u64)
        .unwrap()
        .generator;
    h_z_omega.mul_assign(&omega);
    let z_omega = DScalar::from_host_value_on(&h_z_omega, stream)?;

    // compute all evaluations
    let num_all_evaluations = 8 + 3 + 2;
    let mut all_evaluations: DVec<DScalar<E::Fr>> =
        DVec::with_capacity_on(num_all_evaluations, stream);

    for (monomial, value) in main_gate_selectors_monomial
        .iter()
        .chain(permutation_monomials.iter())
        .chain(trace_monomials.iter())
        .chain([&copy_perm_grand_prod_monomial])
        .zip(all_evaluations.iter_mut())
    {
        monomial.evaluate_at_into_on(&z, value, stream)?;
    }

    let num_aux_evaluations = 1 + 2 + 2;
    let mut aux_evaluations: DVec<DScalar<E::Fr>> =
        DVec::with_capacity_on(num_aux_evaluations, stream);

    let (aux_evaluations_at_z, aux_evaluations_at_z_omega) = aux_evaluations.split_at_mut(3);

    for (monomial, value) in [
        &main_gate_quotient_monomial,
        &copy_perm_first_quotient_monomial,
        &copy_perm_second_quotient_monomial,
    ]
    .into_iter()
    .zip(aux_evaluations_at_z.iter_mut())
    {
        monomial.evaluate_at_into_on(&z, value, stream)?;
    }

    for (monomial, value) in [
        &copy_perm_first_quotient_monomial,
        &copy_perm_second_quotient_monomial,
    ]
    .into_iter()
    .zip(aux_evaluations_at_z_omega.iter_mut())
    {
        monomial.evaluate_at_into_on(&z_omega, value, stream)?;
    }

    let copy_perm_grand_product_at_z_omega = &mut all_evaluations[num_all_evaluations - 2];
    copy_perm_grand_prod_monomial.evaluate_at_into_on(
        &z_omega,
        copy_perm_grand_product_at_z_omega,
        stream,
    )?;

    // commit evaluations into transcript
    let h_all_evaluations = all_evaluations.to_scalars_vec(stream)?;

    h_all_evaluations
        .iter()
        .for_each(|el| transcript.commit_field_element(el));

    let h_aux_evaluations = aux_evaluations.to_scalars_vec(stream)?;

    // get linearization challenge
    let h_alpha = transcript.get_challenge();

    let mut h_alpha_pows = [h_alpha, h_alpha];
    h_alpha_pows[1].mul_assign(&h_alpha);
    let alpha_pows = DVec::from_host_scalars_on(&h_alpha_pows, stream)?;

    let (h0, h1, h2) = fflonk::utils::compute_opening_points::<E::Fr>(
        h_r,
        h_z,
        h_z_omega,
        power,
        8,
        4,
        3,
        domain_size,
        false,
        false,
    );

    // construct r_i(x)  monomials from evaluations
    let [h_r0_monomial, h_r1_monomial, h_r2_monomial] = construct_r_monomials(
        h_all_evaluations.clone(),
        h_aux_evaluations.clone(),
        h0,
        h1,
        h2,
    );

    let mut r0_monomial = Poly::zero(h_r0_monomial.len(), stream);
    mem::h2d_on_stream(&h_r0_monomial, r0_monomial.as_mut(), stream)?;

    let mut r1_monomial = Poly::zero(h_r1_monomial.len(), stream);
    mem::h2d_on_stream(&h_r1_monomial, r1_monomial.as_mut(), stream)?;

    let mut r2_monomial = Poly::zero(h_r2_monomial.len(), stream);
    mem::h2d_on_stream(&h_r2_monomial, r2_monomial.as_mut(), stream)?;

    // re-construct combined polynomial of the preprocessing parts
    let mut c0_monomial =
        Poly::<_, MonomialBasis>::new_monomials_on(common_combined_poly_degree, stream);
    let flattened_setup_polys = main_gate_selectors_monomial
        .iter()
        .chain(permutation_monomials.iter());
    combine_monomials(
        flattened_setup_polys,
        &mut c0_monomial,
        permutation_monomials.num_polys(),
        stream,
    )?;

    // compute opening proof W(x) in monomial  W(x) = f(x) / Z(x)
    let w_monomial = compute_w_monomial(
        domain_size,
        &c0_monomial,
        &c1_monomial,
        &c2_monomial,
        &r0_monomial,
        &r1_monomial,
        &r2_monomial,
        h_z,
        h_z_omega,
        &alpha_pows,
        stream,
    )?;
    // commit W(x) commitment into transcript
    let w_commitment = commit_monomial::<E>(&w_monomial, stream)?;
    commit_point_as_xy::<E, _>(&mut transcript, &w_commitment);

    // get challenge
    let h_y = transcript.get_challenge();

    // evaluate r monomials at challenge point
    let mut r_evals_at_y = [
        DScalar::one(stream)?,
        DScalar::one(stream)?,
        DScalar::one(stream)?,
    ];

    for (monomial, value) in [h_r0_monomial, h_r1_monomial, h_r2_monomial]
        .into_iter()
        .zip(r_evals_at_y.iter_mut())
    {
        let sum = fflonk::utils::horner_evaluation(&monomial, h_y);
        *value = DScalar::from_host_value_on(&sum, stream)?;
    }

    // compute linearization W'(x) = L(x) / (x-y)
    let w_prime_monomial = compute_w_prime_monomial(
        domain_size,
        w_monomial,
        c0_monomial,
        c1_monomial,
        c2_monomial,
        r_evals_at_y,
        h_z,
        h_z_omega,
        h_y,
        h_alpha_pows,
        stream,
    )?;

    let w_prime_commitment = commit_monomial::<E>(&w_prime_monomial, stream)?;

    // make proof
    let mut proof = FflonkProof::empty();
    proof.commitments = vec![
        c1_commitment,
        c2_commitment,
        w_commitment,
        w_prime_commitment,
    ];
    proof.evaluations = h_all_evaluations;
    proof.lagrange_basis_inverses =
        compute_flattened_lagrange_basis_inverses(h0.0, h1.0, (h2.0, h2.1), h_y);

    Ok(proof)
}

pub unsafe fn load_main_gate_selectors<F: PrimeField, A: HostAllocator>(
    h_main_gate_selectors: &[Vec<F, A>; 5],
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<MainGateSelectors<F>> {
    let mut selectors_monomial = MainGateSelectors::allocate_zeroed_on(domain_size, stream);

    for (src_col, dst_col) in h_main_gate_selectors
        .iter()
        .zip(selectors_monomial.iter_mut())
    {
        // load host values first
        let h2d_event = bc_event::new().unwrap();
        assert_eq!(src_col.len(), dst_col.size());
        mem::h2d(src_col, dst_col.as_mut())?;
        h2d_event.sync().unwrap();
        stream.wait(h2d_event).unwrap();
        // convert column into monomial basis
        ntt::inplace_ifft_on(dst_col.as_mut(), stream)?;
    }

    Ok(selectors_monomial)
}

pub unsafe fn load_trace<F: PrimeField, A: Allocator>(
    _input_assignments: &Vec<F>, // TODO
    aux_assignments: &Vec<F, A>,
    indexes: &[Vec<usize, A>; 3],
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<(Trace<F>, DVec<usize>)> {
    let mut trace_monomial = Trace::allocate_zeroed_on(domain_size, stream);

    // transfer flattened witness values
    let mut d_assignments = DVec::with_capacity_on(aux_assignments.len(), stream);
    mem::h2d_on_stream(&aux_assignments, &mut d_assignments, stream)?;
    let mut d_indexes =
        DVec::with_capacity_zeroed_on(trace_monomial.num_polys() * domain_size, stream);
    // assign values column by column
    for ((src_col, dst_col), trace_col) in indexes
        .iter()
        .zip(d_indexes.chunks_mut(domain_size))
        .zip(trace_monomial.iter_mut())
    {
        // load indexes first
        let h2d_event = bc_event::new().unwrap();
        assert!(src_col.len() < dst_col.len());
        mem::h2d(src_col, &mut dst_col[..src_col.len()])?;
        h2d_event.sync().unwrap();
        // TODO consider to pad with generated
        stream.wait(h2d_event).unwrap();
        variable_assignment_for_single_col(
            &d_assignments,
            dst_col,
            trace_col.as_mut(),
            domain_size,
            stream,
        )?;

        // convert column into monomial basis
        ntt::inplace_ifft_on(trace_col.as_mut(), stream)?;
        todo!();
    }

    Ok((trace_monomial, d_indexes))
}

pub fn variable_assignment_for_single_col<F: PrimeField>(
    assignments: &DVec<F>,
    indexes: &DSlice<usize>,
    trace_col: &mut DSlice<F>,
    domain_size: usize,
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
            domain_size as u32,
            stream,
        );
        if result != 0 {
            return Err(CudaError::VariableAssignmentError(result.to_string()));
        }

        Ok(())
    }
}

pub unsafe fn construct_permutation_polys<F: PrimeField>(
    indexes: &DVec<usize>,
    non_residues: &DVec<DScalar<F>>,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<Permutations<F>> {
    use gpu_ffi::generate_permutation_polynomials_configuration;
    let mut permutations_monomial = Permutations::<F>::allocate_zeroed_on(domain_size, stream);
    let num_cols = permutations_monomial.num_polys();

    assert_eq!(num_cols * domain_size, indexes.len());
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
    c0: &Poly<F, MonomialBasis>,
    c1: &Poly<F, MonomialBasis>,
    c2: &Poly<F, MonomialBasis>,
    r0: &Poly<F, MonomialBasis>,
    r1: &Poly<F, MonomialBasis>,
    r2: &Poly<F, MonomialBasis>,
    h_z: F,
    h_z_omega: F,
    alpha_pows: &DVec<DScalar<F>>,
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
    ] = fflonk::utils::construct_set_difference_monomials(
        h_z,
        h_z_omega,
        8, 4, 3,
        false,
    );

    // f(x) = Z_T\S0(x)(C0(x) - r0(x)) + alpha*Z_T\S1(x)(C1(x) - r1(x)) + alpha^2*Z_T\S2(x)(C2(x) - r2(x))
    // W(x) = f(x)/Z_T(x)
    let degree = 9 * domain_size;
    let padded_degree = 16 * domain_size;

    assert_eq!(c0.size(), degree);
    assert_eq!(c1.size(), degree);
    assert_eq!(c2.size(), degree);
    // aggregate parts
    let mut w_monomial = c0.clone_on(stream)?;
    sub_assign_mixed_degree_polys(&mut w_monomial, &r0, stream)?;
    multiply_monomial_with_multiple_sparse_polys(&mut w_monomial, &sparse_polys_for_setup, stream)?;

    for ((c, r, sparse_poly), alpha) in [
        &(c1, r1, sparse_polys_for_trace),
        &(c2, r2, sparse_polys_for_copy_perm),
    ]
    .into_iter()
    .zip(alpha_pows.iter())
    {
        let mut tmp = c.clone_on(stream)?;
        sub_assign_mixed_degree_polys(&mut tmp, &r, stream)?;
        multiply_monomial_with_multiple_sparse_polys(&mut tmp, &sparse_poly, stream)?;
        tmp.scale_on(&alpha, stream)?;
        w_monomial.add_assign_on(&tmp, stream)?;
    }
    // do divison
    let h_sparse_product = multiply_sparse_polys(&sparse_polys);
    let mut sparse_product_monomial = Poly::<_, MonomialBasis>::zero(padded_degree, stream);
    mem::h2d_on_stream(
        &h_sparse_product,
        &mut sparse_product_monomial.as_mut()[..h_sparse_product.len()],
        stream,
    )?;
    divide_in_values(
        &mut w_monomial,
        &sparse_product_monomial,
        domain_size,
        padded_degree,
        stream,
    )?;

    Ok(w_monomial)
}

pub fn compute_w_prime_monomial<F>(
    domain_size: usize,
    mut w: Poly<F, MonomialBasis>,
    mut c0: Poly<F, MonomialBasis>,
    mut c1: Poly<F, MonomialBasis>,
    mut c2: Poly<F, MonomialBasis>,
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
    let [
        sparse_polys_for_trace_at_y, // Z_{T\S1}(x) deg = 14
        sparse_polys_for_copy_perm_at_y,// Z_{T\S2}(x) deg = 12
        sparse_polys_at_y,// Z_T(x) = 18
    ] = evaluate_set_difference_monomials_at_y(
        h_z,
        h_z_omega,
        h_y,
        h_alpha_pows,
        stream
    )?;

    // L(x) = Z_T\S0(y)(C0(x) - r0(y)) + alpha*Z_T\S1(y)(C1(x) - r1(y)) + alpha^2*Z_T\S2(y)(C2(x) - r2(y)) - Z_t(y)*W(x)
    // W'(x) = L(x)/(x-y)
    let degree = 9 * domain_size;
    let padded_degree = degree.next_power_of_two();
    assert_eq!(c0.size(), degree);
    assert_eq!(c1.size(), degree);
    assert_eq!(c2.size(), degree);

    let [r0_at_y, r1_at_y, r2_at_y] = r_evals_at_y;

    c0.sub_constant_on(&r0_at_y, stream)?;
    c1.sub_constant_on(&r1_at_y, stream)?;
    c1.scale_on(&sparse_polys_for_trace_at_y, stream)?;
    c2.sub_constant_on(&r2_at_y, stream)?;
    c2.scale_on(&sparse_polys_for_copy_perm_at_y, stream)?;
    let mut w_prime_monomial = c0;
    w_prime_monomial.add_assign_on(&c1, stream)?;
    w_prime_monomial.add_assign_on(&c2, stream)?;
    w.scale_on(&sparse_polys_at_y, stream)?;
    w_prime_monomial.sub_assign_on(&w, stream)?;

    let mut h_y_negated = h_y;
    h_y_negated.negate();

    let h_divisor = [h_y_negated, F::one()];
    let mut divisor = Poly::zero(2, stream);
    mem::h2d_on_stream(&h_divisor, divisor.as_mut(), stream)?;
    divide_in_values(
        &mut w_prime_monomial,
        &divisor,
        padded_degree,
        domain_size,
        stream,
    )?;

    Ok(w_prime_monomial)
}
