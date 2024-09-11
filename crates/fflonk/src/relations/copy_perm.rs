use super::*;

pub fn compute_copy_perm_grand_product<'a, 'b, F: PrimeField, ITrace, IPermutations>(
    domain_size: usize,
    mut trace: ITrace,
    mut permutations: IPermutations,
    non_residues_by_beta: &Vec<DScalar<F>>,
    beta: &DScalar<F>,
    gamma: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
    ITrace: Iterator<Item = &'a Poly<F, MonomialBasis>>,
    IPermutations: Iterator<Item = &'b Poly<F, MonomialBasis>>,
{
    let a_mon = trace.next().unwrap();
    let b_mon = trace.next().unwrap();
    let c_mon = trace.next().unwrap();
    assert!(trace.next().is_none());

    let sigma_a = permutations.next().unwrap();
    let sigma_b = permutations.next().unwrap();
    let sigma_c = permutations.next().unwrap();
    assert!(permutations.next().is_none());

    let mut num = Poly::<_, LagrangeBasis>::zero(domain_size, stream);
    let mut denum = Poly::<_, LagrangeBasis>::zero(domain_size, stream);
    for ((col_mon, sigma_mon), non_residue_by_beta) in
        [(a_mon, sigma_a), (b_mon, sigma_b), (c_mon, sigma_c)]
            .into_iter()
            .zip(non_residues_by_beta.iter())
    {
        // (A(x) + beta*X + gamma) * (B(x) + beta*X*k1 + gamma) * (C(x) + beta*X*k2 + gamma)
        let mut num_tmp = materialize_domain_elems_bitreversed(domain_size, stream)?;
        num_tmp.scale_on(&non_residue_by_beta, stream)?;
        let col_evals = col_mon.fft_on(stream)?;
        num_tmp.add_assign_on(&col_evals, stream)?;
        num_tmp.add_constant_on(&gamma, stream)?;
        num.mul_assign_on(&num_tmp, stream)?;

        // (A(x) + beta*sigma_a(x) + gamma) * (B(x) + beta*sigma_b(x) + gamma) * (C(x) + beta*sigma_c(x) + gamma)
        let mut denum_tmp = sigma_mon.fft_on(stream)?;
        denum_tmp.scale_on(&beta, stream)?;
        denum_tmp.add_assign_on(&col_evals, stream)?;
        denum_tmp.add_constant_on(&gamma, stream)?;
        denum.mul_assign_on(&denum_tmp, stream)?;
    }

    denum.batch_inverse(stream)?;
    denum.mul_assign_on(&num, stream)?;
    denum.grand_product(stream)?;
    denum.bitreverse(stream)?;

    denum.ifft_on(stream)
}

pub fn evaluate_copy_permutation_constraints<'a, 'b, F, ITrace, IPermutations>(
    domain_size: usize,
    coset_idx: usize,
    mut trace: ITrace,
    mut permutations: IPermutations,
    grand_product_monomial: &Poly<F, MonomialBasis>,
    non_residues_by_beta: &Vec<DScalar<F>>,
    beta: &DScalar<F>,
    gamma: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<(Poly<F, CosetEvals>, Option<Poly<F, CosetEvals>>)>
where
    F: PrimeField,
    ITrace: Iterator<Item = &'a Poly<F, MonomialBasis>>,
    IPermutations: Iterator<Item = &'b Poly<F, MonomialBasis>>,
{
    let quotient_degree = 3usize;
    let padded_quotient_degree = quotient_degree.next_power_of_two();

    let a_mon = trace.next().unwrap();
    let b_mon = trace.next().unwrap();
    let c_mon = trace.next().unwrap();
    assert!(trace.next().is_none());

    let sigma_a = permutations.next().unwrap();
    let sigma_b = permutations.next().unwrap();
    let sigma_c = permutations.next().unwrap();
    assert!(permutations.next().is_none());

    assert_eq!(non_residues_by_beta.len(), 3);

    let mut num = grand_product_monomial.coset_fft_on(coset_idx, padded_quotient_degree, stream)?;

    let first_quotient_eval = if coset_idx == 0 {
        let first_lagrange_evals = first_lagrange_evals_over_coset_bitreversed(
            domain_size,
            coset_idx,
            padded_quotient_degree,
            stream,
        )?;

        let mut first_quotient_eval = num.clone_on(stream)?;
        first_quotient_eval.sub_constant_on(&DScalar::one(stream)?, stream)?;
        first_quotient_eval.mul_assign_on(&first_lagrange_evals, stream)?;

        Some(first_quotient_eval)
    } else {
        None
    };

    let mut denum = num.clone_on(stream)?;
    let h_omega = bellman::plonk::domains::Domain::new_for_size(domain_size as u64)
        .unwrap()
        .generator;
    let omega = DScalar::from_host_value_on(&h_omega, stream)?;
    denum.scale_on(&omega, stream)?;

    for ((col_mon, sigma_mon), non_residue_by_beta) in
        [(a_mon, sigma_a), (b_mon, sigma_b), (c_mon, sigma_c)]
            .into_iter()
            .zip(non_residues_by_beta.iter())
    {
        // (A(x) + beta*X + gamma) * (B(x) + beta*X*k1 + gamma) * (C(x) + beta*X*k2 + gamma)
        let mut num_tmp = materialize_domain_elems_bitreversed_for_coset(
            domain_size,
            coset_idx,
            padded_quotient_degree,
            stream,
        )?;
        num_tmp.scale_on(&non_residue_by_beta, stream)?;
        let col_evals = col_mon.coset_fft_on(coset_idx, padded_quotient_degree, stream)?;
        num_tmp.add_assign_on(&col_evals, stream)?;
        num_tmp.add_constant_on(&gamma, stream)?;
        num.mul_assign_on(&num_tmp, stream)?;

        // (A(x) + beta*sigma_a(x) + gamma) * (B(x) + beta*sigma_b(x) + gamma) * (C(x) + beta*sigma_c(x) + gamma)
        let mut denum_tmp = sigma_mon.coset_fft_on(coset_idx, padded_quotient_degree, stream)?;
        denum_tmp.scale_on(&beta, stream)?;
        denum_tmp.add_assign_on(&col_evals, stream)?;
        denum_tmp.add_constant_on(&gamma, stream)?;
        denum.mul_assign_on(&denum_tmp, stream)?;
    }

    // [z(x) * (A(x) + beta*X + gamma).. ] - [z(w*x) * (A(x) + beta*sigma_a(x) + gamma).. ]
    num.sub_assign_on(&denum, stream)?;

    Ok((num, first_quotient_eval))
}
