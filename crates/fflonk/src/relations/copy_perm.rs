use super::*;

pub fn compute_copy_perm_grand_product<'a, 'b, F: PrimeField, ITrace, IPermutations>(
    domain_size: usize,
    mut trace: ITrace,
    mut permutations: IPermutations, // TODO Lagrange basis
    non_residues_by_beta: &DScalars<F>,
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

    assert_eq!(non_residues_by_beta.len(), 3);

    let one = DScalar::one(stream)?;
    let mut num = Poly::<_, LagrangeBasis>::allocate(domain_size);
    mem::set_value(num.as_mut(), &one, stream)?;
    let mut denum = Poly::<_, LagrangeBasis>::zero(domain_size);
    mem::set_value(denum.as_mut(), &one, stream)?;

    for ((col_mon, sigma_mon), non_residue_by_beta) in
        [(a_mon, sigma_a), (b_mon, sigma_b), (c_mon, sigma_c)]
            .into_iter()
            .zip(non_residues_by_beta.iter())
    {
        let mut num_tmp = DVec::allocate_zeroed(domain_size);
        // (A(x) + beta*X + gamma) * (B(x) + beta*X*k1 + gamma) * (C(x) + beta*X*k2 + gamma)
        materialize_domain_elems_in_natural(&mut num_tmp, domain_size, stream)?;
        let mut num_tmp = Poly::from_buffer(num_tmp);
        num_tmp.scale_on(&non_residue_by_beta, stream)?;
        let mut col_evals_natural = col_mon.clone(stream)?.fft_on(stream)?;
        col_evals_natural.bitreverse(stream)?;
        if SANITY_CHECK {
            let mut tmp = col_evals_natural.clone(stream)?;
            ntt::bitreverse(tmp.as_mut(), stream)?;
            let coeffs = tmp.as_ref()[domain_size - 1..].to_vec(stream)?;
            stream.sync().unwrap();
            assert_eq!(&coeffs, &vec![F::zero(); coeffs.len()]);
        }
        num_tmp.add_assign_on(&col_evals_natural, stream)?;
        num_tmp.add_constant_on(&gamma, stream)?;
        num.mul_assign_on(&num_tmp, stream)?;

        // (A(x) + beta*sigma_a(x) + gamma) * (B(x) + beta*sigma_b(x) + gamma) * (C(x) + beta*sigma_c(x) + gamma)
        let mut denum_tmp = sigma_mon.clone(stream)?.fft_on(stream)?;
        denum_tmp.bitreverse(stream)?;
        denum_tmp.scale_on(&beta, stream)?;
        denum_tmp.add_assign_on(&col_evals_natural, stream)?;
        denum_tmp.add_constant_on(&gamma, stream)?;
        denum.mul_assign_on(&denum_tmp, stream)?;
    }

    denum.batch_inverse(stream)?;
    denum.mul_assign_on(&num, stream)?;
    denum.grand_product(stream)?;
    if SANITY_CHECK {
        assert_eq!(denum.as_ref()[0..1].to_vec(stream).unwrap()[0], F::one())
    }

    denum.ifft_on(stream)
}

pub fn evaluate_copy_permutation_constraint<'a, 'b, F, ITrace, IPermutations>(
    domain_size: usize,
    coset_idx: usize,
    mut trace: ITrace,
    mut permutations: IPermutations,
    grand_product_monomial: &Poly<F, MonomialBasis>,
    non_residues_by_beta: &DScalars<F>,
    beta: &DScalar<F>,
    gamma: &DScalar<F>,
    stream: bc_stream,
) -> CudaResult<Poly<F, CosetEvals>>
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
    let h_omega = bellman::plonk::domains::Domain::new_for_size(domain_size as u64)
        .unwrap()
        .generator;
    let omega = DScalar::from_host_value_on(&h_omega, stream)?;
    let mut shifted_grand_prod = grand_product_monomial.clone(stream)?;
    arithmetic::mul_assign_by_powers(shifted_grand_prod.as_mut(), &omega, stream)?;

    let mut num = grand_product_monomial.coset_fft_on(coset_idx, padded_quotient_degree, stream)?;

    let mut denum = shifted_grand_prod.coset_fft_on(coset_idx, padded_quotient_degree, stream)?;

    for ((col_mon, sigma_mon), non_residue_by_beta) in
        [(a_mon, sigma_a), (b_mon, sigma_b), (c_mon, sigma_c)]
            .into_iter()
            .zip(non_residues_by_beta.iter())
    {
        let mut num_tmp = Poly::zero(domain_size);
        // (A(x) + beta*X + gamma) * (B(x) + beta*X*k1 + gamma) * (C(x) + beta*X*k2 + gamma)
        materialize_domain_elems_bitreversed_for_coset(
            &mut num_tmp,
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
    divide_by_vanishing_poly_over_coset(
        &mut num,
        domain_size,
        coset_idx,
        padded_quotient_degree,
        stream,
    )?;

    Ok(num)
}

pub fn prove_copy_perm_second_constraint<F>(
    grand_product_monomial: &Poly<F, MonomialBasis>,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
{
    assert_eq!(grand_product_monomial.size(), domain_size);

    // shifting size N domain is fine since we don't need LDE here
    let h_coset_shift = F::multiplicative_generator();
    let h_coset_shift_inv = h_coset_shift.inverse().unwrap();
    let coset_shift = DScalar::from_host_value_on(&h_coset_shift, stream)?;
    let coset_shift_inv = DScalar::from_host_value_on(&h_coset_shift_inv, stream)?;

    // constraint is (z(x)-1)*L0(x) and divide by Z_H(x)

    // evaluate grand prod over shifted coset first
    let mut grand_prod_bitreversed = grand_product_monomial.as_ref().to_dvec(stream)?;
    ntt::inplace_coset_fft_for_gen_on(&mut grand_prod_bitreversed, &coset_shift, stream)?;
    let one = DScalar::one(stream)?;
    arithmetic::sub_constant(&mut grand_prod_bitreversed, &one, stream)?;

    let mut l0_bitreversed = DVec::allocate_zeroed(domain_size);
    mem::set_one(&mut l0_bitreversed[..1], stream)?;
    ntt::inplace_ifft_on(&mut l0_bitreversed, stream)?;
    ntt::inplace_coset_fft_for_gen_on(&mut l0_bitreversed, &coset_shift, stream)?;

    arithmetic::mul_assign(&mut grand_prod_bitreversed, &l0_bitreversed, stream)?;

    // vanishing is in form X^n - 1
    // coset elems are coset_shift * {w_j}
    // vanishing becomes {..coset_shift..} since w_j^n=1
    // divison becomes scaling with inverse factor
    let mut h_coset_shift_pow_minus_one = h_coset_shift.pow(&[domain_size as u64]);
    h_coset_shift_pow_minus_one.sub_assign(&F::one());
    let h_coset_shift_pow_minus_one_inv = h_coset_shift_pow_minus_one.inverse().unwrap();
    let coset_shift_pow_minus_one_inv =
        DScalar::from_host_value_on(&h_coset_shift_pow_minus_one_inv, stream)?;
    arithmetic::mul_constant(
        &mut grand_prod_bitreversed,
        &coset_shift_pow_minus_one_inv,
        stream,
    )?;

    // get monomial finally
    ntt::bitreverse(&mut grand_prod_bitreversed, stream)?;
    ntt::inplace_coset_ifft_for_gen_on(&mut grand_prod_bitreversed, &coset_shift_inv, stream)?;

    Ok(Poly::from_buffer(grand_prod_bitreversed))
}
