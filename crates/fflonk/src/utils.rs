use super::*;
use fflonk::utils::*;

pub(crate) fn commit_monomial<E: Engine>(
    monomial: &Poly<E::Fr, MonomialBasis>,
    stream: bc_stream,
) -> CudaResult<E::G1Affine> {
    todo!()
}

pub(crate) fn materialize_domain_elems_bitreversed<F>(
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<Poly<F, LagrangeBasis>>
where
    F: PrimeField,
{
    todo!()
}

pub(crate) fn materialize_domain_elems_bitreversed_for_coset<F>(
    domain_size: usize,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<Poly<F, CosetEvals>>
where
    F: PrimeField,
{
    todo!()
}

pub(crate) fn first_lagrange_evals_over_coset_bitreversed<F>(
    domain_size: usize,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<Poly<F, CosetEvals>>
where
    F: PrimeField,
{
    todo!()
}

pub(crate) fn materialize_vanishing_inv_over_coset<F>(
    domain_size: usize,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
{
    todo!()
}

// TODO ExactSizeItarator
pub(crate) fn combine_monomials<'a, F, I>(
    iter: I,
    combined_monomial: &mut Poly<F, MonomialBasis>,
    num_polys: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
    I: Iterator<Item = &'a Poly<F, MonomialBasis>>,
{
    let mut full_degree = 0;
    for (poly_idx, monomial) in iter.enumerate() {
        full_degree += monomial.size();
        intersperse_coeffs(
            monomial.as_ref(),
            combined_monomial.as_mut(),
            poly_idx,
            num_polys,
            stream,
        )?;
    }
    assert!(full_degree <= combined_monomial.size());

    Ok(())
}

// compute alpha^i*Z_{T\Si}(x)
pub(crate) fn evaluate_set_difference_monomials_at_y<F>(
    h_z: F,
    h_z_omega: F,
    h_y: F,
    h_alpha_pows: [F; 2],
    stream: bc_stream,
) -> CudaResult<[DScalar<F>; 3]>
where
    F: PrimeField,
{
    let all_sparse_polys = construct_set_difference_monomials(h_z, h_z_omega, 8, 3, 4, false);

    let mut all_sparse_polys_at_y = [
        DScalar::one(stream)?,
        DScalar::one(stream)?,
        DScalar::one(stream)?,
    ];

    let mut sparse_polys_for_trace_at_y = F::one();
    for (degree, constant) in all_sparse_polys[0].clone() {
        let mut y_pow = h_y.pow(&[degree as u64]);
        y_pow.sub_assign(&constant);
        sparse_polys_for_trace_at_y.mul_assign(&y_pow);
    }
    let inv_sparse_polys_for_trace_at_y = sparse_polys_for_trace_at_y.inverse().unwrap();

    for (idx, (sparse_polys, sparse_poly_at_y)) in all_sparse_polys
        .into_iter()
        .skip(1)
        .zip(all_sparse_polys_at_y.iter_mut())
        .enumerate()
    {
        let mut product = F::one();
        for (degree, constant) in sparse_polys {
            let mut y_pow = h_y.pow(&[degree as u64]);
            y_pow.sub_assign(&constant);
            product.mul_assign(&y_pow);
        }
        product.mul_assign(&inv_sparse_polys_for_trace_at_y);
        if idx > 0 {
            product.mul_assign(&h_alpha_pows[idx - 1]);
        }
        *sparse_poly_at_y = DScalar::from_host_value_on(&product, stream)?;
    }

    Ok(all_sparse_polys_at_y)
}

pub(crate) fn multiply_monomial_with_multiple_sparse_polys<F>(
    poly: &mut Poly<F, MonomialBasis>,
    pairs: &[(usize, F)],
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    todo!()
}

pub(crate) fn multiply_sparse_polys<F>(polys: &[(usize, F)]) -> Vec<F>
where
    F: PrimeField,
{
    todo!()
}

pub(crate) fn sub_assign_mixed_degree_polys<F, A, B>(
    poly: &mut A,
    other: &B,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
    A: AsMut<DSlice<F>>,
    B: AsRef<DSlice<F>>,
{
    todo!()
}

pub(crate) fn divide_by_chunking_in_values<F>(
    poly: &mut Poly<F, MonomialBasis>,
    divisor: &Poly<F, MonomialBasis>,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    assert_eq!(poly.size(), 9 * domain_size);

    let mut denum = Poly::<_, MonomialBasis>::zero(domain_size, stream);
    mem::d2d_on_stream(
        divisor.as_ref(),
        &mut denum.as_mut()[..divisor.as_ref().len()],
        stream,
    )?;
    let mut denum_evals = denum.fft_on(stream)?;
    denum_evals.batch_inverse(stream)?;

    for num_chunk in poly.as_mut().chunks_mut(domain_size) {
        ntt::inplace_fft_on(num_chunk, stream)?;
        unsafe {
            arithmetic::mul_assign(num_chunk, denum_evals.as_ref(), stream)?;
        }
        ntt::inplace_ifft_on(num_chunk, stream)?;
    }

    Ok(())
}

pub(crate) fn construct_r_monomials<F>(
    mut all_evaluations: Vec<F>,
    aux_evaluations: Vec<F>,
    h0: (F, Option<F>),
    h1: (F, Option<F>),
    h2: (F, F),
) -> [Vec<F>; 3]
where
    F: PrimeField,
{
    let mut aux_evals_iter = aux_evaluations.into_iter();
    let main_gate_quotient_at_z = aux_evals_iter.next().unwrap();
    let copy_perm_first_quotient_at_z = aux_evals_iter.next().unwrap();
    let copy_perm_second_quotient_at_z = aux_evals_iter.next().unwrap();
    let copy_perm_first_quotient_at_z_omega = aux_evals_iter.next().unwrap();
    let copy_perm_second_quotient_at_z_omega = aux_evals_iter.next().unwrap();
    assert!(aux_evals_iter.next().is_none());

    let setup_evals = all_evaluations.drain(..8).collect();
    let mut trace_evals: Vec<_> = all_evaluations.drain(..3).collect();
    trace_evals.push(main_gate_quotient_at_z);

    let grand_prod_eval_at_z_omega = all_evaluations.pop().unwrap();
    let grand_prod_eval_at_z = all_evaluations.pop().unwrap();
    assert!(all_evaluations.is_empty());

    let copy_perm_evals = vec![
        grand_prod_eval_at_z,
        copy_perm_first_quotient_at_z,
        copy_perm_second_quotient_at_z,
    ];
    let copy_perm_evals_shifted = vec![
        grand_prod_eval_at_z_omega,
        copy_perm_first_quotient_at_z_omega,
        copy_perm_second_quotient_at_z_omega,
    ];

    let [setup_omega, trace_omega, copy_perm_omega] = compute_generators(8, 4, 3);

    let h_setup_r_monomial = interpolate_union_set(setup_evals, vec![], 8, h0, setup_omega, false);
    let h_trace_r_monomial = interpolate_union_set(trace_evals, vec![], 8, h1, trace_omega, false);
    let h_copy_perm_r_monomial = interpolate_union_set(
        copy_perm_evals,
        copy_perm_evals_shifted,
        8,
        (h2.0, Some(h2.1)),
        copy_perm_omega,
        true,
    );

    [
        h_setup_r_monomial,
        h_trace_r_monomial,
        h_copy_perm_r_monomial,
    ]
}

pub(crate) fn compute_flattened_lagrange_basis_inverses<F>(h0: F, h1: F, h2: (F, F), y: F) -> Vec<F>
where
    F: PrimeField,
{
    let [_, _, omega_copy_perm] = compute_generators::<F>(8usize, 4, 3);
    let inverses_for_setup = fflonk::utils::compute_lagrange_basis_inverses(8, h0, y);
    let inverses_for_trace = fflonk::utils::compute_lagrange_basis_inverses(4, h1, y);
    let (inverses_for_copy_perm, _) =
        compute_lagrange_basis_inverses_for_union_set(3, h2.0, h2.1, y, omega_copy_perm);
    let mut flattened = inverses_for_setup;
    flattened.extend(inverses_for_trace);
    flattened.extend(inverses_for_copy_perm);

    flattened
}
