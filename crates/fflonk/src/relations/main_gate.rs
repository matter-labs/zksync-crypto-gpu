use super::*;

pub fn evaluate_main_gate_constraints<'a, F, ITrace, ISelectors>(
    coset_idx: usize,
    domain_size: usize,
    mut trace: ITrace,
    mut main_gate_selectors: ISelectors,
    public_inputs: &DVec<F>,
    stream: bc_stream,
) -> CudaResult<Poly<F, CosetEvals>>
where
    F: PrimeField,
    ITrace: Iterator<Item = &'a Poly<F, MonomialBasis>>,
    ISelectors: Iterator<Item = &'a Poly<F, MonomialBasis>>,
{
    let quotient_degree = 2;
    let a_mon = trace.next().unwrap();
    let b_mon = trace.next().unwrap();
    let c_mon = trace.next().unwrap();
    assert!(trace.next().is_none());

    let qa_mon = main_gate_selectors.next().unwrap();
    let qb_mon = main_gate_selectors.next().unwrap();
    let qc_mon = main_gate_selectors.next().unwrap();
    let qab_mon = main_gate_selectors.next().unwrap();
    let qconst_mon = main_gate_selectors.next().unwrap();
    assert!(main_gate_selectors.next().is_none());

    // don't use tmp allocations, rather rely on drop()
    let mut a_evals = a_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;
    let mut b_evals = b_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;

    let mut sum = qab_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;
    sum.mul_assign_on(&a_evals, stream)?;
    sum.mul_assign_on(&b_evals, stream)?;

    let qa_evals = qa_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;
    a_evals.mul_assign_on(&qa_evals, stream)?;
    drop_on(qa_evals, stream);
    sum.add_assign_on(&a_evals, stream)?;
    drop_on(a_evals, stream);

    let qb_evals = qb_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;
    b_evals.mul_assign_on(&qb_evals, stream)?;
    drop_on(qb_evals, stream);
    sum.add_assign_on(&b_evals, stream)?;
    drop_on(b_evals, stream);

    let mut c_evals = c_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;
    let qc_evals = qc_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;
    c_evals.mul_assign_on(&qc_evals, stream)?;
    drop_on(qc_evals, stream);
    sum.add_assign_on(&c_evals, stream)?;
    drop_on(c_evals, stream);

    let qconst_evals = qconst_mon.coset_fft_on(coset_idx, quotient_degree, stream)?;
    sum.add_assign_on(&qconst_evals, stream)?;
    drop_on(qconst_evals, stream);

    let mut public_input_evals = Poly::<F, LagrangeBasis>::zero(domain_size, stream);
    mem::d2d_on(
        &public_inputs,
        &mut public_input_evals.as_mut()[..public_inputs.len()],
        stream,
    )?;
    let public_input_monomial = public_input_evals.ifft_on(stream)?;
    let public_input_evals =
        public_input_monomial.coset_fft_on(coset_idx, quotient_degree, stream)?;
    sum.add_assign_on(&public_input_evals, stream)?;

    divide_by_vanishing_poly_over_coset(&mut sum, domain_size, coset_idx, quotient_degree, stream)?;

    Ok(sum)
}
