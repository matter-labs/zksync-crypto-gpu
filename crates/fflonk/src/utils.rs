use super::*;
use bellman::plonk::domains::Domain;
use fflonk::utils::*;
use memmap2::Mmap;
use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::slice::{ChunksExact, Iter};

pub(crate) fn commit_monomial<E: Engine>(
    monomial: &Poly<E::Fr, MonomialBasis>,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<E::G1Affine> {
    msm::msm::<E>(monomial.as_ref(), domain_size, stream)
}

pub(crate) fn commit_padded_monomial<E: Engine>(
    monomial: &Poly<E::Fr, MonomialBasis>,
    common_combined_degree: usize,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<E::G1Affine> {
    assert!(common_combined_degree <= monomial.size());
    assert_eq!(
        MAX_COMBINED_DEGREE_FACTOR * domain_size,
        common_combined_degree
    );
    msm::msm::<E>(
        &monomial.as_ref()[..common_combined_degree],
        domain_size,
        stream,
    )
}

pub(crate) fn materialize_domain_elems_in_natural<F>(
    into: &mut DVec<F>,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    // let mut evals = DVec::allocate_zeroed_on(domain_size, pool, stream);
    materialize_domain_elems(into.as_mut(), domain_size, false, stream)?;

    Ok(())
}

pub fn bitreverse_idx(n: usize, l: usize) -> usize {
    let mut r = n.reverse_bits();
    // now we need to only use the bits that originally were "last" l, so shift

    r >>= (std::mem::size_of::<usize>() * 8) - l;

    r
}

pub fn get_bitreversed_coset_factor<F>(domain_size: usize, coset_idx: usize, lde_factor: usize) -> F
where
    F: PrimeField,
{
    let lde_gen = Domain::new_for_size((domain_size * lde_factor) as u64)
        .unwrap()
        .generator;

    let mut h_powers_of_coset_gen = vec![];
    let mut current = F::one();
    for _ in 0..lde_factor {
        h_powers_of_coset_gen.push(current);
        current.mul_assign(&lde_gen);
    }

    h_powers_of_coset_gen
        .iter_mut()
        .for_each(|el| el.mul_assign(&F::multiplicative_generator()));

    let bitreversed_idx = bitreverse_idx(coset_idx, lde_factor.trailing_zeros() as usize);
    h_powers_of_coset_gen[bitreversed_idx]
}

pub fn divide_by_vanishing_poly_over_coset<F>(
    poly: &mut Poly<F, CosetEvals>,
    domain_size: usize,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    // vanishing is in form X^n - 1
    // coset elems are (coset_shift * coset_gen^i) * {w_j}
    // vanishing becomes {..coset_shift * coset_gen^i..} since w_j^n=1
    // divison becomes scaling with inverse factor

    let h_coset_factor = get_bitreversed_coset_factor::<F>(domain_size, coset_idx, lde_factor);
    let h_coset_factor_pow = h_coset_factor.pow(&[domain_size as u64]);
    let mut h_coset_factor_pow_minus_one = h_coset_factor_pow;
    h_coset_factor_pow_minus_one.sub_assign(&F::one());
    let h_coset_factor_pow_minus_one_inv = h_coset_factor_pow_minus_one.inverse().unwrap();
    let coset_factor_pow_minus_one_inv =
        DScalar::from_host_value_on(&h_coset_factor_pow_minus_one_inv, stream)?;

    poly.scale_on(&coset_factor_pow_minus_one_inv, stream)
}
pub(crate) fn materialize_domain_elems_bitreversed_for_coset<F>(
    coset_domain_elems: &mut Poly<F, CosetEvals>,
    domain_size: usize,
    coset_idx: usize,
    lde_factor: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    let h_coset_factor = get_bitreversed_coset_factor(domain_size, coset_idx, lde_factor);
    let coset_factor = DScalar::from_host_value_on(&h_coset_factor, stream)?;
    materialize_domain_elems(coset_domain_elems.as_mut(), domain_size, true, stream)?;
    coset_domain_elems.scale_on(&coset_factor, stream)?;

    Ok(())
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
pub fn construct_set_difference_monomials<F: PrimeField>(
    h_z: F,
    h_z_omega: F,
) -> [Vec<(usize, F)>; 4] {
    let mut h_z_negated = h_z;
    h_z_negated.negate();
    let mut h_z_omega_negated = h_z_omega;
    h_z_omega_negated.negate();

    fflonk::utils::construct_set_difference_monomials(
        h_z_negated,
        h_z_omega_negated,
        8,
        4,
        3,
        false,
    )
}

#[inline(always)]
pub(crate) fn product_of_pairs<F: PrimeField>(pairs: &[(usize, F)], point: F) -> F {
    let mut product = F::one();
    for (degree, constant) in pairs {
        let mut pow = point.pow(&[*degree as u64]);
        pow.add_assign(&constant);
        product.mul_assign(&pow);
    }
    product
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
    // constant term of each sparse polys already negated
    let all_sparse_poly_pairs = construct_set_difference_monomials(h_z, h_z_omega);

    let mut all_sparse_polys_at_y = [
        DScalar::one(stream)?,
        DScalar::one(stream)?,
        DScalar::one(stream)?,
    ];

    let sparse_polys_for_setup_at_y = product_of_pairs(&all_sparse_poly_pairs[0], h_y);
    let inv_sparse_polys_for_setup_at_y = sparse_polys_for_setup_at_y.inverse().unwrap();
    dbg!(inv_sparse_polys_for_setup_at_y);

    for ((sparse_poly_pairs, sparse_poly_at_y), alpha) in all_sparse_poly_pairs
        .iter()
        .skip(1)
        .take(2)
        .zip(all_sparse_polys_at_y.iter_mut().take(2))
        .zip(h_alpha_pows.iter())
    {
        let mut product = product_of_pairs(sparse_poly_pairs, h_y);
        product.mul_assign(&inv_sparse_polys_for_setup_at_y);
        product.mul_assign(alpha);
        sparse_poly_at_y.copy_from_host_value_on(&product, stream)?;
    }

    let mut product_of_sparse_polys_at_y =
        product_of_pairs(all_sparse_poly_pairs.last().unwrap(), h_y);
    product_of_sparse_polys_at_y.negate();
    product_of_sparse_polys_at_y.mul_assign(&inv_sparse_polys_for_setup_at_y);
    if SANITY_CHECK {
        let mut expected = h_y.pow(&[8u64]);
        expected.sub_assign(&h_z);
        expected.negate();
        assert_eq!(expected, product_of_sparse_polys_at_y);
    }
    all_sparse_polys_at_y[2].copy_from_host_value_on(&product_of_sparse_polys_at_y, stream)?;

    Ok(all_sparse_polys_at_y)
}

pub(crate) fn multiply_monomial_with_multiple_sparse_polys_inplace<F>(
    poly: &mut Poly<F, MonomialBasis>,
    pairs_negated: Vec<(usize, F)>,
    poly_degree: usize,
    stream: bc_stream,
) -> CudaResult<()>
where
    F: PrimeField,
{
    // degree contrib from set difference monomials are [10, 14, 12] respectively
    // and our monomial already has space for product terms from pairs

    // taking poly mutable indirectly prevents fragmentation

    // P(X)*(X^n-k)
    // multiplication with X^k term shifts monomial
    // then sum with scaled -k*P(X), recall that constant terms already negated
    let degrees: Vec<_> = pairs_negated
        .iter()
        .cloned()
        .map(|(degree, _)| degree)
        .collect();
    let total_degree: usize = degrees.iter().cloned().sum();
    assert!(poly_degree + total_degree <= poly.size());

    let h_constants: Vec<_> = pairs_negated
        .into_iter()
        .map(|(_, constant)| constant)
        .collect();

    let constants = DScalars::from_host_scalars_on(&h_constants, stream)?;

    let mut current_degree = poly_degree;
    for (degree, constant) in degrees.into_iter().zip(constants.iter()) {
        let mut tmp = poly.clone(stream)?;
        // shift coeffs
        mem::d2d_on(
            &tmp.as_ref()[..current_degree],
            &mut poly.as_mut()[degree..(degree + current_degree)],
            stream,
        )?;
        mem::set_zero(&mut poly.as_mut()[..degree], stream)?;
        mem::set_zero(&mut poly.as_mut()[degree + current_degree..], stream)?;

        tmp.scale_on(constant, stream)?;
        poly.add_assign_on(&tmp, stream)?;

        current_degree += degree;
    }

    Ok(())
}

pub(crate) fn compute_product_of_multiple_sparse_monomials<F>(pairs: &[(usize, F)]) -> Vec<F>
where
    F: PrimeField,
{
    let (degree, constant_term) = pairs[0].clone();
    let mut result = vec![F::zero(); degree + 1];
    result[0] = constant_term;
    result[degree] = F::one();
    if pairs.len() == 1 {
        return result;
    }

    for (degree, constant_term) in pairs.iter().skip(1).cloned() {
        let mut sparse = vec![F::zero(); degree + 1];
        sparse[0] = constant_term;
        sparse[degree] = F::one();
        result = multiply_monomials(&result, &sparse);
    }

    result
}

pub(crate) fn multiply_monomials<F>(this: &[F], other: &[F]) -> Vec<F>
where
    F: PrimeField,
{
    let mut result = vec![F::zero(); this.len() + other.len()];
    for (i, p1) in this.iter().enumerate() {
        for (j, p2) in other.iter().enumerate() {
            let mut tmp = p1.clone();
            tmp.mul_assign(p2);
            result[i + j].add_assign(&tmp);
        }
    }
    result
}

pub(crate) fn divide_in_values<F>(
    num_monomial: Poly<F, MonomialBasis>,
    denum_coeffs: Vec<F>,
    domain_size: usize,
    stream: bc_stream,
) -> CudaResult<Poly<F, MonomialBasis>>
where
    F: PrimeField,
{
    assert!(denum_coeffs.len() < num_monomial.size());

    let padding = DVec::<F>::allocate_zeroed(16 * domain_size - num_monomial.size());
    let mut padded_num = DVec::allocate_zeroed(16 * domain_size);
    mem::d2d_on(
        num_monomial.as_ref(),
        &mut padded_num.as_mut()[..num_monomial.size()],
        stream,
    )?;
    drop(num_monomial);
    drop(padding);
    let mut padded_denum = DVec::allocate_zeroed(16 * domain_size);
    mem::h2d_on(
        &denum_coeffs,
        &mut padded_denum.as_mut()[..denum_coeffs.len()],
        stream,
    )?;

    ntt::inplace_fft_on(&mut padded_num, stream)?;
    ntt::inplace_fft_on(&mut padded_denum, stream)?;
    arithmetic::batch_inverse(&mut padded_denum, stream)?;
    arithmetic::mul_assign(&mut padded_denum, &padded_num, stream)?;
    ntt::bitreverse(&mut padded_denum, stream)?;
    ntt::inplace_ifft_on(&mut padded_denum, stream)?;

    let quotient = Poly::from_buffer(padded_denum);
    let quotient_monomial = quotient.trim_to_degree(9 * domain_size, stream)?;
    drop(padded_num);

    Ok(quotient_monomial)
}

pub fn compute_opening_points<F>(
    h_r: F,
    h_z: F,
    h_z_omega: F,
    power: usize,
    domain_size: usize,
) -> [F; 4]
where
    F: PrimeField,
{
    let (h0, h1, h2) = fflonk::utils::compute_opening_points::<F>(
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

    [h0.0, h1.0, h2.0, h2.1]
}

pub(crate) fn construct_r_monomials<F>(
    mut all_evaluations: Vec<F>,
    aux_evaluations: Vec<F>,
    opening_points: [F; 4],
) -> [Vec<F>; 3]
where
    F: PrimeField,
{
    let mut aux_evals_iter = aux_evaluations.into_iter();
    let main_gate_quotient_at_z = aux_evals_iter.next().unwrap();
    let copy_perm_first_quotient_at_z = aux_evals_iter.next().unwrap();
    let copy_perm_second_quotient_at_z = aux_evals_iter.next().unwrap();
    assert!(aux_evals_iter.next().is_none());

    let setup_evals = all_evaluations.drain(..8).collect();
    let mut trace_evals: Vec<_> = all_evaluations.drain(..3).collect();
    trace_evals.push(main_gate_quotient_at_z);

    let mut all_evaluations_iter = all_evaluations.into_iter();
    let grand_prod_eval_at_z = all_evaluations_iter.next().unwrap();
    let grand_prod_eval_at_z_omega = all_evaluations_iter.next().unwrap();
    let copy_perm_first_quotient_at_z_omega = all_evaluations_iter.next().unwrap();
    let copy_perm_second_quotient_at_z_omega = all_evaluations_iter.next().unwrap();
    assert!(all_evaluations_iter.next().is_none());

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

    let [h0, h1, h2, h2_shifted] = opening_points;

    let h_setup_r_monomial =
        interpolate_union_set(setup_evals, vec![], 8, (h0, None), setup_omega, false);
    let h_trace_r_monomial =
        interpolate_union_set(trace_evals, vec![], 4, (h1, None), trace_omega, false);
    let h_copy_perm_r_monomial = interpolate_union_set(
        copy_perm_evals,
        copy_perm_evals_shifted,
        3,
        (h2, Some(h2_shifted)),
        copy_perm_omega,
        true,
    );

    [
        h_setup_r_monomial,
        h_trace_r_monomial,
        h_copy_perm_r_monomial,
    ]
}

pub(crate) fn compute_flattened_lagrange_basis_inverses<F>(opening_points: [F; 4], y: F) -> Vec<F>
where
    F: PrimeField,
{
    let [h0, h1, h2, h2_shifted] = opening_points;
    let [_, _, omega_copy_perm] = compute_generators::<F>(8usize, 4, 3);
    let inverses_for_setup = fflonk::utils::compute_lagrange_basis_inverses(8, h0, y);
    let inverses_for_trace = fflonk::utils::compute_lagrange_basis_inverses(4, h1, y);
    let (inverses_for_copy_perm, _) = fflonk_cpu::compute_lagrange_basis_inverses_for_union_set(
        3,
        h2,
        h2_shifted,
        y,
        omega_copy_perm,
    );
    let mut flattened = inverses_for_setup;
    flattened.extend(inverses_for_trace);
    flattened.extend(inverses_for_copy_perm);

    flattened
}

pub fn get_g2_elems_from_compact_crs<E: Engine>() -> [E::G2Affine; 2] {
    assert!(std::mem::size_of::<E::Fqe>() == 64 || std::mem::size_of::<E::Fqe>() == 72);
    use bellman::{CurveAffine, EncodedPoint};
    let g2_bases = if let Ok(compact_crs_file_path) = std::env::var("COMPACT_CRS_FILE") {
        use byteorder::ReadBytesExt;
        let mut reader = std::fs::File::open(compact_crs_file_path).unwrap();
        let num_g1 = reader.read_u64::<byteorder::BigEndian>().unwrap();

        // skip num_g1 * 32 bytes
        use std::io::Seek;

        let num_bytes_to_skip = num_g1 * 64;
        reader
            .seek(std::io::SeekFrom::Current(num_bytes_to_skip as i64))
            .unwrap();

        let num_g2 = reader.read_u64::<byteorder::BigEndian>().unwrap();
        assert!(num_g2 == 2u64);

        let mut g2_bases = Vec::with_capacity(num_g2 as usize);
        use bellman::compact_bn256::G2Affine;
        let mut g2_repr = <G2Affine as CurveAffine>::Uncompressed::empty();
        for _ in 0..num_g2 {
            std::io::Read::read_exact(&mut reader, g2_repr.as_mut()).unwrap();
            let p = g2_repr
                .into_affine()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                .unwrap();
            let (x, y) = p.as_xy();
            let p = unsafe {
                let x = *(x as *const bellman::bn256::Fq2 as *const E::Fqe);
                let y = *(y as *const bellman::bn256::Fq2 as *const E::Fqe);
                E::G2Affine::from_xy_checked(x.clone(), y.clone()).unwrap()
            };

            g2_bases.push(p);
        }

        g2_bases
    } else {
        // crs 42
        let mut g2_bases = vec![E::G2Affine::one(); 2];

        let gen = E::Fr::from_str("42").unwrap();

        g2_bases[1] = g2_bases[1].mul(gen.into_repr()).into_affine();

        g2_bases
    };

    [g2_bases[0], g2_bases[1]]
}

fn transmuted_len<T, U>(len: usize) -> usize {
    let size_of_t = size_of::<T>();
    let size_of_u = size_of::<U>();
    if size_of_t > size_of_u {
        assert_eq!(size_of_t % size_of_u, 0);
        len * (size_of_t / size_of_u)
    } else {
        assert_eq!(size_of_u % size_of_t, 0);
        len / (size_of_u / size_of_t)
    }
}

pub(crate) unsafe fn transmute_slice<T, U>(slice: &[T]) -> &[U] {
    let ptr = slice.as_ptr();
    assert_eq!(ptr.align_offset(align_of::<U>()), 0);
    let ptr = ptr as *const U;
    let len = transmuted_len::<T, U>(slice.len());
    std::slice::from_raw_parts(ptr, len)
}

pub(crate) unsafe fn transmute_slice_mut<T, U>(slice: &mut [T]) -> &mut [U] {
    let ptr = slice.as_mut_ptr();
    assert_eq!(ptr.align_offset(align_of::<U>()), 0);
    let ptr = ptr as *mut U;
    let len = transmuted_len::<T, U>(slice.len());
    std::slice::from_raw_parts_mut(ptr, len)
}

pub enum BigArrayStorage<T> {
    Vector(Vec<T>),
    Mmap(MmapStorage<T>),
}

impl<T> BigArrayStorage<T> {
    pub fn len(&self) -> usize {
        match self {
            Self::Vector(v) => v.len(),
            Self::Mmap(m) => m.len(),
        }
    }
}

impl<T> Deref for BigArrayStorage<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Vector(v) => v.as_slice(),
            Self::Mmap(m) => m.deref(),
        }
    }
}

pub struct Big2DArrayStorage<T, const N: usize> {
    pub(crate) inner: BigArrayStorage<T>,
}

impl<T, const N: usize> Big2DArrayStorage<T, N> {
    pub fn new(storage: BigArrayStorage<T>) -> Self {
        assert_eq!(storage.len() % N, 0);
        Self{ inner: storage}
    }

    pub fn len(&self) -> usize {
        N
    }
    
    pub fn iter(&self) -> ChunksExact<'_, T> {
        let chunk_size = self.inner.len() / N;
        self.inner.deref().chunks_exact(chunk_size)
    }
}

pub(crate) struct MmapStorage<T> {
    file: File,
    map: Mmap,
    _marker: std::marker::PhantomData<T>,
}

impl<T> MmapStorage<T> {
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        assert_eq!(map.as_ref().len() % size_of::<T>(), 0);
        assert_eq!(map.as_ptr().align_offset(align_of::<T>()), 0);
        let result = Self {
            file,
            map,
            _marker: std::marker::PhantomData,
        };
        
        Ok(result)
    }
    
    pub fn len(&self) -> usize {
        self.deref().len()
    }
}

impl<T> Deref for MmapStorage<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { transmute_slice(&*self.map) }
    }
}

// pub(crate) struct MmapArrayStorage<T, const N: usize>(MmapStorage<T>);
// 
// pub struct BufferedTransferBatch<'a> {
//     functions: Vec<HostFn<'a>>,
// }
//
// impl<'a> BufferedTransferBatch<'a> {
//     pub fn new() -> Self {
//         Self { functions: vec![] }
//     }
// }
//
// pub(crate) struct TransferBuffer {
//     pub buffer: Mutex<HostAllocation<u8>>,
//     pub stream: CudaStream,
//     pub event: CudaEvent,
// }
//
// impl TransferBuffer {
//     pub fn new(size: usize) -> era_cudart::result::CudaResult<Self> {
//         let allocation = HostAllocation::alloc(size, CudaHostAllocFlags::DEFAULT)?;
//         let buffer = Mutex::new(allocation);
//         let stream = CudaStream::create()?;
//         let event = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
//         Ok(Self {
//             buffer,
//             stream,
//             event,
//         })
//     }
// }
//
// pub struct BufferedCudaTransferEngine {
//     pub(crate) buffer_size: usize,
//     pub(crate) buffers: [TransferBuffer; 2],
//     next_buffer: usize,
// }
//
// impl BufferedCudaTransferEngine {
//     pub fn new(buffer_size: usize) -> era_cudart::result::CudaResult<Self> {
//         let buffers = [
//             TransferBuffer::new(buffer_size)?,
//             TransferBuffer::new(buffer_size)?,
//         ];
//         Ok(Self { buffer_size, buffers, next_buffer: 0 })
//     }
//
//     pub fn h2d<'a, T>(
//         &'a mut self,
//         src: &'a [T],
//         dst: &'a mut DeviceSlice<T>,
//         stream: &'a CudaStream,
//         batch: &mut BufferedTransferBatch<'a>,
//     ) -> CudaResult<()> {
//         assert_eq!(src.len(), dst.len());
//         let src = unsafe { transmute_slice(src) };
//         let dst = unsafe { transmute_slice_mut(dst.as_mut()) };
//
//         Ok(())
//     }
// }
