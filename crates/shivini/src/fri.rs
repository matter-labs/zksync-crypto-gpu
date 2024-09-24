use super::*;
use boojum::{
    cs::implementations::{transcript::Transcript, utils::precompute_twiddles_for_fft},
    fft::bitreverse_enumeration_inplace,
    worker::Worker,
};
use boojum_cuda::extension_field::VectorizedExtensionField;
use era_cudart::slice::{CudaSlice, DeviceSlice, DeviceVariable};
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::Deref;
use std::rc::Rc;

type IndexesMap = HashMap<usize, SVec<u32>>;

pub struct CodeWordChunk(DVec<VectorizedExtensionField>);

impl CodeWordChunk {
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Deref for CodeWordChunk {
    type Target = DVec<VectorizedExtensionField>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsSingleSlice for CodeWordChunk {
    fn domain_size(&self) -> usize {
        self.0.len()
    }

    fn num_polys(&self) -> usize {
        1
    }

    fn num_polys_in_base(&self) -> usize {
        2
    }

    fn as_single_slice(&self) -> &[F] {
        unsafe { std::slice::from_raw_parts(self.0.as_ptr() as *const F, self.0.len() << 1) }
    }
}

impl From<ComplexPoly<'_, CosetEvaluations>> for CodeWordChunk {
    fn from(poly: ComplexPoly<CosetEvaluations>) -> Self {
        let vec: DVec<F> = poly.into();
        let (ptr, len, capacity, alloc) = vec.into_raw_parts_with_alloc();
        let vec = DVec::from_raw_parts_in(
            ptr as *mut VectorizedExtensionField,
            len >> 1,
            capacity >> 1,
            alloc,
        );
        Self(vec)
    }
}

pub struct CodeWord {
    pub fri_lde_degree: usize,
    pub domain_size: usize,
    chunks: Vec<Rc<CodeWordChunk>>,
    pub(crate) is_base_code_word: bool,
}

impl CodeWord {
    pub fn new_base(cosets: Vec<ComplexPoly<CosetEvaluations>>) -> Self {
        let fri_lde_degree = cosets.len();
        let domain_size = cosets[0].domain_size();
        assert!(fri_lde_degree.is_power_of_two());
        assert!(domain_size.is_power_of_two());
        assert!(cosets.iter().all(|c| c.domain_size() == domain_size));
        let chunks = cosets
            .into_iter()
            .map(CodeWordChunk::from)
            .map(Rc::new)
            .collect_vec();
        Self {
            fri_lde_degree,
            domain_size,
            chunks,
            is_base_code_word: true,
        }
    }
    pub fn new_intermediate(
        fri_lde_degree: usize,
        domain_size: usize,
        chunks: Vec<CodeWordChunk>,
    ) -> Self {
        assert!(fri_lde_degree.is_power_of_two());
        assert!(domain_size.is_power_of_two());
        let chunk_len = chunks[0].len();
        assert!(chunk_len.is_power_of_two());
        assert!(chunks.iter().all(|c| c.len() == chunk_len));
        assert_eq!(fri_lde_degree * domain_size, chunks.len() * chunk_len);
        let chunks = chunks.into_iter().map(Rc::new).collect_vec();
        Self {
            fri_lde_degree,
            domain_size,
            chunks,
            is_base_code_word: false,
        }
    }

    pub fn compute_oracle<H: GpuTreeHasher>(
        &self,
        cap_size: usize,
        num_elems_per_leaf: usize,
        commitment_cache_strategy: CommitmentCacheStrategy,
    ) -> CudaResult<CommitmentCache<H>> {
        let mut cache = CommitmentCache::<H>::allocate(
            commitment_cache_strategy,
            self.chunks[0].len(),
            self.chunks.len(),
            cap_size,
            num_elems_per_leaf,
        );
        for (chunk_idx, chunk) in self.chunks.iter().enumerate() {
            cache.initialize_for_chunk(chunk_idx, chunk.deref())?;
        }
        cache.build_tree()?;
        Ok(cache)
    }

    pub fn fold(&self, coset_inv: F, challenge: &DeviceVariable<EF>) -> CudaResult<Self> {
        let len = self.chunks[0].len();
        let half_len = len >> 1;
        let chunks = if self.chunks.len() == 1 {
            let src = &self.chunks[0];
            let mut dst = dvec!(half_len);
            arith::fold_chunk(coset_inv, challenge, 0, src, &mut dst, 0)?;
            vec![CodeWordChunk(dst)]
        } else {
            let mut result_chunks = vec![];
            for (i, chunks) in self.chunks.iter().array_chunks::<2>().enumerate() {
                let mut dst = dvec!(len);
                let root_offset = i * len;
                arith::fold_chunk(coset_inv, challenge, root_offset, chunks[0], &mut dst, 0)?;
                arith::fold_chunk(
                    coset_inv,
                    challenge,
                    root_offset + half_len,
                    chunks[1],
                    &mut dst,
                    half_len,
                )?;
                result_chunks.push(CodeWordChunk(dst));
            }
            result_chunks
        };
        Ok(Self::new_intermediate(
            self.fri_lde_degree,
            self.domain_size >> 1,
            chunks,
        ))
    }

    pub fn get_coefficients(&self) -> CudaResult<[Vec<F>; 2]> {
        let len = self.domain_size * self.fri_lde_degree;
        let chunk_size = self.chunks[0].len();
        let mut c0 = vec![F::ZERO; len];
        let mut c1 = vec![F::ZERO; len];
        for (chunk, (c0, c1)) in self
            .chunks
            .iter()
            .zip(c0.chunks_mut(chunk_size).zip(c1.chunks_mut(chunk_size)))
        {
            let (s0, s1) = chunk.as_single_slice().split_at(chunk_size);
            mem::d2h(s0, c0)?;
            mem::d2h(s1, c1)?;
        }
        Ok([c0, c1])
    }
}

impl ChunkLeavesSource for CodeWord {
    fn get_chunk_leaves(&mut self, chunk_idx: usize) -> CudaResult<Rc<impl AsSingleSlice>> {
        Ok(self.chunks[chunk_idx].clone())
    }
}

pub struct FoldingOperator {
    coset_inverse: F,
}

impl FoldingOperator {
    pub fn init() -> CudaResult<Self> {
        let coset_inverse = F::multiplicative_generator().inverse().unwrap();

        Ok(Self { coset_inverse })
    }

    pub fn fold_multiple(
        &mut self,
        codeword: &CodeWord,
        challenges: Vec<EF>,
    ) -> CudaResult<CodeWord> {
        assert!(codeword.domain_size.is_power_of_two());
        let mut previous = codeword;
        let mut result = None;
        let mut d_challenges = svec!(challenges.len());
        mem::h2d(&challenges, &mut d_challenges)?;
        let challenges_slice: &DeviceSlice<EF> = (&d_challenges).into();
        for i in 0..challenges_slice.len() {
            let challenge = &challenges_slice[i];
            result = Some(previous.fold(self.coset_inverse, challenge)?);
            previous = result.as_ref().unwrap();
            self.coset_inverse.square();
        }
        Ok(result.unwrap())
    }
}

pub struct FRICache<H: GpuTreeHasher> {
    pub(crate) base_codeword: CodeWord,
    pub(crate) intermediate_codewords: Vec<CodeWord>,
    pub(crate) base_oracle: CommitmentCache<H>,
    pub(crate) intermediate_oracles: Vec<CommitmentCache<H>>,
    pub(crate) fri_lde_degree: usize,
    pub(crate) folding_schedule: Vec<usize>,
}

impl<H: GpuTreeHasher> FRICache<H> {
    pub fn flatten(&self) -> Vec<(&CodeWord, &CommitmentCache<H>)> {
        let mut fri_layers = vec![];
        fri_layers.push((&self.base_codeword, &self.base_oracle));
        for l in self
            .intermediate_codewords
            .iter()
            .zip(self.intermediate_oracles.iter())
        {
            fri_layers.push(l)
        }

        fri_layers
    }

    pub fn compute_query_indexes(
        &self,
        query_details_for_cosets: &[Vec<u32>],
    ) -> CudaResult<Vec<Vec<IndexesMap>>> {
        let fri_lde_degree = self.fri_lde_degree;
        let folding_schedule = &self.folding_schedule;
        let mut query_indexes = query_details_for_cosets.iter().cloned().collect_vec();
        let mut effective_fri_indexes_for_all = vec![];
        for (layer_idx, (codeword, oracle)) in self.flatten().into_iter().enumerate() {
            let domain_size = codeword.domain_size;
            let num_elems_per_leaf = oracle.num_elems_per_leaf;
            let schedule = folding_schedule[layer_idx];
            assert_eq!(num_elems_per_leaf, 1 << schedule);
            let chunk_size = oracle.chunk_size;
            let chunks_count = oracle.chunks_count;
            assert_eq!(fri_lde_degree * domain_size, chunk_size * chunks_count);
            let chunk_shift = chunk_size.trailing_zeros() - schedule as u32;
            let chunk_mask = (1usize << chunk_shift) - 1;
            let mut indexes = vec![];
            for (coset_idx, coset_query_indexes) in query_indexes.iter_mut().enumerate() {
                let chunk_map = coset_query_indexes
                    .iter()
                    .map(|&index| (index as usize + (coset_idx * domain_size)) >> schedule)
                    .map(|index| (index >> chunk_shift, (index & chunk_mask) as u32))
                    .into_group_map();
                let mut coset_indexes = HashMap::new();
                for (chunk_idx, queries) in chunk_map {
                    let mut d_queries = svec!(queries.len());
                    mem::h2d(&queries, &mut d_queries)?;
                    coset_indexes.insert(chunk_idx, d_queries);
                }
                indexes.push(coset_indexes);
                coset_query_indexes.iter_mut().for_each(|i| *i >>= schedule);
            }
            effective_fri_indexes_for_all.push(indexes);
        }

        Ok(effective_fri_indexes_for_all)
    }

    pub fn base_oracle_batch_query<A: GoodAllocator>(
        &mut self,
        indexes: &IndexesMap,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<H::DigestElementType, A>,
    ) -> CudaResult<()> {
        let code_word = &mut self.base_codeword;
        assert!(code_word.is_base_code_word);
        let oracle = &self.base_oracle;
        let num_elems_per_leaf = 1 << self.folding_schedule[0];
        assert_eq!(num_elems_per_leaf, oracle.num_elems_per_leaf);
        for (&chunk_idx, indexes) in indexes.iter().sorted_by_key(|&(&i, _)| i) {
            oracle.batch_query_for_chunk(
                code_word,
                chunk_idx,
                indexes,
                h_all_leaf_elems,
                h_all_proofs,
            )?;
        }
        Ok(())
    }

    pub fn intermediate_oracle_batch_query<A: GoodAllocator>(
        &mut self,
        layer_idx: usize,
        indexes: &IndexesMap,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<H::DigestElementType, A>,
    ) -> CudaResult<()> {
        assert!(layer_idx < self.folding_schedule.len());
        let code_word = &mut self.intermediate_codewords[layer_idx - 1];
        assert!(!code_word.is_base_code_word);
        let oracle = &self.intermediate_oracles[layer_idx - 1];
        let num_elems_per_leaf = 1 << self.folding_schedule[layer_idx];
        assert_eq!(num_elems_per_leaf, oracle.num_elems_per_leaf);
        for (&chunk_idx, indexes) in indexes.iter().sorted_by_key(|&(&i, _)| i) {
            oracle.batch_query_for_chunk(
                code_word,
                chunk_idx,
                indexes,
                h_all_leaf_elems,
                h_all_proofs,
            )?;
        }
        Ok(())
    }
}

pub fn compute_fri<T: Transcript<F>, H: GpuTreeHasher<Output = T::CompatibleCap>>(
    base_code_word: CodeWord,
    transcript: &mut T,
    folding_schedule: Vec<usize>,
    fri_lde_degree: usize,
    cap_size: usize,
    commitment_cache_strategy: CommitmentCacheStrategy,
    worker: &Worker,
) -> CudaResult<(FRICache<H>, [Vec<F>; 2])> {
    assert_eq!(fri_lde_degree, base_code_word.fri_lde_degree);
    let final_degree = base_code_word.domain_size >> folding_schedule.iter().sum::<usize>();
    assert!(final_degree.is_power_of_two());
    let mut operator = FoldingOperator::init()?;

    let mut intermediate_oracles = vec![];
    let mut intermediate_codewords = vec![];
    let mut prev_code_word = &base_code_word;

    for log_schedule in folding_schedule.iter().cloned() {
        let num_elems_per_leaf = 1 << log_schedule;
        let num_layers_to_skip = log_schedule;

        assert!(num_elems_per_leaf > 0);
        assert!(num_elems_per_leaf < 1 << 4);

        let current_oracle = prev_code_word.compute_oracle(
            cap_size,
            num_elems_per_leaf,
            commitment_cache_strategy,
        )?;
        let oracle_cap = current_oracle.get_tree_cap();
        intermediate_oracles.push(current_oracle);

        transcript.witness_merkle_tree_cap(&oracle_cap);
        let h_challenge = if is_dry_run()? {
            [F::ZERO; 2]
        } else {
            transcript.get_multiple_challenges_fixed::<2>()
        };

        let mut h_challenge = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_challenge);
        let mut challenge_powers = vec![];

        for _ in 0..num_layers_to_skip {
            challenge_powers.push(h_challenge);
            h_challenge.square();
        }

        let folded_code_word = operator.fold_multiple(prev_code_word, challenge_powers)?;
        intermediate_codewords.push(folded_code_word);

        prev_code_word = intermediate_codewords.last().unwrap();
    }

    let first_oracle = intermediate_oracles.drain(0..1).next().unwrap();

    // since last codeword is tiny we can do ifft and asserts on the cpu
    let last_code_word = intermediate_codewords.pop().unwrap();
    let [mut last_c0, mut last_c1] = last_code_word.get_coefficients()?;
    // FIXME: we can still construct monomials on the device for better stream handling
    synchronize_streams()?;
    bitreverse_enumeration_inplace(&mut last_c0);
    bitreverse_enumeration_inplace(&mut last_c1);

    let last_coset_inverse = operator.coset_inverse;

    let coset = last_coset_inverse.inverse().unwrap();
    // IFFT our presumable LDE of some low degree poly
    let fft_size = last_c0.len();
    let roots: Vec<F> = if is_dry_run()? {
        vec![F::ZERO; fft_size]
    } else {
        precompute_twiddles_for_fft::<_, _, _, true>(fft_size, worker, &mut ())
    };
    boojum::fft::ifft_natural_to_natural(&mut last_c0, coset, &roots[..fft_size / 2]);
    boojum::fft::ifft_natural_to_natural(&mut last_c1, coset, &roots[..fft_size / 2]);

    assert_eq!(final_degree, fft_size / fri_lde_degree);

    if !is_dry_run()? {
        // self-check
        if !boojum::config::DEBUG_SATISFIABLE {
            for el in last_c0[final_degree..].iter() {
                assert_eq!(*el, F::ZERO);
            }

            for el in last_c1[final_degree..].iter() {
                assert_eq!(*el, F::ZERO);
            }
        }
    }

    // add to the transcript
    transcript.witness_field_elements(&last_c0[..final_degree]);
    transcript.witness_field_elements(&last_c1[..final_degree]);

    // now we should do some PoW and we are good to go

    let monomial_form_0 = last_c0[..(fft_size / fri_lde_degree)].to_vec();
    let monomial_form_1 = last_c1[..(fft_size / fri_lde_degree)].to_vec();
    let fri_holder = FRICache {
        base_codeword: base_code_word,
        base_oracle: first_oracle,
        intermediate_codewords,
        intermediate_oracles,
        folding_schedule,
        fri_lde_degree,
    };
    Ok((fri_holder, [monomial_form_0, monomial_form_1]))
}
