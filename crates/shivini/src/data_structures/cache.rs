use super::*;
use crate::cs::GpuSetup;
use crate::data_structures::{GenericPolynomialStorageLayout, GenericStorage};
use crate::gpu_proof_config::GpuProofConfig;
use crate::oracle::SubTree;
use crate::poly::{CosetEvaluations, LagrangeBasis, MonomialBasis};
use crate::prover::{compute_quotient_degree, gpu_prove_from_external_witness_data_with_cache_strategy_inner};
use boojum::cs::implementations::prover::ProofConfig;
use boojum::cs::implementations::transcript::Transcript;
use boojum::cs::implementations::verifier::{VerificationKey, VerificationKeyCircuitGeometry};
use boojum::cs::implementations::witness::WitnessVec;
use boojum::worker::Worker;
use era_cudart_sys::CudaError::ErrorMemoryAllocation;
use itertools::Itertools;
use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;

pub trait ChunkLeavesSource {
    fn get_chunk_leaves(&mut self, chunk_idx: usize) -> CudaResult<Rc<impl AsSingleSlice>>;
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum PolynomialsCacheStrategy {
    InPlace,
    CacheMonomials,
    CacheMonomialsAndFirstCoset,
    CacheMonomialsAndFriCosets,
    CacheEvaluationsAndMonomials,
    CacheEvaluationsMonomialsAndFirstCoset,
    CacheEvaluationsMonomialsAndFriCosets,
    CacheEvaluationsAndAllCosets,
}

use PolynomialsCacheStrategy::*;

impl PolynomialsCacheStrategy {
    pub fn required_storages_count(&self, fri_lde_degree: usize, max_lde_degree: usize) -> usize {
        match self {
            InPlace => 1,
            CacheMonomials => 1,
            CacheMonomialsAndFirstCoset => 2,
            CacheMonomialsAndFriCosets => 1 + fri_lde_degree,
            CacheEvaluationsAndMonomials => 2,
            CacheEvaluationsMonomialsAndFirstCoset => 3,
            CacheEvaluationsMonomialsAndFriCosets => 2 + fri_lde_degree,
            CacheEvaluationsAndAllCosets => 1 + max_lde_degree,
        }
    }

    pub fn cache_evaluations(&self) -> bool {
        match self {
            InPlace | CacheMonomials | CacheMonomialsAndFirstCoset | CacheMonomialsAndFriCosets => {
                false
            }
            CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets
            | CacheEvaluationsAndAllCosets => true,
        }
    }

    pub fn cache_monomials(&self) -> bool {
        match self {
            InPlace | CacheEvaluationsAndAllCosets => false,
            CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets
            | CacheMonomials
            | CacheMonomialsAndFirstCoset
            | CacheMonomialsAndFriCosets => true,
        }
    }
}

pub struct PolynomialsCache<L: GenericPolynomialStorageLayout> {
    pub strategy: PolynomialsCacheStrategy,
    pub layout: L,
    pub domain_size: usize,
    pub fri_lde_degree: usize,
    pub max_lde_degree: usize,
    evaluations: Option<Rc<GenericStorage<LagrangeBasis, L>>>,
    monomials: Option<Rc<GenericStorage<MonomialBasis, L>>>,
    coset_evaluations: BTreeMap<usize, Rc<GenericStorage<CosetEvaluations, L>>>,
    uninitialized_storages: Vec<GenericStorage<Undefined, L>>,
}

impl<L: GenericPolynomialStorageLayout> PolynomialsCache<L> {
    fn required_storages_count(&self) -> usize {
        self.strategy
            .required_storages_count(self.fri_lde_degree, self.max_lde_degree)
    }

    fn is_storage_borrowed(&self) -> bool {
        self.uninitialized_storages.len() != self.required_storages_count()
    }

    fn allocate(
        strategy: PolynomialsCacheStrategy,
        layout: L,
        domain_size: usize,
        fri_lde_degree: usize,
        max_lde_degree: usize,
    ) -> Self {
        let storages_count = strategy.required_storages_count(fri_lde_degree, max_lde_degree);
        let uninitialized_storages = (0..storages_count)
            .map(|_| GenericStorage::allocate(layout, domain_size))
            .collect();
        Self {
            strategy,
            layout,
            domain_size,
            fri_lde_degree,
            max_lde_degree,
            evaluations: None,
            monomials: None,
            coset_evaluations: BTreeMap::new(),
            uninitialized_storages,
        }
    }

    fn pop_storage(&mut self) -> GenericStorage<Undefined, L> {
        self.uninitialized_storages.pop().unwrap()
    }

    pub fn get_temp_storage(&self) -> GenericStorage<Undefined, L> {
        GenericStorage::allocate(self.layout, self.domain_size)
    }

    pub fn borrow_storage(&mut self) -> GenericStorage<Undefined, L> {
        assert_eq!(
            self.uninitialized_storages.len(),
            self.required_storages_count()
        );
        self.pop_storage()
    }

    pub fn num_polys(&self) -> usize {
        self.layout.num_polys()
    }

    pub fn get_evaluations(&mut self) -> CudaResult<Rc<GenericStorage<LagrangeBasis, L>>> {
        let result = match self.strategy {
            InPlace => {
                if let Some(evaluations) = &self.evaluations {
                    evaluations.clone()
                } else {
                    let monomials = self.get_monomials()?;
                    drop(self.monomials.take());
                    let monomials = Rc::into_inner(monomials).unwrap();
                    let evaluations = Rc::new(monomials.into_evaluations()?);
                    self.evaluations = Some(evaluations.clone());
                    evaluations
                }
            }
            CacheMonomials | CacheMonomialsAndFirstCoset | CacheMonomialsAndFriCosets => {
                Rc::new(self.monomials.as_ref().unwrap().create_evaluations()?)
            }
            _ => self.evaluations.as_ref().unwrap().clone(),
        };
        Ok(result)
    }

    pub fn get_monomials(&mut self) -> CudaResult<Rc<GenericStorage<MonomialBasis, L>>> {
        let result = match self.strategy {
            InPlace => {
                if let Some(monomials) = &self.monomials {
                    monomials.clone()
                } else {
                    let monomials = if let Some(evaluations) = self.evaluations.take() {
                        let evaluations = Rc::into_inner(evaluations).unwrap();
                        evaluations.into_monomials()?
                    } else {
                        let (coset_idx, coset) = self.coset_evaluations.pop_first().unwrap();
                        assert!(self.coset_evaluations.is_empty());
                        let coset = Rc::into_inner(coset).unwrap();
                        coset.into_monomials(coset_idx, self.max_lde_degree)?
                    };
                    let monomials = Rc::new(monomials);
                    self.monomials = Some(monomials.clone());
                    monomials
                }
            }
            CacheEvaluationsAndAllCosets => {
                let (coset_idx, coset) = self.coset_evaluations.first_key_value().unwrap();
                let monomials = coset.create_monomials(*coset_idx, self.max_lde_degree)?;
                Rc::new(monomials)
            }
            _ => self.monomials.as_ref().unwrap().clone(),
        };
        Ok(result)
    }

    pub fn get_coset_evaluations(
        &mut self,
        coset_idx: usize,
    ) -> CudaResult<Rc<GenericStorage<CosetEvaluations, L>>> {
        assert!(coset_idx < self.max_lde_degree);
        let result = match self.strategy {
            InPlace => {
                if let Some(coset) = self.coset_evaluations.get(&coset_idx) {
                    coset.clone()
                } else {
                    let monomials = self.get_monomials()?;
                    drop(self.monomials.take());
                    let monomials = Rc::into_inner(monomials).unwrap();
                    let coset =
                        Rc::new(monomials.into_coset_evaluations(coset_idx, self.max_lde_degree)?);
                    self.coset_evaluations.insert(coset_idx, coset.clone());
                    coset
                }
            }
            CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                if coset_idx == 0 =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                if coset_idx < self.fri_lde_degree =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomials
            | CacheMonomialsAndFirstCoset
            | CacheMonomialsAndFriCosets
            | CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets => Rc::new(
                self.monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations(coset_idx, self.max_lde_degree)?,
            ),
            CacheEvaluationsAndAllCosets => self.coset_evaluations.get(&coset_idx).unwrap().clone(),
        };
        Ok(result)
    }

    pub fn get_coset_evaluations_subset(
        &mut self,
        coset_idx: usize,
        subset: L::PolyType,
    ) -> CudaResult<Rc<GenericStorage<CosetEvaluations, L>>> {
        assert!(coset_idx < self.max_lde_degree);
        let result = match self.strategy {
            InPlace | CacheEvaluationsAndAllCosets => self.get_coset_evaluations(coset_idx)?,
            CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                if coset_idx == 0 =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                if coset_idx < self.fri_lde_degree =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomials
            | CacheMonomialsAndFirstCoset
            | CacheMonomialsAndFriCosets
            | CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets => Rc::new(
                self.monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations_subset(coset_idx, self.max_lde_degree, subset)?,
            ),
        };
        Ok(result)
    }
}

impl<L: GenericPolynomialStorageLayout> ChunkLeavesSource for PolynomialsCache<L> {
    fn get_chunk_leaves(&mut self, chunk_idx: usize) -> CudaResult<Rc<impl AsSingleSlice>> {
        self.get_coset_evaluations(chunk_idx)
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum CommitmentCacheStrategy {
    CacheAllLayers,
    CacheCosetCaps,
}

use CommitmentCacheStrategy::*;

impl CommitmentCacheStrategy {
    pub fn required_storages_count(&self, chunks_count: usize) -> usize {
        match self {
            CacheAllLayers => chunks_count,
            CacheCosetCaps => 0,
        }
    }
}

pub struct CommitmentCache<H: GpuTreeHasher> {
    pub strategy: CommitmentCacheStrategy,
    pub chunk_size: usize,
    pub chunks_count: usize,
    pub cap_size: usize,
    pub chunk_cap_size: usize,
    pub num_elems_per_leaf: usize,
    pub subtrees_by_chunk: BTreeMap<usize, SubTree<H>>,
    pub caps_by_chunk: BTreeMap<usize, Vec<H::Output>>,
    pub tree: Vec<Vec<H::Output>>,
    pub(crate) uninitialized_subtree_nodes: Vec<DVec<H::DigestElementType>>,
}

impl<H: GpuTreeHasher> CommitmentCache<H> {
    fn get_nodes_size_per_chunk(chunk_size: usize, num_elems_per_leaf: usize) -> usize {
        assert!(chunk_size.is_power_of_two());
        assert!(num_elems_per_leaf.is_power_of_two());
        assert!(num_elems_per_leaf <= chunk_size);
        H::CAPACITY << (chunk_size.trailing_zeros() - num_elems_per_leaf.trailing_zeros() + 1)
    }

    pub fn allocate(
        strategy: CommitmentCacheStrategy,
        chunk_size: usize,
        chunks_count: usize,
        cap_size: usize,
        num_elems_per_leaf: usize,
    ) -> Self {
        assert!(chunks_count.is_power_of_two());
        assert!(cap_size.is_power_of_two());
        let chunk_cap_size = coset_cap_size(cap_size, chunks_count);
        let storages_count = strategy.required_storages_count(chunks_count);
        let nodes_size = Self::get_nodes_size_per_chunk(chunk_size, num_elems_per_leaf);
        let uninitialized_subtree_nodes = (0..storages_count).map(|_| dvec!(nodes_size)).collect();
        Self {
            strategy,
            chunk_size,
            chunks_count,
            cap_size,
            chunk_cap_size,
            num_elems_per_leaf,
            subtrees_by_chunk: BTreeMap::new(),
            caps_by_chunk: BTreeMap::new(),
            tree: vec![],
            uninitialized_subtree_nodes,
        }
    }

    fn get_coset_cap_from_tree(&self, coset_index: usize) -> &[H::Output] {
        let start = coset_index * self.chunk_cap_size;
        let end = start + self.chunk_cap_size;
        &self.tree[0][start..end]
    }

    pub(crate) fn initialize_for_chunk(
        &mut self,
        chunk_idx: usize,
        chunk: &impl AsSingleSlice,
    ) -> CudaResult<()> {
        assert!(chunk_idx < self.chunks_count);
        match self.strategy {
            CacheAllLayers => {
                let nodes = self.uninitialized_subtree_nodes.pop().unwrap();
                let (subtree, subtree_cap) =
                    build_subtree::<H>(chunk, self.chunk_cap_size, self.num_elems_per_leaf, nodes)?;
                self.subtrees_by_chunk.insert(chunk_idx, subtree);
                self.caps_by_chunk.insert(chunk_idx, subtree_cap);
            }
            CacheCosetCaps => {
                let nodes_size =
                    Self::get_nodes_size_per_chunk(self.chunk_size, self.num_elems_per_leaf);
                let nodes = dvec!(nodes_size);
                let (_, subtree_cap) =
                    build_subtree::<H>(chunk, self.chunk_cap_size, self.num_elems_per_leaf, nodes)?;
                self.caps_by_chunk.insert(chunk_idx, subtree_cap);
            }
        };
        Ok(())
    }

    pub(crate) fn build_tree(&mut self) -> CudaResult<()> {
        let is_dry_run = is_dry_run()?;
        if self.tree.is_empty() {
            assert_eq!(self.caps_by_chunk.len(), self.chunks_count);
            self.tree.push(
                (0..self.chunks_count)
                    .flat_map(|i| self.caps_by_chunk.get(&i).unwrap().clone())
                    .collect_vec(),
            );
            while self.tree.last().unwrap().len() != self.cap_size {
                self.tree.push(
                    self.tree
                        .last()
                        .unwrap()
                        .chunks(2)
                        .map(|chunk| {
                            if is_dry_run {
                                H::Output::default()
                            } else {
                                H::hash_into_node(&chunk[0], &chunk[1], 0)
                            }
                        })
                        .collect_vec(),
                );
            }
        } else {
            for chunk_index in 0..self.chunks_count {
                let tree_coset_cap = self.get_coset_cap_from_tree(chunk_index);
                if let Some(map_coset_cap) = self.caps_by_chunk.get(&chunk_index) {
                    if !is_dry_run {
                        assert_eq!(tree_coset_cap, map_coset_cap);
                    }
                } else {
                    self.caps_by_chunk
                        .insert(chunk_index, tree_coset_cap.to_vec());
                }
            }
        }
        Ok(())
    }

    pub fn get_tree_cap(&self) -> Vec<H::Output> {
        self.tree.last().unwrap().clone()
    }

    pub fn batch_query_for_chunk<A: GoodAllocator>(
        &self,
        chunk_source: &mut impl ChunkLeavesSource,
        chunk_idx: usize,
        d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<H::DigestElementType, A>,
    ) -> CudaResult<()> {
        let leaf_sources = chunk_source.get_chunk_leaves(chunk_idx)?;
        let subtree = if let Some(subtree) = self.subtrees_by_chunk.get(&chunk_idx) {
            subtree.clone()
        } else {
            let coset_cap_size = coset_cap_size(self.cap_size, self.chunks_count);
            let nodes_size =
                Self::get_nodes_size_per_chunk(self.chunk_size, self.num_elems_per_leaf);
            let nodes = dvec!(nodes_size);
            let (subtree, subtree_cap) = build_subtree::<H>(
                leaf_sources.deref(),
                coset_cap_size,
                self.num_elems_per_leaf,
                nodes,
            )?;
            if !is_dry_run()? {
                assert_eq!(&subtree_cap, self.caps_by_chunk.get(&chunk_idx).unwrap());
            }
            subtree
        };
        batch_query::<H, A>(
            d_indexes,
            leaf_sources.deref(),
            leaf_sources.num_polys_in_base(),
            &subtree,
            subtree.cap_size,
            self.chunk_size,
            self.num_elems_per_leaf,
            h_all_leaf_elems,
            h_all_proofs,
        )?;
        if self.chunks_count > self.cap_size {
            let remainder_count =
                (self.chunks_count.trailing_zeros() - self.cap_size.trailing_zeros()) as usize;
            assert_eq!(remainder_count + 1, self.tree.len());
            let num_queries = d_indexes.len();
            let mut index = chunk_idx;
            for layer in self.tree.iter().take(remainder_count) {
                let elements = H::DigestElements::from(layer[index ^ 1]);
                index >>= 1;
                elements
                    .iter()
                    .for_each(|e| h_all_proofs.extend(std::iter::repeat(e).take(num_queries)));
            }
        }
        Ok(())
    }
}

pub struct StorageCache<L: GenericPolynomialStorageLayout, H: GpuTreeHasher, T = ()> {
    pub polynomials_cache: PolynomialsCache<L>,
    pub commitment_cache: CommitmentCache<H>,
    pub domain_size: usize,
    pub fri_lde_degree: usize,
    pub max_lde_degree: usize,
    pub aux: T,
}

impl<L: GenericPolynomialStorageLayout, H: GpuTreeHasher, T> StorageCache<L, H, T> {
    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        polynomials_strategy: PolynomialsCacheStrategy,
        commitments_strategy: CommitmentCacheStrategy,
        layout: L,
        domain_size: usize,
        fri_lde_degree: usize,
        max_lde_degree: usize,
        cap_size: usize,
        num_elems_per_leaf: usize,
        aux: T,
    ) -> Self {
        let polynomials_cache = PolynomialsCache::allocate(
            polynomials_strategy,
            layout,
            domain_size,
            fri_lde_degree,
            max_lde_degree,
        );
        let commitments_cache = CommitmentCache::allocate(
            commitments_strategy,
            domain_size,
            fri_lde_degree,
            cap_size,
            num_elems_per_leaf,
        );
        Self {
            polynomials_cache,
            commitment_cache: commitments_cache,
            domain_size,
            fri_lde_degree,
            max_lde_degree,
            aux,
        }
    }

    pub fn initialize_from_evaluations(
        &mut self,
        evaluations: Rc<GenericStorage<LagrangeBasis, L>>,
    ) -> CudaResult<()> {
        let can_own_evaluations = Rc::strong_count(&evaluations) == 1;
        let polynomials_strategy = self.polynomials_cache.strategy;
        let cache_evaluations = polynomials_strategy.cache_evaluations();
        let is_storage_borrowed = self.polynomials_cache.is_storage_borrowed();
        let must_own_evaluations = !cache_evaluations && is_storage_borrowed;
        assert!(can_own_evaluations || !must_own_evaluations);
        let monomials = if cache_evaluations {
            assert!(is_storage_borrowed);
            let monomials = evaluations.fill_monomials(self.polynomials_cache.pop_storage())?;
            self.polynomials_cache.evaluations = Some(evaluations);
            monomials
        } else if is_storage_borrowed {
            Rc::into_inner(evaluations).unwrap().into_monomials()?
        } else {
            evaluations.fill_monomials(self.polynomials_cache.pop_storage())?
        };
        self.initialize_from_monomials(Rc::new(monomials))
    }

    pub fn initialize_from_monomials(
        &mut self,
        monomials: Rc<GenericStorage<MonomialBasis, L>>,
    ) -> CudaResult<()> {
        let can_owns_monomials = Rc::strong_count(&monomials) == 1;
        let polynomials_strategy = self.polynomials_cache.strategy;
        let cache_monomials = polynomials_strategy.cache_monomials();
        let is_storage_borrowed = self.polynomials_cache.is_storage_borrowed();
        let must_own_monomials = !cache_monomials && is_storage_borrowed;
        assert!(can_owns_monomials || !must_own_monomials);
        let mut monomials = Some(monomials);
        if polynomials_strategy.cache_evaluations() && self.polynomials_cache.evaluations.is_none()
        {
            let evaluations = monomials
                .as_ref()
                .unwrap()
                .fill_evaluations(self.polynomials_cache.pop_storage())?;
            self.polynomials_cache.evaluations = Some(Rc::new(evaluations));
        };
        let polynomials_cosets_count = match polynomials_strategy {
            InPlace => 0,
            CacheMonomials => 0,
            CacheMonomialsAndFirstCoset => 1,
            CacheMonomialsAndFriCosets => self.fri_lde_degree,
            CacheEvaluationsAndMonomials => 0,
            CacheEvaluationsMonomialsAndFirstCoset => 1,
            CacheEvaluationsMonomialsAndFriCosets => self.fri_lde_degree,
            CacheEvaluationsAndAllCosets => self.max_lde_degree,
        };
        let commitment_cosets_count = match self.commitment_cache.strategy {
            CacheCosetCaps if !self.commitment_cache.tree.is_empty() => 0,
            _ => self.fri_lde_degree,
        };
        let cosets_count = usize::max(polynomials_cosets_count, commitment_cosets_count);
        for coset_idx in 0..cosets_count {
            let coset = match polynomials_strategy {
                InPlace => Rc::into_inner(monomials.take().unwrap())
                    .unwrap()
                    .into_coset_evaluations(coset_idx, self.max_lde_degree)?,
                CacheMonomials | CacheEvaluationsAndMonomials => monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations(coset_idx, self.max_lde_degree)?,
                CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                    if coset_idx == 0 =>
                {
                    monomials.as_ref().unwrap().fill_coset_evaluations(
                        coset_idx,
                        self.max_lde_degree,
                        self.polynomials_cache.pop_storage(),
                    )?
                }
                CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                    if coset_idx < self.fri_lde_degree =>
                {
                    monomials.as_ref().unwrap().fill_coset_evaluations(
                        coset_idx,
                        self.max_lde_degree,
                        self.polynomials_cache.pop_storage(),
                    )?
                }
                CacheMonomialsAndFirstCoset
                | CacheMonomialsAndFriCosets
                | CacheEvaluationsMonomialsAndFirstCoset
                | CacheEvaluationsMonomialsAndFriCosets => monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations(coset_idx, self.max_lde_degree)?,
                CacheEvaluationsAndAllCosets => {
                    if coset_idx + 1 == self.max_lde_degree {
                        Rc::into_inner(monomials.take().unwrap())
                            .unwrap()
                            .into_coset_evaluations(coset_idx, self.max_lde_degree)?
                    } else {
                        monomials.as_ref().unwrap().fill_coset_evaluations(
                            coset_idx,
                            self.max_lde_degree,
                            self.polynomials_cache.pop_storage(),
                        )?
                    }
                }
            };
            if coset_idx < commitment_cosets_count {
                self.commitment_cache
                    .initialize_for_chunk(coset_idx, &coset)?;
            };
            match polynomials_strategy {
                InPlace => {
                    monomials = Some(Rc::new(
                        coset.into_monomials(coset_idx, self.max_lde_degree)?,
                    ));
                }
                CacheMonomials | CacheEvaluationsAndMonomials => {}
                CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                    if coset_idx == 0 =>
                {
                    self.polynomials_cache
                        .coset_evaluations
                        .insert(coset_idx, Rc::new(coset));
                }
                CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                    if coset_idx < self.fri_lde_degree =>
                {
                    self.polynomials_cache
                        .coset_evaluations
                        .insert(coset_idx, Rc::new(coset));
                }
                CacheMonomialsAndFirstCoset
                | CacheMonomialsAndFriCosets
                | CacheEvaluationsMonomialsAndFirstCoset
                | CacheEvaluationsMonomialsAndFriCosets => {}
                CacheEvaluationsAndAllCosets => {
                    self.polynomials_cache
                        .coset_evaluations
                        .insert(coset_idx, Rc::new(coset));
                }
            };
        }
        self.commitment_cache.build_tree()?;
        match polynomials_strategy {
            CacheEvaluationsAndAllCosets => {}
            _ => {
                self.polynomials_cache.monomials = Some(monomials.take().unwrap());
            }
        };
        Ok(())
    }

    pub fn num_polys(&self) -> usize {
        self.polynomials_cache.num_polys()
    }

    pub fn get_evaluations(&mut self) -> CudaResult<Rc<GenericStorage<LagrangeBasis, L>>> {
        self.polynomials_cache.get_evaluations()
    }

    pub fn get_monomials(&mut self) -> CudaResult<Rc<GenericStorage<MonomialBasis, L>>> {
        self.polynomials_cache.get_monomials()
    }

    pub fn get_coset_evaluations(
        &mut self,
        coset_idx: usize,
    ) -> CudaResult<Rc<GenericStorage<CosetEvaluations, L>>> {
        self.polynomials_cache.get_coset_evaluations(coset_idx)
    }

    pub fn get_coset_evaluations_subset(
        &mut self,
        coset_idx: usize,
        subset: L::PolyType,
    ) -> CudaResult<Rc<GenericStorage<CosetEvaluations, L>>> {
        self.polynomials_cache
            .get_coset_evaluations_subset(coset_idx, subset)
    }

    pub fn get_tree_cap(&mut self) -> Vec<H::Output> {
        self.commitment_cache.get_tree_cap()
    }

    pub fn batch_query_for_coset<A: GoodAllocator>(
        &mut self,
        coset_idx: usize,
        d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<H::DigestElementType, A>,
    ) -> CudaResult<()> {
        self.commitment_cache.batch_query_for_chunk(
            &mut self.polynomials_cache,
            coset_idx,
            d_indexes,
            h_all_leaf_elems,
            h_all_proofs,
        )
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub struct CacheStrategy {
    pub setup_polynomials: PolynomialsCacheStrategy,
    pub trace_polynomials: PolynomialsCacheStrategy,
    pub other_polynomials: PolynomialsCacheStrategy,
    pub commitment: CommitmentCacheStrategy,
}

impl CacheStrategy {
    pub(crate) fn get<
        TR: Transcript<F, CompatibleCap: Hash>,
        H: GpuTreeHasher<Output = TR::CompatibleCap>,
        POW: GPUPoWRunner,
        A: GoodAllocator,
    >(
        config: &GpuProofConfig,
        external_witness_data: &WitnessVec<F>,
        proof_config: ProofConfig,
        setup: &GpuSetup<H>,
        vk: &VerificationKey<F, H>,
        transcript_params: TR::TransciptParameters,
        worker: &Worker,
    ) -> CudaResult<Self> {
        let cap = &vk.setup_merkle_tree_cap;
        let mut hasher = DefaultHasher::new();
        Hash::hash_slice(cap, &mut hasher);
        let cap_hash = hasher.finish();
        if let Some(strategy) = _strategy_cache_get().get(&cap_hash) {
            println!("reusing cache strategy");
            Ok(*strategy)
        } else {
            let strategies =
                Self::get_strategy_candidates(config, &proof_config, setup, &vk.fixed_parameters);
            for (_, strategy) in strategies.iter().copied() {
                _setup_cache_reset();
                dry_run_start();
                let result = gpu_prove_from_external_witness_data_with_cache_strategy_inner::<
                    TR,
                    H,
                    POW,
                    A,
                >(
                    config,
                    external_witness_data,
                    proof_config.clone(),
                    setup,
                    vk,
                    transcript_params.clone(),
                    worker,
                    strategy,
                );
                _setup_cache_reset();
                let result = result.and(dry_run_stop());
                match result {
                    Ok(_) => {
                        println!("determined cache strategy: {:?}", strategy);
                        _strategy_cache_get().insert(cap_hash, strategy);
                        return Ok(strategy);
                    }
                    Err(ErrorMemoryAllocation) => {
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(ErrorMemoryAllocation)
        }
    }

    pub(crate) fn get_strategy_candidates<H: GpuTreeHasher>(
        config: &GpuProofConfig,
        proof_config: &ProofConfig,
        setup: &GpuSetup<H>,
        geometry: &VerificationKeyCircuitGeometry,
    ) -> Vec<((usize, usize), CacheStrategy)> {
        let fri_lde_degree = proof_config.fri_lde_factor;
        let quotient_degree = compute_quotient_degree(config, &setup.selectors_placement);
        let max_lde_degree = usize::max(quotient_degree, fri_lde_degree);
        let setup_layout = setup.layout;
        let domain_size = geometry.domain_size as usize;
        let lookup_parameters = geometry.lookup_parameters;
        let total_tables_len = geometry.total_tables_len as usize;
        let num_multiplicity_cols =
            lookup_parameters.num_multipicities_polys(total_tables_len, domain_size);
        let trace_layout = TraceLayout {
            num_variable_cols: setup.variables_hint.len(),
            num_witness_cols: setup.witnesses_hint.len(),
            num_multiplicity_cols,
        };
        let arguments_layout = ArgumentsLayout::from_trace_layout_and_lookup_params(
            trace_layout,
            quotient_degree,
            geometry.lookup_parameters,
        );
        let quotient_layout = QuotientLayout::new(quotient_degree);
        let setup_num_polys = setup_layout.num_polys();
        let trace_num_polys = trace_layout.num_polys();
        let other_num_polys = arguments_layout.num_polys() + quotient_layout.num_polys();
        let commitments_cache_strategies = [CacheAllLayers, CacheCosetCaps];
        let setup_polynomials_strategies = [
            InPlace,
            CacheMonomials,
            CacheMonomialsAndFirstCoset,
            CacheMonomialsAndFriCosets,
            CacheEvaluationsAndMonomials,
            CacheEvaluationsMonomialsAndFirstCoset,
            CacheEvaluationsMonomialsAndFriCosets,
            CacheEvaluationsAndAllCosets,
        ];
        let other_polynomials_strategies = [
            InPlace,
            CacheMonomials,
            CacheMonomialsAndFirstCoset,
            CacheMonomialsAndFriCosets,
        ];
        let mut strategies = Vec::new();
        for commitment_strategy in commitments_cache_strategies.iter().copied() {
            for setup_strategy in setup_polynomials_strategies.iter().copied() {
                for other_strategies in (0..2)
                    .map(|_| other_polynomials_strategies.iter().cloned())
                    .multi_cartesian_product()
                {
                    let strategy = CacheStrategy {
                        setup_polynomials: setup_strategy,
                        trace_polynomials: other_strategies[0],
                        other_polynomials: other_strategies[1],
                        commitment: commitment_strategy,
                    };
                    let setup_cost =
                        strategy.get_setup_cost(fri_lde_degree, max_lde_degree) * setup_num_polys;
                    let proof_cost_setup = strategy
                        .get_proof_cost_setup(fri_lde_degree, max_lde_degree)
                        * setup_num_polys;
                    let proof_cost_trace = strategy
                        .get_proof_cost_trace(fri_lde_degree, max_lde_degree)
                        * trace_num_polys;
                    let proof_cost_other = strategy
                        .get_proof_cost_other(fri_lde_degree, max_lde_degree)
                        * other_num_polys;
                    let proof_cost = proof_cost_setup + proof_cost_trace + proof_cost_other;
                    let costs = match commitment_strategy {
                        CacheAllLayers => (proof_cost, setup_cost),
                        CacheCosetCaps => (proof_cost << 16, setup_cost << 16),
                    };
                    strategies.push((costs, strategy));
                }
            }
        }
        strategies.sort_by_key(|x| x.0);
        strategies
    }

    fn get_setup_cost(&self, fri_lde_degree: usize, max_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = max_lde_degree;
        match self.setup_polynomials {
            InPlace => 1 + 2 * f,
            CacheMonomials
            | CacheMonomialsAndFirstCoset
            | CacheMonomialsAndFriCosets
            | CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets => 1 + f,
            CacheEvaluationsAndAllCosets => 1 + u,
        }
    }

    fn get_proof_cost_setup(&self, fri_lde_degree: usize, max_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = max_lde_degree;
        match self.setup_polynomials {
            InPlace => 2 + 2 * u + 2 + 2 * (f - 1) + 2 * f,
            CacheMonomials => 1 + u + 1 + f + f,
            CacheMonomialsAndFirstCoset => 1 + u - 1 + f - 1 + f - 1,
            CacheMonomialsAndFriCosets => 1 + u - f,
            CacheEvaluationsAndMonomials => u + 1 + f + f,
            CacheEvaluationsMonomialsAndFirstCoset => u - 1 + f - 1 + f - 1,
            CacheEvaluationsMonomialsAndFriCosets => u - f,
            CacheEvaluationsAndAllCosets => 0,
        }
    }

    fn get_proof_cost_trace(&self, fri_lde_degree: usize, max_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = max_lde_degree;
        match self.trace_polynomials {
            InPlace => 1 + 2 * f + 1 + 2 * u + 2 + 2 * (f - 1) + 2 * f,
            CacheMonomials => 1 + f + u + 1 + f + f,
            CacheMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndMonomials => 1 + f + u + 1 + f + f,
            CacheEvaluationsMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheEvaluationsMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndAllCosets => 1 + u,
        }
    }

    fn get_proof_cost_other(&self, fri_lde_degree: usize, max_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = max_lde_degree;
        match self.other_polynomials {
            InPlace => 1 + 2 * f + 2 * u - 1 + 1 + 1 + 2 * (f - 1) + 2 * f,
            CacheMonomials => 1 + f + u + 1 + f + f,
            CacheMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndMonomials => 1 + f + u + 1 + f + f,
            CacheEvaluationsMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheEvaluationsMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndAllCosets => 1 + u,
        }
    }
}
