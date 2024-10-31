#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(array_chunks)]
#![feature(associated_type_defaults)]
#![feature(iter_array_chunks)]
#![feature(get_mut_unchecked)]
#![feature(generic_const_exprs)]
pub use boojum;
mod context;
#[cfg(test)]
mod test;
use boojum::field::goldilocks::GoldilocksExt2 as EXT;
use boojum::field::goldilocks::GoldilocksField as F;
use boojum::field::{ExtensionField, Field, PrimeField};
use context::*;
mod oracle;
use oracle::*;
mod fri;
use fri::*;
mod pow;

mod utils;
use utils::*;
mod primitives;
use primitives::dry_run::*;
mod static_allocator;
use static_allocator::*;
mod dvec;
use dvec::*;
mod constraint_evaluation;
mod copy_permutation;
pub mod cs;
mod data_structures;
mod lookup;
mod poly;
mod traits;
use constraint_evaluation::*;
use copy_permutation::*;
use data_structures::*;
use lookup::*;
use poly::*;
pub mod gpu_proof_config;
mod prover;
mod quotient;
#[cfg(feature = "zksync")]
pub mod synthesis_utils;
#[cfg(feature = "zksync")]
pub use circuit_definitions;

use quotient::*;
type EF = ExtensionField<F, 2, EXT>;

use primitives::*;
use std::alloc::Global;
use std::fmt::Debug;
use std::slice::Chunks;

use primitives::arith;
use primitives::cs_helpers;
use primitives::helpers;

use primitives::ntt;
use primitives::tree;

use boojum::cs::traits::GoodAllocator;
pub use context::ProverContext;
pub use context::ProverContextConfig;
pub use data_structures::CacheStrategy;
pub use data_structures::CommitmentCacheStrategy;
pub use data_structures::PolynomialsCacheStrategy;
pub use prover::gpu_prove_from_external_witness_data;
pub use prover::gpu_prove_from_external_witness_data_with_cache_strategy;
