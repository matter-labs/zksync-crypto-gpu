#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(allocator_api)]

mod blob_storage;
use blob_storage::*;

mod chain;
use chain::*;

mod compression;
use compression::*;

mod context;
use context::*;

mod proof_system;
use proof_system::*;

mod snark_wrapper;
use snark_wrapper::*;

mod serialization;
use serialization::*;

mod utils;
use utils::*;   

#[cfg(test)]
mod test;

cfg_if::cfg_if! {
    if #[cfg(feature = "gpu")] {
        //

    } else {
        //
    }
}
use bellman::worker::Worker;
use boojum::pairing::bn256::{Bn256, Fr};
use circuit_definitions::circuit_definitions::{
    aux_layer::{ZkSyncSnarkWrapperCircuit, ZkSyncSnarkWrapperCircuitNoLookupCustomGate},
    recursion_layer::RecursiveProofsTreeHasher,
};
use fflonk::{
    bellman::plonk::better_better_cs::{
        cs::VerificationKey as PlonkVerificationKey, proof::Proof as PlonkProof,
        setup::Setup as PlonkSetup,
    },
    fflonk::{FflonkProof, FflonkSetup, FflonkVerificationKey},
    fflonk_cpu::franklin_crypto,
    FflonkSnarkVerifierCircuitDeviceSetup,
};
use franklin_crypto::bellman;
use gpu_prover::AsyncSetup;
use shivini::boojum::{self, field::goldilocks::GoldilocksExt2};

pub type FflonkSnarkVerifierCircuit = ZkSyncSnarkWrapperCircuitNoLookupCustomGate;
pub type FflonkSnarkVerifierCircuitVK = FflonkVerificationKey<Bn256, FflonkSnarkVerifierCircuit>;
pub type FflonkSnarkVerifierCircuitProof = FflonkProof<Bn256, FflonkSnarkVerifierCircuit>;

pub type PlonkSnarkVerifierCircuit = ZkSyncSnarkWrapperCircuit;
pub type PlonkSnarkVerifierCircuitVK = PlonkVerificationKey<Bn256, PlonkSnarkVerifierCircuit>;
pub type PlonkSnarkVerifierCircuitProof = PlonkProof<Bn256, PlonkSnarkVerifierCircuit>;
pub type PlonkSnarkVerifierCircuitDeviceSetup = AsyncSetup;

pub use fflonk::{GlobalHost, HostAllocator};
use std::alloc::Global;

use shivini::boojum::field::goldilocks::GoldilocksField;

pub type SchedulerProof = boojum::cs::implementations::proof::Proof<
    GoldilocksField,
    RecursiveProofsTreeHasher,
    GoldilocksExt2,
>;

use circuit_definitions::circuit_definitions::aux_layer::{
    compression::{CompressionLayerCircuit, ProofCompressionFunction},
    compression_modes::{
        CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
        CompressionMode4, CompressionMode5ForWrapper, CompressionTranscriptForWrapper,
        CompressionTreeHasherForWrapper,
    },
};
