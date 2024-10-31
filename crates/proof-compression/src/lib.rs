#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(allocator_api)]

mod common;
use common::*;

cfg_if::cfg_if! {
    if #[cfg(feature = "gpu")] {
        mod gpu;
        pub use gpu::*;

    } else {
        mod cpu;
        pub use cpu::*;
    }
}
use boojum::pairing::bn256::{Bn256, Fr};
use circuit_definitions::circuit_definitions::aux_layer::ZkSyncSnarkWrapperCircuitNoLookupCustomGate;
use fflonk::{
    fflonk::{FflonkProof, FflonkSetup, FflonkVerificationKey},
    fflonk_cpu::franklin_crypto,
};
use franklin_crypto::bellman;
use shivini::boojum;

use bellman::worker::Worker;

pub type FflonkSnarkVerifierCircuit = ZkSyncSnarkWrapperCircuitNoLookupCustomGate;
pub type FflonkSnarkVerifierCircuitVK = FflonkVerificationKey<Bn256, FflonkSnarkVerifierCircuit>;
pub type FflonkSnarkVerifierCircuitProof = FflonkProof<Bn256, FflonkSnarkVerifierCircuit>;
pub type FflonkSnarkVerifierCircuitSetup = FflonkSetup<Bn256, FflonkSnarkVerifierCircuit>;
