use super::*;

use circuit_definitions::circuit_definitions::aux_layer::compression::{
    CompressionLayerCircuit, ProofCompressionFunction,
};
use shivini::{
    boojum::cs::implementations::{
        fast_serialization::MemcopySerializable, transcript::Transcript,
    },
    GPUPoWRunner, GpuTreeHasher,
};

pub trait ProofSystemDefinition: Sized {
    type FieldElement;
    type ExternalWitnessData;
    type Precomputation: MemcopySerializable + Send + Sync + 'static;
    type Proof: serde::Serialize + serde::de::DeserializeOwned;
    type VK: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + Clone + 'static;
    type FinalizationHint: serde::Serialize
        + serde::de::DeserializeOwned
        + Clone
        + Send
        + Sync
        + 'static;
    type Allocator: std::alloc::Allocator + Default;
    type ProvingAssembly: Sized + Send + Sync + 'static;
    type Transcript;
    fn take_witnesses(proving_assembly: &mut Self::ProvingAssembly) -> Self::ExternalWitnessData;
    fn verify(_: &Self::Proof, _: &Self::VK) -> bool;
}

pub trait CompressionProofSystem:
    ProofCompressionFunction<
        ThisLayerHasher: GpuTreeHasher,
        ThisLayerTranscript: Transcript<GoldilocksField, TransciptParameters = ()>,
        ThisLayerPoW: GPUPoWRunner,
    > + ProofSystemDefinition
{
    type ContextConfig: Send + Sync + 'static;
    type Context: Send + Sync + 'static;
    type AuxConfig;
    fn get_context_config() -> Self::ContextConfig;
    fn get_context_config_from_hint(_: &Self::FinalizationHint) -> Self::ContextConfig;
    fn init_context(config: Self::ContextConfig) -> anyhow::Result<Self::Context>;
    fn aux_config_from_assembly(proving_assembly: &Self::ProvingAssembly) -> Self::AuxConfig;

    fn synthesize_for_proving(
        circuit: CompressionLayerCircuit<Self>,
        finalization_hint: Self::FinalizationHint,
    ) -> Self::ProvingAssembly;

    fn prove(
        _: &Self::Context,
        _: Self::ProvingAssembly,
        _: Self::AuxConfig,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
        _: &Self::VK,
    ) -> anyhow::Result<Self::Proof>;

    fn prove_from_witnesses(
        _: &Self::Context,
        _: Self::ExternalWitnessData,
        _: Self::AuxConfig,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
        _: &Self::VK,
    ) -> anyhow::Result<Self::Proof>;
}

pub trait CompressionProofSystemExt: CompressionProofSystem {
    type SetupAssembly;
    fn generate_precomputation_and_vk(
        _: Self::Context,
        _: Self::SetupAssembly,
        _: &Self::FinalizationHint,
    ) -> anyhow::Result<(Self::Precomputation, Self::VK)>;
    fn synthesize_for_setup(
        circuit: CompressionLayerCircuit<Self>,
    ) -> (Self::FinalizationHint, Self::SetupAssembly);
}

pub trait SnarkWrapperProofSystem: ProofSystemDefinition {
    type Circuit;
    type Context: Send + Sync + 'static;
    type CRS: Send + Sync + 'static;
    fn pre_init();
    fn init_context(crs: &Self::CRS) -> anyhow::Result<Self::Context>;
    fn load_compact_raw_crs<R: std::io::Read>(src: R) -> anyhow::Result<Self::CRS>;
    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly;
    fn prove(
        _: &Self::Context,
        _: Self::ProvingAssembly,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
    ) -> anyhow::Result<Self::Proof>;

    fn prove_from_witnesses(
        _: &Self::Context,
        _: Self::ExternalWitnessData,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
    ) -> anyhow::Result<Self::Proof>;
}

pub trait SnarkWrapperProofSystemExt: SnarkWrapperProofSystem {
    type SetupAssembly;
    fn synthesize_for_setup(circuit: Self::Circuit) -> Self::SetupAssembly;
    fn generate_precomputation_and_vk(
        _: &Self::Context,
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> anyhow::Result<(Self::Precomputation, Self::VK)>;
}

pub(crate) struct MarkerProofSystem;

impl ProofSystemDefinition for MarkerProofSystem {
    type FieldElement = ();
    type Precomputation = MarkerPrecomputation;
    type Proof = ();
    type VK = ();
    type ExternalWitnessData = ();
    type FinalizationHint = ();
    type Allocator = std::alloc::Global;
    type ProvingAssembly = ();
    type Transcript = ();
    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        unreachable!()
    }

    fn take_witnesses(_proving_assembly: &mut Self::ProvingAssembly) -> Self::ExternalWitnessData {
        unreachable!()
    }
}
