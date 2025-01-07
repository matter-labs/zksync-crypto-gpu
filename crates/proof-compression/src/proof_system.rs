use super::*;
use bellman::pairing::compact_bn256::G1Affine as CompactG1Affine;
use bellman::plonk::better_better_cs::cs::SynthesisModeGenerateSetup;
use bellman::plonk::better_better_cs::{
    cs::{Assembly, PlonkCsWidth3Params, PlonkCsWidth4WithNextStepParams, SynthesisModeProve},
    gates::{
        naive_main_gate::NaiveMainGate,
        selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
    },
};
use boojum::config::SetupCSConfig;
use boojum::cs::implementations::prover::ProofConfig;
use boojum::cs::oracle::TreeHasher;
use boojum::{
    config::ProvingCSConfig,
    cs::implementations::{
        fast_serialization::MemcopySerializable, proof::Proof, reference_cs::CSReferenceAssembly,
        setup::FinalizationHintsForProver, verifier::VerificationKey,
    },
    field::goldilocks::GoldilocksExt2,
};
use circuit_definitions::circuit_definitions::aux_layer::compression::ProofCompressionFunction;
use fflonk::DeviceContextWithSingleDevice;
use gpu_prover::{DeviceMemoryManager, ManagerConfigs};
use shivini::ProverContextConfig;
use shivini::{cs::GpuSetup, GpuTreeHasher, ProverContext};

// We can't have circuit interface as part of the proof system
// definition due to type dependency between step inputs
// circuit(input_vk, input_proof), however we can synthesize it
// through aux interface
pub trait ProofSystemDefinition: Sized {
    type FieldElement;
    type Precomputation: MemcopySerializable;
    type Proof: serde::Serialize + serde::de::DeserializeOwned;
    type VK: serde::Serialize + serde::de::DeserializeOwned;
    type FinalizationHint: serde::Serialize + serde::de::DeserializeOwned;
    type Allocator: std::alloc::Allocator;
    type ProvingAssembly: Sized + Send + Sync + 'static;
    type ContextConfig;
    type Context;
    type ProofConfig: 'static;
    fn get_context_config() -> Self::ContextConfig {
        todo!()
    }
    fn init_context(config: Self::ContextConfig) -> AsyncHandler<Self::Context>;
    fn build_proving_assembly() -> Self::ProvingAssembly;

    fn prove(
        _: Self::ProvingAssembly,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof;

    fn take_witnesses(
        proving_assembly: &mut Self::ProvingAssembly,
    ) -> Vec<Self::FieldElement, Self::Allocator>;

    fn prove_from_witnesses(
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof;

    fn verify(_: &Self::Proof, _: &Self::VK) -> bool;
}

pub trait CompressionProofSystem: ProofSystemDefinition {
    fn proof_config_for_compression_step<CF>() -> Self::ProofConfig
    where
        CF: ProofCompressionFunction;

    fn synthesize_for_proving<P>(
        input_vk: P::VK,
        input_proof: P::Proof,
        compression_mode: u8,
    ) -> Self::ProvingAssembly
    where
        P: ProofSystemDefinition,
    {
        todo!()
    }
}

pub trait CompressionProofSystemExt: ProofSystemExt {
    fn synthesize_for_setup<P>(input_vk: P::VK, mode: u8) -> Self::SetupAssembly
    where
        P: ProofSystemDefinition;
}

pub trait SnarkWrapperProofSystem: ProofSystemDefinition {
    fn proof_config() -> Self::ProofConfig;
    fn synthesize_for_proving<P>(input_vk: P::VK, input_proof: P::Proof) -> Self::ProvingAssembly
    where
        P: ProofSystemDefinition;
}

pub trait SnarkWrapperProofSystemExt: ProofSystemExt {
    fn synthesize_for_setup<P>(input_vk: P::VK) -> Self::SetupAssembly
    where
        P: ProofSystemDefinition;
}

pub trait ProofSystemExt: ProofSystemDefinition {
    type SetupAssembly: Sized + Send + Sync + 'static;
    fn generate_precomputation_and_vk(
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> (AsyncHandler<Self::Precomputation>, Self::VK);
}
pub struct BoojumProofSystem<H: TreeHasher<GoldilocksField>> {
    _hasher: std::marker::PhantomData<H>,
}

type BoojumAssembly<CSConfig, A> =
    CSReferenceAssembly<GoldilocksField, GoldilocksField, CSConfig, A>;

pub struct TreeHasherCompatibleGpuSetup<H: TreeHasher<GoldilocksField>>(
    std::marker::PhantomData<H>,
);

impl<H> ProofSystemDefinition for BoojumProofSystem<H>
where
    H: TreeHasher<GoldilocksField, Output: serde::Serialize + serde::de::DeserializeOwned>,
{
    type FieldElement = GoldilocksField;
    type Precomputation = TreeHasherCompatibleGpuSetup<H>;
    type Proof = Proof<Self::FieldElement, H, GoldilocksExt2>;
    type VK = VerificationKey<Self::FieldElement, H>;
    type FinalizationHint = FinalizationHintsForProver;
    type Allocator = std::alloc::Global;
    type ProvingAssembly = BoojumAssembly<ProvingCSConfig, Self::Allocator>;
    type ContextConfig = usize; // domain_size
    type Context = ProverContext;
    type ProofConfig = ProofConfig;
    // type Circuit = ZkSyncCompressionLayerCircuit;
    fn init_context(domain_size: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        let config =
            ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
        let context = Self::Context::create_with_config(config).expect("gpu prover context");
        todo!()
    }

    fn build_proving_assembly() -> Self::ProvingAssembly {
        todo!()
    }

    fn prove(
        _: Self::ProvingAssembly,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }
    fn prove_from_witnesses(
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }

    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        todo!()
    }

    fn take_witnesses(
        proving_assembly: &mut Self::ProvingAssembly,
    ) -> Vec<Self::FieldElement, Self::Allocator> {
        todo!()
    }
}

impl<H: TreeHasher<GoldilocksField>> CompressionProofSystem for BoojumProofSystem<H>
where
    H: TreeHasher<GoldilocksField, Output: serde::Serialize + serde::de::DeserializeOwned>,
{
    fn proof_config_for_compression_step<CF>() -> Self::ProofConfig
    where
        CF: ProofCompressionFunction,
    {
        CF::proof_config_for_compression_step()
    }

    fn synthesize_for_proving<P>(
        input_vk: P::VK,
        input_proof: P::Proof,
        compression_mode: u8,
    ) -> Self::ProvingAssembly
    where
        P: ProofSystemDefinition,
    {
        todo!()
    }
}

impl<H> ProofSystemExt for BoojumProofSystem<H>
where
    H: TreeHasher<GoldilocksField, Output: serde::Serialize + serde::de::DeserializeOwned>,
{
    type SetupAssembly = BoojumAssembly<SetupCSConfig, Self::Allocator>;
    fn generate_precomputation_and_vk(
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> (AsyncHandler<Self::Precomputation>, Self::VK) {
        todo!()
    }
}

pub struct PlonkProofSystem;

type PlonkAssembly<CSConfig, A> = Assembly<
    Bn256,
    PlonkCsWidth4WithNextStepParams,
    SelectorOptimizedWidth4MainGateWithDNext,
    CSConfig,
    A,
>;

pub struct PlonkProverDeviceMemoryManager;

impl ManagerConfigs for PlonkProverDeviceMemoryManager {
    const NUM_GPUS_LOG: usize = 0;
    const FULL_SLOT_SIZE_LOG: usize = 24;
    const NUM_SLOTS: usize = 29;
    const NUM_HOST_SLOTS: usize = 2;
}

impl ProofSystemDefinition for PlonkProofSystem {
    type FieldElement = Fr;
    type Precomputation = PlonkSnarkVerifierCircuitDeviceSetupWrapper;
    type Proof = PlonkSnarkVerifierCircuitProof;
    type VK = PlonkSnarkVerifierCircuitVK;
    type FinalizationHint = usize;
    type Allocator = GlobalHost;
    type ProvingAssembly = PlonkAssembly<SynthesisModeProve, Self::Allocator>;
    type ContextConfig = (Vec<usize>, Vec<CompactG1Affine>);
    type Context = DeviceMemoryManager<Fr, PlonkProverDeviceMemoryManager>;
    type ProofConfig = ();
    fn init_context(config: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        let (device_ids, compact_crs) = config;
        let ctx = Self::Context::init(&device_ids, &compact_crs[..]);
        todo!()
    }

    fn build_proving_assembly() -> Self::ProvingAssembly {
        todo!()
    }

    fn prove(
        _: Self::ProvingAssembly,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }

    fn take_witnesses(
        proving_assembly: &mut Self::ProvingAssembly,
    ) -> Vec<Self::FieldElement, Self::Allocator> {
        todo!()
    }

    fn prove_from_witnesses(
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }

    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        todo!()
    }
}

impl SnarkWrapperProofSystem for PlonkProofSystem {
    fn synthesize_for_proving<P>(input_vk: P::VK, input_proof: P::Proof) -> Self::ProvingAssembly
    where
        P: ProofSystemDefinition,
    {
        todo!()
    }

    fn proof_config() -> Self::ProofConfig {
        todo!()
    }
}

impl ProofSystemExt for PlonkProofSystem {
    type SetupAssembly = PlonkAssembly<SynthesisModeGenerateSetup, Self::Allocator>;

    fn generate_precomputation_and_vk(
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> (AsyncHandler<Self::Precomputation>, Self::VK) {
        todo!()
    }
}

pub struct FflonkProofSystem;

type FflonkAssembly<CSConfig, A> = Assembly<Bn256, PlonkCsWidth3Params, NaiveMainGate, CSConfig, A>;

impl ProofSystemDefinition for FflonkProofSystem {
    type FieldElement = Fr;
    type Precomputation = FflonkSnarkVerifierCircuitDeviceSetupWrapper;
    type Proof = FflonkSnarkVerifierCircuitProof;
    type VK = FflonkSnarkVerifierCircuitVK;
    type FinalizationHint = usize;
    type Allocator = GlobalHost;
    type ProvingAssembly = FflonkAssembly<SynthesisModeProve, Self::Allocator>;
    type ContextConfig = usize; // domain_size
    type Context = DeviceContextWithSingleDevice;
    type ProofConfig = ();
    // type Circuit = ();
    fn get_context_config() -> Self::ContextConfig {
        todo!()
    }
    fn init_context(log_domain_size: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        let domain_size = 1 << log_domain_size;
        let context = Self::Context::init(domain_size).unwrap();

        todo!()
    }

    fn build_proving_assembly() -> Self::ProvingAssembly {
        todo!()
    }

    fn prove(
        _: Self::ProvingAssembly,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }

    fn take_witnesses(
        proving_assembly: &mut Self::ProvingAssembly,
    ) -> Vec<Self::FieldElement, Self::Allocator> {
        todo!()
    }

    fn prove_from_witnesses(
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }

    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        todo!()
    }
}

impl SnarkWrapperProofSystem for FflonkProofSystem {
    fn synthesize_for_proving<P>(input_vk: P::VK, input_proof: P::Proof) -> Self::ProvingAssembly
    where
        P: ProofSystemDefinition,
    {
        todo!()
    }

    fn proof_config() -> Self::ProofConfig {
        todo!()
    }
}

impl ProofSystemExt for FflonkProofSystem {
    type SetupAssembly = FflonkAssembly<SynthesisModeGenerateSetup, Self::Allocator>;
    fn generate_precomputation_and_vk(
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> (AsyncHandler<Self::Precomputation>, Self::VK) {
        todo!()
    }
}

pub struct MarkerProofSystem;
impl ProofSystemDefinition for MarkerProofSystem {
    type FieldElement = ();
    type Precomputation = MarkerPrecomputation;
    type Proof = ();
    type VK = ();
    type FinalizationHint = ();
    type Allocator = std::alloc::Global;
    type ProvingAssembly = ();
    type ContextConfig = ();
    type Context = ();
    type ProofConfig = ();
    // type Circuit = ();
    fn get_context_config() -> Self::ContextConfig {
        todo!()
    }
    fn init_context(config: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        todo!()
    }

    fn build_proving_assembly() -> Self::ProvingAssembly {
        todo!()
    }

    fn prove(
        _: Self::ProvingAssembly,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }

    fn prove_from_witnesses(
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: Self::Precomputation,
        _: Self::FinalizationHint,
        _: Self::ProofConfig,
    ) -> Self::Proof {
        todo!()
    }

    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        todo!()
    }

    fn take_witnesses(
        proving_assembly: &mut Self::ProvingAssembly,
    ) -> Vec<Self::FieldElement, Self::Allocator> {
        todo!()
    }
}
