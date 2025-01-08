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
use bellman::plonk::commitments::transcript::keccak_transcript::RollingKeccakTranscript;
use boojum::config::SetupCSConfig;
use boojum::cs::implementations::transcript::Transcript;
use boojum::cs::implementations::witness::WitnessVec;
use boojum::cs::traits::circuit::CircuitBuilderProxy;
use boojum::cs::traits::GoodAllocator;
use boojum::worker::Worker;
use boojum::{
    config::ProvingCSConfig,
    cs::implementations::{
        fast_serialization::MemcopySerializable, proof::Proof, reference_cs::CSReferenceAssembly,
        setup::FinalizationHintsForProver, verifier::VerificationKey,
    },
    field::goldilocks::GoldilocksExt2,
};
use circuit_definitions::circuit_definitions::aux_layer::compression::{
    CompressionLayerCircuit, ProofCompressionFunction,
};

use fflonk::bellman::plonk::better_better_cs::cs::Circuit;
use fflonk::{CombinedMonomialDeviceStorage, DeviceContextWithSingleDevice};
use gpu_prover::{DeviceMemoryManager, ManagerConfigs};
use shivini::gpu_proof_config::GpuProofConfig;
use shivini::{cs::GpuSetup, GpuTreeHasher, ProverContext};
use shivini::{
    CacheStrategy, CommitmentCacheStrategy, GPUPoWRunner, PolynomialsCacheStrategy,
    ProverContextConfig,
};
pub trait ProofSystemDefinition: Sized {
    type FieldElement;
    type ExternalWitnessData;
    type Precomputation: MemcopySerializable + Send + Sync + 'static;
    type Proof: serde::Serialize + serde::de::DeserializeOwned;
    type VK: serde::Serialize + serde::de::DeserializeOwned + Send + Sync + Clone + 'static;
    type FinalizationHint: serde::Serialize + serde::de::DeserializeOwned + Clone;
    type Allocator: std::alloc::Allocator;
    type ProvingAssembly: Sized + Send + Sync + 'static;
    type ContextConfig;
    type Context;
    fn get_context_config() -> Self::ContextConfig;
    fn init_context(config: Self::ContextConfig) -> AsyncHandler<Self::Context>;
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
    type AuxConfig;

    fn aux_config_from_assembly(proving_assembly: &Self::ProvingAssembly) -> Self::AuxConfig;

    fn synthesize_for_proving(
        circuit: CompressionLayerCircuit<Self>,
        finalization_hint: Self::FinalizationHint,
    ) -> Self::ProvingAssembly;

    fn prove(
        _: AsyncHandler<Self::Context>,
        _: Self::ProvingAssembly,
        _: Self::AuxConfig,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
        _: Self::VK,
    ) -> Self::Proof;

    fn prove_from_witnesses(
        _: AsyncHandler<Self::Context>,
        _: Self::ExternalWitnessData,
        _: Self::AuxConfig,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
        _: Self::VK,
    ) -> Self::Proof;
}

pub trait CompressionProofSystemExt: CompressionProofSystem {
    type SetupAssembly;
    fn generate_precomputation_and_vk(
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> AsyncHandler<(Self::Precomputation, Self::VK)>;
    fn synthesize_for_setup(
        circuit: CompressionLayerCircuit<Self>,
    ) -> (Self::FinalizationHint, Self::SetupAssembly);
}

pub trait SnarkWrapperProofSystem: ProofSystemDefinition {
    type Circuit;
    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly;
    fn prove(
        _: AsyncHandler<Self::Context>,
        _: Self::ProvingAssembly,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
    ) -> Self::Proof;

    fn prove_from_witnesses(
        _: AsyncHandler<Self::Context>,
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
    ) -> Self::Proof;
}

pub trait SnarkWrapperProofSystemExt: SnarkWrapperProofSystem {
    type SetupAssembly;
    fn synthesize_for_setup(circuit: Self::Circuit) -> Self::SetupAssembly;
    fn generate_precomputation_and_vk(
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> AsyncHandler<(Self::Precomputation, Self::VK)>;
}

type BoojumAssembly<CSConfig, A> =
    CSReferenceAssembly<GoldilocksField, GoldilocksField, CSConfig, A>;

impl<CF> ProofSystemDefinition for CF
where
    CF: ProofCompressionFunction<
            ThisLayerHasher: GpuTreeHasher,
            ThisLayerTranscript: Transcript<
                GoldilocksField,
                TransciptParameters = (),
                CompatibleCap: Send + Sync + 'static,
            >,
            ThisLayerPoW: GPUPoWRunner,
        > + 'static,
{
    type FieldElement = GoldilocksField;
    type ExternalWitnessData = WitnessVec<Self::FieldElement, Self::Allocator>;
    type Precomputation = GpuSetup<CF::ThisLayerHasher>;
    type Proof = Proof<Self::FieldElement, CF::ThisLayerHasher, GoldilocksExt2>;
    type VK = VerificationKey<Self::FieldElement, CF::ThisLayerHasher>;
    type FinalizationHint = FinalizationHintsForProver;
    type Allocator = std::alloc::Global;
    type ProvingAssembly = BoojumAssembly<ProvingCSConfig, Self::Allocator>;
    type ContextConfig = usize; // domain_size
    type Context = ProverContext;
    fn get_context_config() -> Self::ContextConfig {
        todo!("domain size")
    }

    fn init_context(domain_size: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let config =
                ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
            let context = Self::Context::create_with_config(config).expect("gpu prover context");
            sender.send(context).unwrap();

            receiver
        };

        AsyncHandler::spawn(f)
    }

    fn verify(proof: &Self::Proof, vk: &Self::VK) -> bool {
        let verifier_builder =
            CircuitBuilderProxy::<GoldilocksField, CompressionLayerCircuit<CF>>::dyn_verifier_builder::<GoldilocksExt2>();
        let verifier = verifier_builder.create_verifier();
        verifier.verify::<CF::ThisLayerHasher, CF::ThisLayerTranscript, CF::ThisLayerPoW>(
            (),
            vk,
            proof,
        )
    }

    fn take_witnesses(proving_assembly: &mut Self::ProvingAssembly) -> Self::ExternalWitnessData {
        proving_assembly.witness.take().unwrap()
    }
}

impl<CF> CompressionProofSystem for CF
where
    CF: ProofCompressionFunction<
            ThisLayerHasher: GpuTreeHasher,
            ThisLayerTranscript: Transcript<
                GoldilocksField,
                TransciptParameters = (),
                CompatibleCap: Send + Sync + 'static,
            >,
            ThisLayerPoW: GPUPoWRunner,
        > + 'static,
{
    type AuxConfig = GpuProofConfig;

    fn aux_config_from_assembly(proving_assembly: &Self::ProvingAssembly) -> Self::AuxConfig {
        GpuProofConfig::from_assembly(proving_assembly)
    }

    fn synthesize_for_proving(
        circuit: CompressionLayerCircuit<CF>,
        finalization_hint: Self::FinalizationHint,
    ) -> Self::ProvingAssembly {
        synthesize_circuit_for_proving(circuit, &finalization_hint)
    }

    fn prove(
        ctx: AsyncHandler<Self::Context>,
        proving_assembly: Self::ProvingAssembly,
        aux_config: Self::AuxConfig,
        precomputation: AsyncHandler<Self::Precomputation>,
        finalization_hint: Self::FinalizationHint,
        vk: Self::VK,
    ) -> Self::Proof {
        Self::prove_from_witnesses(
            ctx,
            proving_assembly.witness.unwrap(),
            aux_config,
            precomputation,
            finalization_hint,
            vk,
        )
    }

    fn prove_from_witnesses(
        ctx: AsyncHandler<Self::Context>,
        witness: Self::ExternalWitnessData,
        aux_config: Self::AuxConfig,
        precomputation: AsyncHandler<Self::Precomputation>,
        finalization_hint: Self::FinalizationHint,
        vk: Self::VK,
    ) -> Self::Proof {
        let domain_size = vk.fixed_parameters.domain_size as usize;
        assert_eq!(finalization_hint.final_trace_len, domain_size);
        let cache_strategy = CacheStrategy {
            setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            commitment: CommitmentCacheStrategy::CacheCosetCaps,
        };
        let worker = Worker::new();
        let precomputation = precomputation.wait();
        let ctx = ctx.wait();
        let gpu_proof = shivini::gpu_prove_from_external_witness_data_with_cache_strategy::<
            CF::ThisLayerTranscript,
            CF::ThisLayerHasher,
            CF::ThisLayerPoW,
            Self::Allocator,
        >(
            &aux_config,
            &witness,
            CF::proof_config_for_compression_step(),
            &precomputation,
            &vk,
            (),
            &worker,
            cache_strategy,
        )
        .expect("gpu proof");
        drop(ctx);
        let proof = gpu_proof.into();

        proof
    }
}

impl<CF> CompressionProofSystemExt for CF
where
    CF: ProofCompressionFunction<
            ThisLayerHasher: GpuTreeHasher,
            ThisLayerTranscript: Transcript<
                GoldilocksField,
                TransciptParameters = (),
                CompatibleCap: Send + Sync + 'static,
            >,
            ThisLayerPoW: GPUPoWRunner,
        > + 'static,
    Self::Allocator: GoodAllocator,
{
    type SetupAssembly = BoojumAssembly<SetupCSConfig, Self::Allocator>;
    fn generate_precomputation_and_vk(
        setup_assembly: Self::SetupAssembly,
        finalization_hint: Self::FinalizationHint,
    ) -> AsyncHandler<(Self::Precomputation, Self::VK)> {
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let worker = Worker::new();
            let proof_config = CF::proof_config_for_compression_step();
            let (setup_base, vk_params, vars_hint, wits_hint) = setup_assembly.get_light_setup(
                &worker,
                proof_config.fri_lde_factor,
                proof_config.merkle_tree_cap_size,
            );
            let domain_size = vk_params.domain_size as usize;
            assert_eq!(finalization_hint.final_trace_len, domain_size);
            let config =
                ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
            let _ctx = ProverContext::create_with_config(config).expect("gpu prover context");
            let (device_setup, vk) =
                shivini::cs::gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
                    CF::ThisLayerHasher,
                    _,
                >(setup_base, vk_params, vars_hint, wits_hint, &worker)
                .unwrap();
            sender.send((device_setup, vk)).unwrap();

            receiver
        };

        AsyncHandler::spawn(f)
    }
    fn synthesize_for_setup(
        circuit: CompressionLayerCircuit<Self>,
    ) -> (Self::FinalizationHint, Self::SetupAssembly) {
        synthesize_circuit_for_setup(circuit)
    }
}

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

impl ProofSystemDefinition for PlonkSnarkWrapper {
    type FieldElement = Fr;
    type Precomputation = PlonkSnarkVerifierCircuitDeviceSetupWrapper;
    type ExternalWitnessData = Vec<Self::FieldElement, Self::Allocator>;
    type Proof = PlonkSnarkVerifierCircuitProof;
    type VK = PlonkSnarkVerifierCircuitVK;
    type FinalizationHint = usize;
    type Allocator = GlobalHost;
    type ProvingAssembly = PlonkAssembly<SynthesisModeProve, Self::Allocator>;
    type ContextConfig = (Vec<usize>, Vec<CompactG1Affine>);
    type Context = DeviceMemoryManager<Fr, PlonkProverDeviceMemoryManager>;
    fn get_context_config() -> Self::ContextConfig {
        todo!()
    }
    fn init_context(config: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        let (device_ids, compact_crs) = config;
        let ctx = Self::Context::init(&device_ids, &compact_crs[..]);
        todo!()
    }
    fn take_witnesses(
        proving_assembly: &mut Self::ProvingAssembly,
    ) -> Vec<Self::FieldElement, Self::Allocator> {
        todo!()
    }
    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        todo!()
    }
}

impl SnarkWrapperProofSystem for PlonkSnarkWrapper {
    type Circuit = PlonkSnarkVerifierCircuit;
    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly {
        todo!()
    }

    fn prove(
        _: AsyncHandler<Self::Context>,
        _: Self::ProvingAssembly,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
    ) -> Self::Proof {
        todo!()
    }

    fn prove_from_witnesses(
        _: AsyncHandler<Self::Context>,
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
    ) -> Self::Proof {
        todo!()
    }
}

impl SnarkWrapperProofSystemExt for PlonkSnarkWrapper {
    type SetupAssembly = PlonkAssembly<SynthesisModeGenerateSetup, Self::Allocator>;

    fn synthesize_for_setup(circuit: Self::Circuit) -> Self::SetupAssembly {
        todo!()
    }

    fn generate_precomputation_and_vk(
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> AsyncHandler<(Self::Precomputation, Self::VK)> {
        todo!()
    }
}

type FflonkAssembly<CSConfig, A> = Assembly<Bn256, PlonkCsWidth3Params, NaiveMainGate, CSConfig, A>;

impl ProofSystemDefinition for FflonkSnarkWrapper {
    type FieldElement = Fr;
    type Precomputation = FflonkSnarkVerifierCircuitDeviceSetupWrapper;
    type ExternalWitnessData = (
        Vec<Self::FieldElement>,
        Vec<Self::FieldElement, Self::Allocator>,
    );
    type Proof = FflonkSnarkVerifierCircuitProof;
    type VK = FflonkSnarkVerifierCircuitVK;
    type FinalizationHint = usize;
    type Allocator = GlobalHost;
    type ProvingAssembly = FflonkAssembly<SynthesisModeProve, Self::Allocator>;
    type ContextConfig = usize; // domain_size
    type Context = DeviceContextWithSingleDevice;

    fn get_context_config() -> Self::ContextConfig {
        fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG
    }
    fn init_context(log_domain_size: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let domain_size = 1 << log_domain_size;
            let context = Self::Context::init(domain_size).unwrap();
            sender.send(context).unwrap();

            receiver
        };

        AsyncHandler::spawn(f)
    }
    fn take_witnesses(proving_assembly: &mut Self::ProvingAssembly) -> Self::ExternalWitnessData {
        let input_assignments =
            std::mem::replace(&mut proving_assembly.input_assingments, Vec::new());
        let aux_assignments = std::mem::replace(
            &mut proving_assembly.aux_assingments,
            Vec::new_in(Self::Allocator::default()),
        );

        (input_assignments, aux_assignments)
    }

    fn verify(proof: &Self::Proof, vk: &Self::VK) -> bool {
        fflonk::fflonk_cpu::verify::<_, FflonkSnarkVerifierCircuit, RollingKeccakTranscript<Fr>>(
            vk, proof, None,
        )
        .unwrap()
    }
}

impl SnarkWrapperProofSystem for FflonkSnarkWrapper {
    type Circuit = FflonkSnarkVerifierCircuit;
    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly {
        let mut proving_assembly = FflonkAssembly::<SynthesisModeProve, Self::Allocator>::new();
        circuit
            .synthesize(&mut proving_assembly)
            .expect("must work");
        proving_assembly
    }

    fn prove(
        ctx: AsyncHandler<Self::Context>,
        mut proving_assembly: Self::ProvingAssembly,
        precomputation: AsyncHandler<Self::Precomputation>,
        finalization_hint: Self::FinalizationHint,
    ) -> Self::Proof {
        assert!(proving_assembly.is_satisfied());
        let raw_trace_len = proving_assembly.n();
        proving_assembly.finalize_to_size_log_2(1 << finalization_hint);
        let domain_size = proving_assembly.n() + 1;
        assert!(domain_size.is_power_of_two());
        assert!(domain_size <= 1 << Self::get_context_config());

        let ctx = ctx.wait();
        let precomputation = precomputation.wait().into_inner();
        let proof = fflonk::create_proof::<
            _,
            _,
            _,
            RollingKeccakTranscript<_>,
            CombinedMonomialDeviceStorage<Fr>,
            _,
        >(&proving_assembly, &precomputation, raw_trace_len)
        .unwrap();
        drop(ctx);
        proof
    }

    fn prove_from_witnesses(
        _: AsyncHandler<Self::Context>,
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
    ) -> Self::Proof {
        unimplemented!()
    }
}

impl SnarkWrapperProofSystemExt for FflonkSnarkWrapper {
    type SetupAssembly = FflonkAssembly<SynthesisModeGenerateSetup, Self::Allocator>;

    fn synthesize_for_setup(circuit: Self::Circuit) -> Self::SetupAssembly {
        let mut setup_assembly =
            FflonkAssembly::<SynthesisModeGenerateSetup, Self::Allocator>::new();
        circuit.synthesize(&mut setup_assembly).unwrap();

        setup_assembly
    }

    fn generate_precomputation_and_vk(
        setup_assembly: Self::SetupAssembly,
        finalization_hint: Self::FinalizationHint,
    ) -> AsyncHandler<(Self::Precomputation, Self::VK)> {
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let device_setup =
                FflonkSnarkVerifierCircuitDeviceSetup::create_setup_from_assembly_on_device(
                    &setup_assembly,
                )
                .unwrap();
            let vk = device_setup.get_verification_key();
            sender
                .send((
                    FflonkSnarkVerifierCircuitDeviceSetupWrapper(device_setup),
                    vk,
                ))
                .unwrap();

            receiver
        };

        AsyncHandler::spawn(f)
    }
}

pub struct MarkerProofSystem;
impl ProofSystemDefinition for MarkerProofSystem {
    type FieldElement = ();
    type Precomputation = MarkerPrecomputation;
    type Proof = ();
    type VK = ();
    type ExternalWitnessData = ();
    type FinalizationHint = ();
    type Allocator = std::alloc::Global;
    type ProvingAssembly = ();
    type ContextConfig = ();
    type Context = ();
    fn get_context_config() -> Self::ContextConfig {
        todo!()
    }
    fn init_context(config: Self::ContextConfig) -> AsyncHandler<Self::Context> {
        todo!()
    }

    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        todo!()
    }

    fn take_witnesses(proving_assembly: &mut Self::ProvingAssembly) -> Self::ExternalWitnessData {
        todo!()
    }
}
