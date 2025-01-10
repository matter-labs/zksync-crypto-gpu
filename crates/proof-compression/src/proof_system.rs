use std::alloc::Allocator;

use super::*;
use bellman::kate_commitment::{Crs, CrsForMonomialForm};
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
use bellman::CurveAffine;
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
use fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG;
use fflonk::{init_compact_crs, CombinedMonomialDeviceStorage, DeviceContextWithSingleDevice};
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
    fn init_context(config: Self::ContextConfig) -> Self::Context;
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
        _: &Self::VK,
    ) -> Self::Proof;

    fn prove_from_witnesses(
        _: AsyncHandler<Self::Context>,
        _: Self::ExternalWitnessData,
        _: Self::AuxConfig,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
        _: &Self::VK,
    ) -> Self::Proof;
}

pub trait CompressionProofSystemExt: CompressionProofSystem {
    type SetupAssembly;
    fn generate_precomputation_and_vk(
        _: AsyncHandler<Self::Context>,
        _: Self::SetupAssembly,
        _: &Self::FinalizationHint,
    ) -> (Self::Precomputation, Self::VK);
    fn synthesize_for_setup(
        circuit: CompressionLayerCircuit<Self>,
    ) -> (Self::FinalizationHint, Self::SetupAssembly);
}

pub trait SnarkWrapperProofSystem: ProofSystemDefinition {
    type CRS;
    type Circuit;
    type ContextConfig: Send + Sync + 'static;
    type Context: Send + Sync + 'static;
    fn get_context_config() -> Self::ContextConfig;
    fn init_context(config: Self::ContextConfig) -> Self::Context;
    fn load_crs(_: Self::FinalizationHint) -> Self::CRS;
    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly;
    fn prove(
        _: AsyncHandler<Self::Context>,
        _: Self::ProvingAssembly,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
        _: &Self::VK,
    ) -> Self::Proof;

    fn prove_from_witnesses(
        _: AsyncHandler<Self::Context>,
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
        _: &Self::VK,
    ) -> Self::Proof;
}

pub trait SnarkWrapperProofSystemExt: SnarkWrapperProofSystem {
    type SetupAssembly;
    fn synthesize_for_setup(circuit: Self::Circuit) -> Self::SetupAssembly;
    fn generate_precomputation_and_vk(
        _: AsyncHandler<Self::Context>,
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> (Self::Precomputation, Self::VK);
    fn create_crs(_: Self::FinalizationHint);
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
    type ContextConfig = usize; // domain_size
    type Context = ProverContext;

    fn get_context_config() -> Self::ContextConfig {
        // println!("Using hardcoded domain size 2^17 for compression step");
        1 << 17
    }

    fn init_context(domain_size: Self::ContextConfig) -> Self::Context {
        let config =
            ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
        let context = Self::Context::create_with_config(config).expect("gpu prover context");

        context
    }
    fn get_context_config_from_hint(hint: &Self::FinalizationHint) -> Self::ContextConfig {
        hint.final_trace_len
    }
    fn aux_config_from_assembly(proving_assembly: &Self::ProvingAssembly) -> Self::AuxConfig {
        GpuProofConfig::from_assembly(proving_assembly)
    }

    fn synthesize_for_proving(
        circuit: CompressionLayerCircuit<CF>,
        finalization_hint: Self::FinalizationHint,
    ) -> Self::ProvingAssembly {
        let geometry = circuit.geometry();
        let (max_trace_len, num_vars) = circuit.size_hint();

        let builder_impl = boojum::cs::cs_builder_reference::CsReferenceImplementationBuilder::<
            GoldilocksField,
            GoldilocksField,
            ProvingCSConfig,
        >::new(geometry, max_trace_len.unwrap());
        let builder = boojum::cs::cs_builder::new_builder::<_, GoldilocksField>(builder_impl);

        let builder = circuit.configure_builder_proxy(builder);
        let mut cs = builder.build(num_vars.unwrap());
        circuit.add_tables(&mut cs);
        circuit.synthesize_into_cs(&mut cs);
        let _ = cs.pad_and_shrink_using_hint(&finalization_hint);
        let cs = cs.into_assembly::<std::alloc::Global>();

        cs
    }

    fn prove(
        ctx: AsyncHandler<Self::Context>,
        proving_assembly: Self::ProvingAssembly,
        aux_config: Self::AuxConfig,
        precomputation: AsyncHandler<Self::Precomputation>,
        finalization_hint: Self::FinalizationHint,
        vk: &Self::VK,
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
        vk: &Self::VK,
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
        ctx: AsyncHandler<Self::Context>,
        setup_assembly: Self::SetupAssembly,
        finalization_hint: &Self::FinalizationHint,
    ) -> (Self::Precomputation, Self::VK) {
        let worker = Worker::new();
        let proof_config = CF::proof_config_for_compression_step();
        let (setup_base, vk_params, vars_hint, wits_hint) = setup_assembly.get_light_setup(
            &worker,
            proof_config.fri_lde_factor,
            proof_config.merkle_tree_cap_size,
        );
        let domain_size = vk_params.domain_size as usize;
        assert_eq!(finalization_hint.final_trace_len, domain_size);
        let ctx = ctx.wait();
        let (precomputation, vk) =
            shivini::cs::gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
                CF::ThisLayerHasher,
                _,
            >(setup_base, vk_params, vars_hint, wits_hint, &worker)
            .unwrap();
        drop(ctx);
        (precomputation, vk)
    }
    fn synthesize_for_setup(
        circuit: CompressionLayerCircuit<Self>,
    ) -> (Self::FinalizationHint, Self::SetupAssembly) {
        let geometry = circuit.geometry();
        let (max_trace_len, num_vars) = circuit.size_hint();

        let builder_impl = boojum::cs::cs_builder_reference::CsReferenceImplementationBuilder::<
            GoldilocksField,
            GoldilocksField,
            SetupCSConfig,
        >::new(geometry, max_trace_len.unwrap());
        let builder = boojum::cs::cs_builder::new_builder::<_, GoldilocksField>(builder_impl);

        let builder = circuit.configure_builder_proxy(builder);
        let mut cs = builder.build(num_vars.unwrap());
        circuit.add_tables(&mut cs);
        circuit.synthesize_into_cs(&mut cs);
        let (_domain_size, finalization_hint) = cs.pad_and_shrink();
        let cs = cs.into_assembly::<std::alloc::Global>();

        (finalization_hint, cs)
    }
}

type PlonkAssembly<CSConfig, A> = Assembly<
    Bn256,
    PlonkCsWidth4WithNextStepParams,
    SelectorOptimizedWidth4MainGateWithDNext,
    CSConfig,
    A,
>;

pub struct UnsafePlonkProverDeviceMemoryManagerWrapper(
    DeviceMemoryManager<Fr, PlonkProverDeviceMemoryManagerConfig>,
);
unsafe impl Send for UnsafePlonkProverDeviceMemoryManagerWrapper {}
unsafe impl Sync for UnsafePlonkProverDeviceMemoryManagerWrapper {}
pub struct PlonkProverDeviceMemoryManagerConfig;

impl ManagerConfigs for PlonkProverDeviceMemoryManagerConfig {
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
    type CRS = bellman::kate_commitment::Crs<bellman::compact_bn256::Bn256, CrsForMonomialForm>;
    type Circuit = PlonkSnarkVerifierCircuit;
    type ContextConfig = (Vec<usize>, Vec<CompactG1Affine>);
    type Context = UnsafePlonkProverDeviceMemoryManagerWrapper;
    fn init_context(config: Self::ContextConfig) -> Self::Context {
        let (device_ids, compact_crs) = config;
        let manager = DeviceMemoryManager::init(&device_ids, &compact_crs[..]).unwrap();
        UnsafePlonkProverDeviceMemoryManagerWrapper(manager)
    }

    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly {
        todo!()
    }

    fn prove(
        _: AsyncHandler<Self::Context>,
        _: Self::ProvingAssembly,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
        _: &Self::VK,
    ) -> Self::Proof {
        todo!()
    }

    fn prove_from_witnesses(
        _: AsyncHandler<Self::Context>,
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: AsyncHandler<Self::Precomputation>,
        _: Self::FinalizationHint,
        _: &Self::VK,
    ) -> Self::Proof {
        todo!()
    }

    fn get_context_config() -> Self::ContextConfig {
        todo!()
    }

    fn load_crs(domain_size: Self::FinalizationHint) -> Self::CRS {
        let raw_compact_crs_file_path = std::env::var("PLONK_COMPACT_RAW_CRS_FILE").unwrap();
        let raw_compact_crs_file = std::fs::File::open(raw_compact_crs_file_path).unwrap();
        let num_points = domain_size * L1_VERIFIER_DOMAIN_SIZE_LOG;
        read_crs_from_raw_compact_form::<_, Self::Allocator>(raw_compact_crs_file, num_points)
            .unwrap()
    }
}

impl SnarkWrapperProofSystemExt for PlonkSnarkWrapper {
    type SetupAssembly = PlonkAssembly<SynthesisModeGenerateSetup, Self::Allocator>;

    fn synthesize_for_setup(circuit: Self::Circuit) -> Self::SetupAssembly {
        todo!()
    }

    fn generate_precomputation_and_vk(
        _: AsyncHandler<Self::Context>,
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> (Self::Precomputation, Self::VK) {
        todo!()
    }

    fn create_crs(_: Self::FinalizationHint) {
        todo!()
    }
}

type FflonkAssembly<CSConfig, A> = Assembly<Bn256, PlonkCsWidth3Params, NaiveMainGate, CSConfig, A>;

impl ProofSystemDefinition for FflonkSnarkWrapper {
    type FieldElement = Fr;
    type Precomputation = FflonkSnarkVerifierCircuitDeviceSetupWrapper<Self::Allocator>;
    type ExternalWitnessData = (
        Vec<Self::FieldElement>,
        Vec<Self::FieldElement, Self::Allocator>,
    );
    type Proof = FflonkSnarkVerifierCircuitProof;
    type VK = FflonkSnarkVerifierCircuitVK;
    type FinalizationHint = usize;
    // type Allocator = GlobalHost; // TODO need global host with preallocated host memory
    type Allocator = std::alloc::Global;
    type ProvingAssembly = FflonkAssembly<SynthesisModeProve, Self::Allocator>;
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
    type CRS = bellman::kate_commitment::Crs<bellman::compact_bn256::Bn256, CrsForMonomialForm>;
    type Circuit = FflonkSnarkVerifierCircuit;
    type ContextConfig = (usize, Self::CRS);
    type Context = DeviceContextWithSingleDevice;

    fn load_crs(domain_size: Self::FinalizationHint) -> Self::CRS {
        // let raw_compact_crs_file_path = std::env::var("FFLONK_COMPACT_RAW_CRS_FILE").unwrap();
        // let raw_compact_crs_file = std::fs::File::open(raw_compact_crs_file_path).unwrap();
        // let num_points = domain_size * fflonk::MAX_COMBINED_DEGREE_FACTOR;
        // read_crs_from_raw_compact_form::<_, Self::Allocator>(raw_compact_crs_file, num_points)
        //     .unwrap()
        init_compact_crs(&bellman::worker::Worker::new(), domain_size)
    }

    fn get_context_config() -> Self::ContextConfig {
        let domain_size = 1 << fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG;
        let crs = Self::load_crs(domain_size);
        (domain_size, crs)
    }

    fn init_context(config: Self::ContextConfig) -> Self::Context {
        let (domain_size, crs) = config;
        let context = Self::Context::init_from_preloaded_crs(domain_size, crs).unwrap();

        context
    }

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
        _vk: &Self::VK,
    ) -> Self::Proof {
        assert!(proving_assembly.is_satisfied());
        let raw_trace_len = proving_assembly.n();
        assert!(finalization_hint.is_power_of_two());
        proving_assembly.finalize_to_size_log_2(finalization_hint.trailing_zeros() as usize);
        let domain_size = proving_assembly.n() + 1;
        assert!(domain_size.is_power_of_two());
        assert_eq!(domain_size, finalization_hint);

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
        _: &Self::VK,
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
        ctx: AsyncHandler<Self::Context>,
        setup_assembly: Self::SetupAssembly,
        _finalization_hint: Self::FinalizationHint,
    ) -> (Self::Precomputation, Self::VK) {
        let ctx = ctx.wait();
        let device_setup = fflonk::FflonkDeviceSetup::<
            Bn256,
            FflonkSnarkVerifierCircuit,
            Self::Allocator,
        >::create_setup_from_assembly_on_device(&setup_assembly)
        .unwrap();
        let vk = device_setup.get_verification_key();
        drop(ctx);
        (
            FflonkSnarkVerifierCircuitDeviceSetupWrapper(device_setup),
            vk,
        )
    }

    fn create_crs(domain_size: Self::FinalizationHint) {
        assert!(domain_size < fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG);
        let num_points = fflonk::MAX_COMBINED_DEGREE_FACTOR * domain_size;
        let original_crs = make_fflonk_crs_from_ignition_transcripts(num_points);
        let raw_compact_crs_file_path = std::env::var("FFLONK_COMPACT_RAW_CRS_FILE").unwrap();
        assert!(!std::path::Path::exists(std::path::Path::new(
            &raw_compact_crs_file_path
        )));
        let raw_compact_crs_file = std::fs::File::create(raw_compact_crs_file_path).unwrap();
        write_crs_into_raw_compact_form(original_crs, raw_compact_crs_file, num_points).unwrap();
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

    fn verify(_: &Self::Proof, _: &Self::VK) -> bool {
        unreachable!()
    }

    fn take_witnesses(proving_assembly: &mut Self::ProvingAssembly) -> Self::ExternalWitnessData {
        unreachable!()
    }
}

pub fn write_crs_into_raw_compact_form<W: std::io::Write>(
    original_crs: Crs<bellman::bn256::Bn256, CrsForMonomialForm>,
    mut dst_raw_compact_crs: W,
    num_points: usize,
) -> std::io::Result<()> {
    assert!(num_points <= original_crs.g1_bases.len());
    use bellman::{PrimeField, PrimeFieldRepr};
    use byteorder::{BigEndian, WriteBytesExt};
    assert!(num_points < u32::MAX as usize);
    dst_raw_compact_crs.write_u32::<BigEndian>(num_points as u32)?;
    for g1_base in original_crs.g1_bases.iter() {
        let (x, y) = g1_base.as_xy();
        x.into_raw_repr().write_be(&mut dst_raw_compact_crs)?;
        y.into_raw_repr().write_be(&mut dst_raw_compact_crs)?;
    }
    for g2_base in original_crs.g2_monomial_bases.iter() {
        let (x, y) = g2_base.as_xy();
        x.c0.into_raw_repr().write_be(&mut dst_raw_compact_crs)?;
        x.c1.into_raw_repr().write_be(&mut dst_raw_compact_crs)?;
        y.c0.into_raw_repr().write_be(&mut dst_raw_compact_crs)?;
        y.c1.into_raw_repr().write_be(&mut dst_raw_compact_crs)?;
    }

    Ok(())
}

// TODO: Crs doesn't allow bases located in a custom allocator
pub fn read_crs_from_raw_compact_form<R: std::io::Read, A: Allocator + Default>(
    mut src_raw_compact_crs: R,
    num_points: usize,
) -> std::io::Result<Crs<bellman::compact_bn256::Bn256, CrsForMonomialForm>> {
    println!("Reading Raw compact CRS");
    use byteorder::{BigEndian, ReadBytesExt};
    let actual_num_points = src_raw_compact_crs.read_u32::<BigEndian>()? as usize;
    assert!(num_points <= actual_num_points as usize);
    use bellman::{PrimeField, PrimeFieldRepr};
    // let mut g1_bases = Vec::with_capacity_in(num_points, A::default());
    println!("Reading G1 points");
    let mut g1_bases = Vec::with_capacity(num_points);
    unsafe {
        g1_bases.set_len(num_points);
        let buf = std::slice::from_raw_parts_mut(
            g1_bases.as_mut_ptr() as *mut u8,
            num_points * std::mem::size_of::<bellman::compact_bn256::G1Affine>(),
        );
        src_raw_compact_crs.read_exact(buf)?;
    }
    let num_g2_points = 2;
    // let mut g2_bases = Vec::with_capacity_in(num_g2_points, A::default());
    println!("Reading G2 points");
    let mut g2_bases = Vec::with_capacity(num_g2_points);
    unsafe {
        g2_bases.set_len(num_g2_points);
        let buf = std::slice::from_raw_parts_mut(
            g2_bases.as_mut_ptr() as *mut u8,
            num_g2_points * std::mem::size_of::<bellman::compact_bn256::G2Affine>(),
        );
        src_raw_compact_crs.read_exact(buf)?;
    }
    let mut compact_crs = Crs::<_, CrsForMonomialForm>::dummy_crs(1);
    compact_crs.g1_bases = std::sync::Arc::new(g1_bases);
    compact_crs.g2_monomial_bases = std::sync::Arc::new(g2_bases);

    Ok(compact_crs)
}
