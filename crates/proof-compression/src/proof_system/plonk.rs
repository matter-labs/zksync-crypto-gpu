use std::fs::File;

use bellman::{
    kate_commitment::{Crs, CrsForMonomialForm},
    plonk::{
        better_better_cs::{
            cs::{
                Assembly, Circuit, PlonkCsWidth4WithNextStepAndCustomGatesParams,
                SynthesisModeGenerateSetup, SynthesisModeProve,
            },
            gates::selector_optimized_with_d_next::SelectorOptimizedWidth4MainGateWithDNext,
        },
        commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    },
};
use circuit_definitions::circuit_definitions::aux_layer::ZkSyncSnarkWrapperCircuit;
use gpu_prover::{AsyncSetup, DeviceMemoryManager, ManagerConfigs};

use super::*;

use bellman::plonk::better_better_cs::{
    cs::VerificationKey as PlonkVerificationKey, proof::Proof as PlonkProof,
};

use bellman::bn256::{Bn256, Fr};

pub(crate) type PlonkSnarkVerifierCircuit = ZkSyncSnarkWrapperCircuit;
pub(crate) type PlonkSnarkVerifierCircuitVK =
    PlonkVerificationKey<Bn256, PlonkSnarkVerifierCircuit>;
pub(crate) type PlonkSnarkVerifierCircuitProof = PlonkProof<Bn256, PlonkSnarkVerifierCircuit>;
pub(crate) type PlonkSnarkVerifierCircuitDeviceSetup = AsyncSetup;

type PlonkAssembly<CSConfig, A> = Assembly<
    Bn256,
    PlonkCsWidth4WithNextStepAndCustomGatesParams,
    SelectorOptimizedWidth4MainGateWithDNext,
    CSConfig,
    A,
>;

const COMPACT_CRS_ENV_VAR: &str = "COMPACT_CRS_FILE";

pub struct UnsafePlonkProverDeviceMemoryManagerWrapper(
    DeviceMemoryManager<Fr, PlonkProverDeviceMemoryManagerConfig>,
);
impl GenericWrapper for UnsafePlonkProverDeviceMemoryManagerWrapper {
    type Inner = DeviceMemoryManager<Fr, PlonkProverDeviceMemoryManagerConfig>;

    fn into_inner(self) -> Self::Inner {
        self.0
    }

    fn from_inner(inner: Self::Inner) -> Self {
        Self(inner)
    }
}
unsafe impl Send for UnsafePlonkProverDeviceMemoryManagerWrapper {}
unsafe impl Sync for UnsafePlonkProverDeviceMemoryManagerWrapper {}
pub struct PlonkProverDeviceMemoryManagerConfig;

impl ManagerConfigs for PlonkProverDeviceMemoryManagerConfig {
    const NUM_GPUS_LOG: usize = 0;
    const FULL_SLOT_SIZE_LOG: usize = 24;
    const NUM_SLOTS: usize = 29;
    const NUM_HOST_SLOTS: usize = 2;
}

pub struct PlonkSnarkWrapper;

impl ProofSystemDefinition for PlonkSnarkWrapper {
    type FieldElement = Fr;
    type Precomputation = PlonkSnarkVerifierCircuitDeviceSetupWrapper;
    type ExternalWitnessData = Vec<Self::FieldElement, Self::Allocator>;
    type Proof = PlonkSnarkVerifierCircuitProof;
    type VK = PlonkSnarkVerifierCircuitVK;
    type FinalizationHint = usize;
    type Allocator = gpu_prover::cuda_bindings::CudaAllocator;
    type ProvingAssembly = PlonkAssembly<SynthesisModeProve, Self::Allocator>;
    type Transcript = RollingKeccakTranscript<Self::FieldElement>;
    fn take_witnesses(
        _proving_assembly: &mut Self::ProvingAssembly,
    ) -> Vec<Self::FieldElement, Self::Allocator> {
        // let input_assignments =
        //     std::mem::replace(&mut proving_assembly.input_assingments, Vec::new());
        // let aux_assignments = std::mem::replace(
        //     &mut proving_assembly.aux_assingments,
        //     Vec::new_in(Self::Allocator::default()),
        // );

        todo!()
    }
    fn verify(proof: &Self::Proof, vk: &Self::VK) -> bool {
        bellman::plonk::better_better_cs::verifier::verify::<_, _, Self::Transcript>(
            vk, proof, None,
        )
        .unwrap()
    }
}

impl SnarkWrapperProofSystem for PlonkSnarkWrapper {
    type Circuit = PlonkSnarkVerifierCircuit;
    type Context = UnsafePlonkProverDeviceMemoryManagerWrapper;

    type CRS = bellman::kate_commitment::Crs<
        bellman::compact_bn256::Bn256,
        CrsForMonomialForm,
        Self::Allocator,
    >;
    fn pre_init() {
        // TODO: initialize static pinned memory
    }

    fn load_compact_raw_crs<R: std::io::Read>(src: R) -> Self::CRS {
        let num_g1_points_for_crs = 1 << PlonkProverDeviceMemoryManagerConfig::FULL_SLOT_SIZE_LOG;
        read_crs_from_raw_compact_form(src, num_g1_points_for_crs).unwrap()
    }

    fn init_context(compact_raw_crs: Self::CRS) -> Self::Context {
        let device_ids: Vec<_> =
            (0..<PlonkProverDeviceMemoryManagerConfig as ManagerConfigs>::NUM_GPUS).collect();
        let manager = DeviceMemoryManager::init(&device_ids, &compact_raw_crs.g1_bases).unwrap();
        UnsafePlonkProverDeviceMemoryManagerWrapper(manager)
    }

    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly {
        let mut proving_assembly = PlonkAssembly::<SynthesisModeProve, Self::Allocator>::new();
        circuit
            .synthesize(&mut proving_assembly)
            .expect("must work");
        proving_assembly
    }

    fn prove(
        ctx: &Self::Context,
        mut proving_assembly: Self::ProvingAssembly,
        precomputation: &Self::Precomputation,
        finalization_hint: &Self::FinalizationHint,
    ) -> Self::Proof {
        assert!(proving_assembly.is_satisfied());
        assert!(finalization_hint.is_power_of_two());
        proving_assembly.finalize_to_size_log_2(finalization_hint.trailing_zeros() as usize);
        let domain_size = proving_assembly.n() + 1;
        assert!(domain_size.is_power_of_two());
        assert_eq!(domain_size, finalization_hint.clone());

        // ctx loading
        let filepath =
            std::env::var(COMPACT_CRS_ENV_VAR).expect("No compact CRS file path provided");
        let reader = Box::new(File::open(filepath).unwrap());
        let crs = <Self as SnarkWrapperProofSystem>::load_compact_raw_crs(reader);
        let ctx = Self::init_context(crs);
        let mut ctx = ctx.into_inner();

        // precomputation placeholder
        let mut precomputation = AsyncSetup::empty();

        let worker = bellman::worker::Worker::new();
        let start = std::time::Instant::now();
        let proof = gpu_prover::create_proof::<_, _, Self::Transcript, _>(
            &proving_assembly,
            &mut ctx,
            &worker,
            &mut precomputation,
            None,
        )
        .unwrap();
        println!("plonk proving takes {} s", start.elapsed().as_secs());
        ctx.free_all_slots();

        proof
    }

    fn prove_from_witnesses(
        _: &Self::Context,
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
    ) -> Self::Proof {
        unimplemented!()
    }
}

impl SnarkWrapperProofSystemExt for PlonkSnarkWrapper {
    type SetupAssembly = PlonkAssembly<SynthesisModeGenerateSetup, Self::Allocator>;

    fn synthesize_for_setup(circuit: Self::Circuit) -> Self::SetupAssembly {
        let mut setup_assembly =
            PlonkAssembly::<SynthesisModeGenerateSetup, Self::Allocator>::new();
        circuit.synthesize(&mut setup_assembly).expect("must work");
        setup_assembly
    }

    fn generate_precomputation_and_vk(
        ctx: &Self::Context,
        mut setup_assembly: Self::SetupAssembly,
        hardcoded_finalization_hint: Self::FinalizationHint,
    ) -> (Self::Precomputation, Self::VK) {
        assert!(setup_assembly.is_satisfied());
        assert!(hardcoded_finalization_hint.is_power_of_two());
        setup_assembly
            .finalize_to_size_log_2(hardcoded_finalization_hint.trailing_zeros() as usize);
        let domain_size = setup_assembly.n() + 1;
        assert!(domain_size.is_power_of_two());
        assert_eq!(domain_size, hardcoded_finalization_hint);

        // let mut ctx = ctx.into_inner();
        let filepath =
            std::env::var(COMPACT_CRS_ENV_VAR).expect("No compact CRS file path provided");
        let reader = Box::new(File::open(filepath).unwrap());
        let crs = <Self as SnarkWrapperProofSystem>::load_compact_raw_crs(reader);
        let ctx = Self::init_context(crs);
        let mut ctx = ctx.into_inner();

        let worker = bellman::worker::Worker::new();
        let mut precomputation =
            AsyncSetup::<Self::Allocator>::allocate(hardcoded_finalization_hint);
        precomputation
            .generate_from_assembly(&worker, &setup_assembly, &mut ctx)
            .unwrap();

        let hardcoded_g2_bases = hardcoded_canonical_g2_bases();
        let mut dummy_crs = Crs::<bellman::bn256::Bn256, CrsForMonomialForm>::dummy_crs(1);
        dummy_crs.g2_monomial_bases = std::sync::Arc::new(hardcoded_g2_bases.to_vec());
        let vk = gpu_prover::compute_vk_from_assembly::<
            _,
            _,
            PlonkCsWidth4WithNextStepAndCustomGatesParams,
            SynthesisModeGenerateSetup,
        >(&mut ctx, &setup_assembly, &dummy_crs)
        .unwrap();

        ctx.free_all_slots();

        (
            PlonkSnarkVerifierCircuitDeviceSetupWrapper::from_inner(precomputation),
            vk,
        )
    }
}
