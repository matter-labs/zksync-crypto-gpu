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

use franklin_crypto::boojum::cs::{
    implementations::proof::Proof, implementations::verifier::VerificationKey,
};

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

pub struct UnsafePlonkProverDeviceMemoryManagerWrapper(
    DeviceMemoryManager<Fr, PlonkProverDeviceMemoryManagerConfig>,
);
impl GenericWrapper for UnsafePlonkProverDeviceMemoryManagerWrapper {
    type Inner = DeviceMemoryManager<Fr, PlonkProverDeviceMemoryManagerConfig>;

    fn into_inner(self) -> Self::Inner {
        self.0
    }
    fn into_inner_ref(&self) -> &Self::Inner {
        &self.0
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

impl PlonkSnarkWrapper {
    pub fn prove_plonk_snark_wrapper_step(
        input_proof: Proof<
            GoldilocksField,
            <Self as SnarkWrapperStep>::PreviousStepTreeHasher,
            GoldilocksExt2,
        >,
        setup_data_cache: SnarkWrapperSetupData<Self>,
    ) -> anyhow::Result<<Self as ProofSystemDefinition>::Proof> {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let input_vk = setup_data_cache.previous_vk;
        let mut ctx = setup_data_cache.ctx.into_inner();
        let finalization_hint = setup_data_cache.finalization_hint;
        let circuit = Self::build_circuit(input_vk.clone(), Some(input_proof));
        let mut proving_assembly =
            <Self as SnarkWrapperProofSystem>::synthesize_for_proving(circuit);
        let vk = setup_data_cache.vk;
        let mut precomputation = setup_data_cache.precomputation.into_inner();

        assert!(proving_assembly.is_satisfied());
        assert!(finalization_hint.is_power_of_two());
        proving_assembly.finalize_to_size_log_2(finalization_hint.trailing_zeros() as usize);
        let domain_size = proving_assembly.n() + 1;
        assert!(domain_size.is_power_of_two());
        assert_eq!(domain_size, finalization_hint.clone());

        let worker = bellman::worker::Worker::new();
        let start = std::time::Instant::now();
        let proof =
            gpu_prover::create_proof::<_, _, <Self as ProofSystemDefinition>::Transcript, _>(
                &proving_assembly,
                &mut ctx,
                &worker,
                &mut precomputation,
                None,
            )
            .map_err(|e| {
                anyhow::anyhow!("Failed to create proof for PlonkSnarkWrapper: {:?}", e)
            })?;
        println!("plonk proving takes {} s", start.elapsed().as_secs());
        ctx.free_all_slots();

        assert!(<Self as ProofSystemDefinition>::verify(&proof, &vk));

        Ok(proof)
    }

    pub fn precompute_plonk_snark_wrapper_circuit(
        input_vk: VerificationKey<
            GoldilocksField,
            <Self as SnarkWrapperStep>::PreviousStepTreeHasher,
        >,
        hardcoded_finalization_hint: <Self as ProofSystemDefinition>::FinalizationHint,
        ctx: <Self as SnarkWrapperProofSystem>::Context,
    ) -> anyhow::Result<(
        <Self as ProofSystemDefinition>::Precomputation,
        <Self as ProofSystemDefinition>::VK,
    )> {
        let circuit = Self::build_circuit(input_vk, None);
        let mut setup_assembly =
            <Self as SnarkWrapperProofSystemExt>::synthesize_for_setup(circuit);
        assert!(setup_assembly.is_satisfied());
        assert!(hardcoded_finalization_hint.is_power_of_two());
        setup_assembly
            .finalize_to_size_log_2(hardcoded_finalization_hint.trailing_zeros() as usize);
        let domain_size = setup_assembly.n() + 1;
        assert!(domain_size.is_power_of_two());
        assert_eq!(domain_size, hardcoded_finalization_hint);
        let mut ctx = ctx.into_inner();

        let worker = bellman::worker::Worker::new();
        let mut precomputation = AsyncSetup::<<Self as ProofSystemDefinition>::Allocator>::allocate(
            hardcoded_finalization_hint,
        );
        precomputation
            .generate_from_assembly(&worker, &setup_assembly, &mut ctx)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to generate precomputation for PlonkSnarkWrapper: {:?}",
                    e
                )
            })?;

        let hardcoded_g2_bases = hardcoded_canonical_g2_bases();
        let mut dummy_crs = Crs::<bellman::bn256::Bn256, CrsForMonomialForm>::dummy_crs(1);
        dummy_crs.g2_monomial_bases = std::sync::Arc::new(hardcoded_g2_bases.to_vec());
        let vk = gpu_prover::compute_vk_from_assembly::<
            _,
            _,
            PlonkCsWidth4WithNextStepAndCustomGatesParams,
            SynthesisModeGenerateSetup,
        >(&mut ctx, &setup_assembly, &dummy_crs)
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to compute verification key for PlonkSnarkWrapper: {:?}",
                e
            )
        })?;

        ctx.free_all_slots();

        Ok((
            PlonkSnarkVerifierCircuitDeviceSetupWrapper::from_inner(precomputation),
            vk,
        ))
    }
}

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

    fn load_compact_raw_crs<R: std::io::Read>(src: R) -> anyhow::Result<Self::CRS> {
        let num_g1_points_for_crs = 1 << PlonkProverDeviceMemoryManagerConfig::FULL_SLOT_SIZE_LOG;
        Ok(read_crs_from_raw_compact_form(src, num_g1_points_for_crs)?)
    }

    fn init_context(compact_raw_crs: Self::CRS) -> anyhow::Result<Self::Context> {
        let device_ids: Vec<_> =
            (0..<PlonkProverDeviceMemoryManagerConfig as ManagerConfigs>::NUM_GPUS).collect();
        let manager =
            DeviceMemoryManager::init(&device_ids, &compact_raw_crs.g1_bases).map_err(|e| {
                anyhow::anyhow!("Failed to initialize Plonk device memory manager: {:?}", e)
            })?;
        Ok(UnsafePlonkProverDeviceMemoryManagerWrapper(manager))
    }

    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly {
        let mut proving_assembly = PlonkAssembly::<SynthesisModeProve, Self::Allocator>::new();
        circuit
            .synthesize(&mut proving_assembly)
            .expect("must work");
        proving_assembly
    }

    fn prove(
        _: &Self::Context,
        _: Self::ProvingAssembly,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
    ) -> anyhow::Result<Self::Proof> {
        // We use a custom proving function because Plonk requires mutable ownership of the setup data
        unimplemented!()
    }

    fn prove_from_witnesses(
        _: &Self::Context,
        _: Vec<Self::FieldElement, Self::Allocator>,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
    ) -> anyhow::Result<Self::Proof> {
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
        _: &Self::Context,
        _: Self::SetupAssembly,
        _: Self::FinalizationHint,
    ) -> anyhow::Result<(Self::Precomputation, Self::VK)> {
        unimplemented!()
    }
}
