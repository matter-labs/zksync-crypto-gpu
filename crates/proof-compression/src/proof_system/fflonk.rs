use super::*;
use ::fflonk::fflonk_cpu::{FflonkProof, FflonkVerificationKey};
pub(crate) use ::fflonk::HostAllocator;
use ::fflonk::{DeviceContextWithSingleDevice, FflonkSnarkVerifierCircuitDeviceSetup};
use bellman::bn256::{Bn256, Fr};
use bellman::{
    kate_commitment::CrsForMonomialForm,
    plonk::{
        better_better_cs::{
            cs::{
                Assembly, Circuit, PlonkCsWidth3Params, SynthesisModeGenerateSetup,
                SynthesisModeProve,
            },
            gates::naive_main_gate::NaiveMainGate,
        },
        commitments::transcript::keccak_transcript::RollingKeccakTranscript,
    },
};
use circuit_definitions::circuit_definitions::aux_layer::ZkSyncSnarkWrapperCircuitNoLookupCustomGate;
pub(crate) type FflonkSnarkVerifierCircuit = ZkSyncSnarkWrapperCircuitNoLookupCustomGate;
pub(crate) type FflonkSnarkVerifierCircuitVK =
    FflonkVerificationKey<Bn256, FflonkSnarkVerifierCircuit>;
pub(crate) type FflonkSnarkVerifierCircuitProof = FflonkProof<Bn256, FflonkSnarkVerifierCircuit>;
type FflonkAssembly<CSConfig, A> = Assembly<Bn256, PlonkCsWidth3Params, NaiveMainGate, CSConfig, A>;
pub struct FflonkSnarkWrapper;

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
    // Pinned memory with small allocations is expensive e.g Assembly storage
    type Allocator = std::alloc::Global;
    type ProvingAssembly = FflonkAssembly<SynthesisModeProve, Self::Allocator>;
    type Transcript = RollingKeccakTranscript<Self::FieldElement>;
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
        ::fflonk::fflonk_cpu::verify::<_, FflonkSnarkVerifierCircuit, Self::Transcript>(
            vk, proof, None,
        )
        .unwrap()
    }
}

impl SnarkWrapperProofSystem for FflonkSnarkWrapper {
    type Circuit = FflonkSnarkVerifierCircuit;
    type Context = DeviceContextWithSingleDevice;
    type CRS = bellman::kate_commitment::Crs<
        bellman::compact_bn256::Bn256,
        CrsForMonomialForm,
        Self::Allocator,
    >;

    fn pre_init() {
        let domain_size = 1 << ::fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG;
        Self::Context::init_pinned_memory(domain_size).unwrap();
    }

    fn load_compact_raw_crs<R: std::io::Read>(src: R) -> anyhow::Result<Self::CRS> {
        let domain_size = 1 << ::fflonk::fflonk_cpu::L1_VERIFIER_DOMAIN_SIZE_LOG;
        let num_g1_bases_for_crs = ::fflonk::fflonk_cpu::MAX_COMBINED_DEGREE_FACTOR * domain_size;
        Ok(read_crs_from_raw_compact_form(src, num_g1_bases_for_crs)?)
    }

    fn init_context(compact_raw_crs: &Self::CRS) -> anyhow::Result<Self::Context> {
        let start = std::time::Instant::now();
        let domain_size = 1 << ::fflonk::fflonk_cpu::L1_VERIFIER_DOMAIN_SIZE_LOG;
        let context = DeviceContextWithSingleDevice::init_from_preloaded_crs::<Self::Allocator>(
            domain_size,
            compact_raw_crs,
        )
        .unwrap();
        println!("fflonk ctx initialization takes {:?}", start.elapsed());
        Ok(context)
    }

    fn synthesize_for_proving(circuit: Self::Circuit) -> Self::ProvingAssembly {
        let mut proving_assembly = FflonkAssembly::<SynthesisModeProve, Self::Allocator>::new();
        circuit
            .synthesize(&mut proving_assembly)
            .expect("must work");
        proving_assembly
    }

    fn prove(
        ctx: Self::Context,
        mut proving_assembly: Self::ProvingAssembly,
        precomputation: &Self::Precomputation,
        finalization_hint: &Self::FinalizationHint,
    ) -> anyhow::Result<Self::Proof> {
        anyhow::ensure!(proving_assembly.is_satisfied());
        let raw_trace_len = proving_assembly.n();
        anyhow::ensure!(finalization_hint.is_power_of_two());
        proving_assembly.finalize_to_size_log_2(finalization_hint.trailing_zeros() as usize);
        let domain_size = proving_assembly.n() + 1;
        anyhow::ensure!(domain_size.is_power_of_two());
        anyhow::ensure!(domain_size == finalization_hint.clone());
        let precomputation = precomputation.into_inner_ref();
        let start = std::time::Instant::now();
        let proof = ::fflonk::create_proof::<_, _, _, RollingKeccakTranscript<_>, _>(
            &proving_assembly,
            &precomputation,
            raw_trace_len,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create proof for FflonkSnarkWrapper: {:?}", e))?;
        println!("fflonk proving takes {} s", start.elapsed().as_secs());
        drop(ctx);
        Ok(proof)
    }

    fn prove_from_witnesses(
        _: Self::Context,
        _: Self::ExternalWitnessData,
        _: &Self::Precomputation,
        _: &Self::FinalizationHint,
    ) -> anyhow::Result<Self::Proof> {
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
        ctx: Self::Context,
        setup_assembly: Self::SetupAssembly,
        _hardcoded_finalization_hint: Self::FinalizationHint,
    ) -> anyhow::Result<(Self::Precomputation, Self::VK)> {
        let device_setup =
            FflonkSnarkVerifierCircuitDeviceSetup::<Self::Allocator>::create_setup_from_assembly_on_device(
                &setup_assembly,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create setup from assembly on device: {:?}",
                    e
                )
            })?;
        let vk = device_setup.get_verification_key();
        drop(ctx);
        Ok((
            FflonkSnarkVerifierCircuitDeviceSetupWrapper(device_setup),
            vk,
        ))
    }
}
