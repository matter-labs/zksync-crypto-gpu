use anyhow::Context;
use shivini::{
    boojum::{
        config::{ProvingCSConfig, SetupCSConfig},
        cs::{
            implementations::{
                proof::Proof, reference_cs::CSReferenceAssembly, setup::FinalizationHintsForProver,
                transcript::Transcript, verifier::VerificationKey, witness::WitnessVec,
            },
            traits::{circuit::CircuitBuilderProxy, GoodAllocator},
        },
        worker::Worker,
    },
    gpu_proof_config::GpuProofConfig,
    CacheStrategy, CommitmentCacheStrategy, GPUPoWRunner, GpuTreeHasher, PolynomialsCacheStrategy,
    ProverContext, ProverContextConfig,
};

use circuit_definitions::circuit_definitions::aux_layer::compression::{
    CompressionLayerCircuit, ProofCompressionFunction,
};

use super::*;
pub(crate) use shivini::boojum;
pub(crate) use shivini::boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};

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
    type Precomputation = BoojumDeviceSetupWrapper<CF::ThisLayerHasher>;
    type Proof = Proof<Self::FieldElement, CF::ThisLayerHasher, GoldilocksExt2>;
    type VK = VerificationKey<Self::FieldElement, CF::ThisLayerHasher>;
    type FinalizationHint = FinalizationHintsForProver;
    type Allocator = std::alloc::Global;
    type ProvingAssembly = BoojumAssembly<ProvingCSConfig, Self::Allocator>;
    type Transcript = CF::ThisLayerTranscript;
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

    fn init_context(domain_size: Self::ContextConfig) -> anyhow::Result<Self::Context> {
        let config =
            ProverContextConfig::default().with_smallest_supported_domain_size(domain_size);
        let context = Self::Context::create_with_config(config).context("gpu prover context");

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
        ctx: Self::Context,
        proving_assembly: Self::ProvingAssembly,
        aux_config: Self::AuxConfig,
        precomputation: &Self::Precomputation,
        finalization_hint: &Self::FinalizationHint,
        vk: &Self::VK,
    ) -> anyhow::Result<Self::Proof> {
        Self::prove_from_witnesses(
            ctx,
            proving_assembly.witness.context("missing witness")?,
            aux_config,
            precomputation,
            finalization_hint,
            vk,
        )
    }

    fn prove_from_witnesses(
        ctx: Self::Context,
        witness: Self::ExternalWitnessData,
        aux_config: Self::AuxConfig,
        precomputation: &Self::Precomputation,
        finalization_hint: &Self::FinalizationHint,
        vk: &Self::VK,
    ) -> anyhow::Result<Self::Proof> {
        let domain_size = vk.fixed_parameters.domain_size as usize;
        assert_eq!(finalization_hint.final_trace_len, domain_size);
        let cache_strategy = CacheStrategy {
            setup_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            trace_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            other_polynomials: PolynomialsCacheStrategy::CacheMonomialsAndFirstCoset,
            commitment: CommitmentCacheStrategy::CacheCosetCaps,
        };
        let worker = Worker::new();
        let precomputation = precomputation.into_inner_ref();
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
        .context("failed to generate gpu compression proof")?;
        drop(ctx);
        let proof = gpu_proof.into();

        Ok(proof)
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
        ctx: Self::Context,
        setup_assembly: Self::SetupAssembly,
        finalization_hint: &Self::FinalizationHint,
    ) -> anyhow::Result<(Self::Precomputation, Self::VK)> {
        let worker = Worker::new();
        let proof_config = CF::proof_config_for_compression_step();
        let (setup_base, vk_params, vars_hint, wits_hint) = setup_assembly.get_light_setup(
            &worker,
            proof_config.fri_lde_factor,
            proof_config.merkle_tree_cap_size,
        );
        let domain_size = vk_params.domain_size as usize;
        assert_eq!(finalization_hint.final_trace_len, domain_size);
        let (precomputation, vk) =
            shivini::cs::gpu_setup_and_vk_from_base_setup_vk_params_and_hints::<
                CF::ThisLayerHasher,
                _,
            >(setup_base, vk_params, vars_hint, wits_hint, &worker)
            .context("failed to generate precomputation and vk for compression")?;
        drop(ctx);
        Ok((BoojumDeviceSetupWrapper::from_inner(precomputation), vk))
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
