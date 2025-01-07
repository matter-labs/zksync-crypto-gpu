use circuit_definitions::circuit_definitions::aux_layer::{
    compression::ProofCompressionFunction,
    compression_modes::{
        CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
        CompressionMode4, CompressionMode5ForWrapper,
    },
};

use super::*;

pub struct CompressionChain<C: CompressionStep, U>(std::marker::PhantomData<(C, U)>);

impl<C, U> CompressionChain<C, U>
where
    C: CompressionStep,
{
    pub fn new(step: C) -> CompressionChain<C, U>
    where
        U: CompressionStep,
    {
        todo!()
    }

    pub fn next(self) -> U {
        todo!()
    }
}

impl<C, U> StepDefinition for CompressionChain<C, U>
where
    C: CompressionStep,
{
    type PreviousProofSystem = C::PreviousProofSystem;
    type ThisProofSystem = C::ThisProofSystem;
}

impl<C, U> ProofCompressionFunction for CompressionChain<C, U>
where
    C: CompressionStep,
{
    type PreviousLayerPoW = C::PreviousLayerPoW;

    type ThisLayerPoW = C::ThisLayerPoW;

    type ThisLayerHasher = C::ThisLayerHasher;

    type ThisLayerTranscript = C::ThisLayerTranscript;

    fn this_layer_transcript_parameters(
    ) -> <Self::ThisLayerTranscript as boojum::cs::implementations::transcript::Transcript<
        GoldilocksField,
    >>::TransciptParameters {
        C::this_layer_transcript_parameters()
    }

    fn description_for_compression_step() -> String {
        C::description_for_compression_step()
    }

    fn size_hint_for_compression_step() -> (usize, usize) {
        C::size_hint_for_compression_step()
    }

    fn geometry_for_compression_step() -> boojum::cs::CSGeometry {
        C::geometry_for_compression_step()
    }

    fn lookup_parameters_for_compression_step() -> boojum::cs::LookupParameters {
        C::lookup_parameters_for_compression_step()
    }

    fn configure_builder_for_compression_step<
        T: boojum::cs::cs_builder::CsBuilderImpl<GoldilocksField, T>,
        GC: boojum::cs::GateConfigurationHolder<GoldilocksField>,
        TB: boojum::cs::StaticToolboxHolder,
    >(
        builder: boojum::cs::cs_builder::CsBuilder<T, GoldilocksField, GC, TB>,
    ) -> boojum::cs::cs_builder::CsBuilder<
        T,
        GoldilocksField,
        impl boojum::cs::GateConfigurationHolder<GoldilocksField>,
        impl boojum::cs::StaticToolboxHolder,
    > {
        C::configure_builder_for_compression_step(builder)
    }

    fn previous_step_builder_for_compression<
        CS: boojum::cs::traits::cs::ConstraintSystem<GoldilocksField> + 'static,
    >() -> Box<
        dyn boojum::cs::traits::circuit::ErasedBuilderForRecursiveVerifier<
            GoldilocksField,
            GoldilocksExt2,
            CS,
        >,
    > {
        C::previous_step_builder_for_compression()
    }

    fn proof_config_for_compression_step() -> boojum::cs::implementations::prover::ProofConfig {
        C::proof_config_for_compression_step()
    }
}

impl<C, T> CompressionStep for CompressionChain<C, T>
where
    C: CompressionStep,
{
    const MODE: u8 = C::MODE;
    const IS_WRAPPER: bool = C::IS_WRAPPER;
}

pub struct CompressionChainBuilder<C: CompressionStep>(std::marker::PhantomData<C>);

impl<C> CompressionChainBuilder<C>
where
    C: CompressionStep,
{
    pub fn new(step: C) -> CompressionChainBuilder<C> {
        todo!()
    }
}

impl<C> CompressionChainBuilder<C>
where
    C: CompressionStep,
{
    pub fn link_compression_step<CC>(
        self,
        step: CC,
    ) -> CompressionChainBuilder<CompressionChain<C, CC>>
    where
        CC: CompressionStep,
    {
        todo!()
    }

    pub fn build(self) -> C {
        todo!()
    }
}

pub fn build_full_chain_with_fflonk() -> CompressionChain<
    CompressionChain<
        CompressionChain<CompressionChain<CompressionMode1, CompressionMode2>, CompressionMode3>,
        CompressionMode4,
    >,
    CompressionMode5ForWrapper,
> {
    let chain = CompressionChainBuilder::new(CompressionMode1)
        .link_compression_step(CompressionMode2)
        .link_compression_step(CompressionMode3)
        .link_compression_step(CompressionMode4)
        .link_compression_step(CompressionMode5ForWrapper)
        .build();

    chain
}

pub fn build_full_chain_with_plonk() -> CompressionChain<CompressionMode1ForWrapper, ()> {
    todo!()
}

pub fn run_step_chain_with_fflonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: BS,
) -> FflonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let chain = build_full_chain_with_fflonk();
    run_step_chain::<_, _, FflonkSnarkWrapper, _>(chain, input_proof, blob_storage)
}

pub fn run_step_chain_with_plonk<BS>(
    input_proof: SchedulerProof,
    blob_storage: BS,
) -> PlonkSnarkVerifierCircuitProof
where
    BS: BlobStorage,
{
    let chain = build_full_chain_with_plonk();
    todo!()
}

pub fn run_step_chain<C1, C2, SW, BS>(
    chain: CompressionChain<C1, C2>,
    input_proof: SchedulerProof,
    blob_storage: BS,
) -> <SW::ThisProofSystem as ProofSystemDefinition>::Proof
where
    C1: CompressionStep,
    C2: CompressionStep,
    SW: SnarkWrapperStep,
    BS: BlobStorage,
{
    let artifact_loader = SimpleArtifactLoader::init(blob_storage);

    let next_step = chain.next();
    todo!()
}
