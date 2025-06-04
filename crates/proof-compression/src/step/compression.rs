use std::io::{Read, Write};

use super::*;

use circuit_definitions::circuit_definitions::{
    aux_layer::{
        compression::{CompressionLayerCircuit, ProofCompressionFunction},
        compression_modes::{
            CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
            CompressionMode4, CompressionMode5ForWrapper,
        },
        ZkSyncCompressionForWrapperCircuit, ZkSyncCompressionLayerCircuit,
    },
    recursion_layer::RecursiveProofsTreeHasher,
};
use franklin_crypto::boojum::cs::{
    implementations::{
        fast_serialization::MemcopySerializable, proof::Proof, verifier::VerificationKey,
    },
    oracle::TreeHasher,
};

pub struct CompressionSetupData<T: CompressionStep> {
    pub precomputation: <T as ProofSystemDefinition>::Precomputation,
    pub vk: <T as ProofSystemDefinition>::VK,
    pub finalization_hint: <T as ProofSystemDefinition>::FinalizationHint,
    pub previous_vk: VerificationKey<GoldilocksField, T::PreviousStepTreeHasher>,
}

pub trait CompressionStep: CompressionProofSystem {
    type PreviousStepTreeHasher: TreeHasher<
        GoldilocksField,
        Output: serde::Serialize + serde::de::DeserializeOwned,
    >;

    const MODE: u8;
    const IS_WRAPPER: bool;
    fn load_finalization_hint(
        reader: Box<dyn Read>,
    ) -> anyhow::Result<<Self as ProofSystemDefinition>::FinalizationHint> {
        Ok(serde_json::from_reader(reader)?)
    }

    fn load_previous_vk(
        reader: Box<dyn Read>,
    ) -> anyhow::Result<VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>> {
        Ok(serde_json::from_reader(reader)?)
    }

    fn load_this_vk(reader: Box<dyn Read>) -> anyhow::Result<<Self as ProofSystemDefinition>::VK> {
        Ok(serde_json::from_reader(reader)?)
    }

    fn get_precomputation(
        reader: Box<dyn Read>,
    ) -> anyhow::Result<<Self as ProofSystemDefinition>::Precomputation> {
        Ok(<<Self as ProofSystemDefinition>::Precomputation as MemcopySerializable>::read_from_buffer(
            reader,
        ).map_err(|e| {
            anyhow::anyhow!("Failed to read precomputation: {}", e)
        })?)
    }

    fn prove_compression_step(
        input_proof: Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
        setup_data_cache: &CompressionSetupData<Self>,
    ) -> anyhow::Result<<Self as ProofSystemDefinition>::Proof> {
        let input_vk = &setup_data_cache.previous_vk;
        let vk = &setup_data_cache.vk;
        let precomputation = &setup_data_cache.precomputation;
        let finalization_hint = &setup_data_cache.finalization_hint;
        let ctx_config = Self::get_context_config_from_hint(&finalization_hint);
        let ctx = Self::init_context(ctx_config)?;
        let circuit = Self::build_circuit(input_vk.clone(), Some(input_proof));
        let proving_assembly = <Self as CompressionProofSystem>::synthesize_for_proving(
            circuit,
            finalization_hint.clone(),
        );
        let aux_config =
            <Self as CompressionProofSystem>::aux_config_from_assembly(&proving_assembly);
        let proof = <Self as CompressionProofSystem>::prove(
            ctx,
            proving_assembly,
            aux_config,
            precomputation,
            &finalization_hint,
            &vk,
        )?;
        anyhow::ensure!(<Self as ProofSystemDefinition>::verify(&proof, &vk));

        Ok(proof)
    }

    // CompressionLayerCircuit is unified type for both compression circuits
    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> CompressionLayerCircuit<Self>;
}

pub trait CompressionStepExt: CompressionProofSystemExt + CompressionStep {
    fn store_precomputation(
        precomputation: &<Self as ProofSystemDefinition>::Precomputation,
        writer: Box<dyn Write>,
    ) -> anyhow::Result<()> {
        precomputation
            .write_into_buffer(writer)
            .map_err(|e| anyhow::anyhow!("Failed to write precomputation: {}", e))?;
        Ok(())
    }

    fn store_vk(
        vk: &<Self as ProofSystemDefinition>::VK,
        writer: Box<dyn Write>,
    ) -> anyhow::Result<()> {
        serde_json::to_writer_pretty(writer, vk)?;
        Ok(())
    }

    fn store_finalization_hint(
        finalization_hint: &<Self as ProofSystemDefinition>::FinalizationHint,
        writer: Box<dyn Write>,
    ) -> anyhow::Result<()> {
        serde_json::to_writer_pretty(writer, finalization_hint)?;
        Ok(())
    }

    fn precompute_compression_circuits(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
    ) -> anyhow::Result<(
        <Self as ProofSystemDefinition>::Precomputation,
        <Self as ProofSystemDefinition>::VK,
        <Self as ProofSystemDefinition>::FinalizationHint,
    )> {
        let circuit = Self::build_circuit(input_vk.clone(), None);
        // Workaround: trace length is not known at this point, so thats totally fine
        // to use a hardcoded trace length
        let ctx_config = Self::get_context_config();
        let ctx = Self::init_context(ctx_config)?;
        let (finalization_hint, setup_assembly) =
            <Self as CompressionProofSystemExt>::synthesize_for_setup(circuit);
        let (precomputation, vk) =
            <Self as CompressionProofSystemExt>::generate_precomputation_and_vk(
                ctx,
                setup_assembly,
                &finalization_hint,
            )?;

        Ok((precomputation, vk, finalization_hint))
    }
}

macro_rules! impl_compression_circuit {
    ($type:ty,$mode:expr, $is_wrapper:expr, $enum:ident::$variant:ident, $hasher:ty) => {
        impl CompressionStep for $type {
            const MODE: u8 = $mode;
            const IS_WRAPPER: bool = $is_wrapper;
            type PreviousStepTreeHasher = $hasher;
            fn build_circuit(
                input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
                input_proof: Option<
                    Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
                >,
            ) -> CompressionLayerCircuit<Self> {
                let circuit = $enum::from_witness_and_vk(input_proof, input_vk, Self::MODE);
                match circuit {
                    $enum::$variant(compression_layer_circuit) => compression_layer_circuit,
                    _ => unreachable!(),
                }
            }
        }

        impl CompressionStepExt for $type {}
    };
}

impl_compression_circuit!(
    CompressionMode1,
    1,
    false,
    ZkSyncCompressionLayerCircuit::CompressionMode1Circuit,
    RecursiveProofsTreeHasher
);
impl_compression_circuit!(
    CompressionMode2,
    2,
    false,
    ZkSyncCompressionLayerCircuit::CompressionMode2Circuit,
    <CompressionMode1 as ProofCompressionFunction>::ThisLayerHasher
);

impl_compression_circuit!(
    CompressionMode3,
    3,
    false,
    ZkSyncCompressionLayerCircuit::CompressionMode3Circuit,
    <CompressionMode2 as ProofCompressionFunction>::ThisLayerHasher
);

impl_compression_circuit!(
    CompressionMode4,
    4,
    false,
    ZkSyncCompressionLayerCircuit::CompressionMode4Circuit,
    <CompressionMode3 as ProofCompressionFunction>::ThisLayerHasher
);

impl_compression_circuit!(
    CompressionMode1ForWrapper,
    1,
    true,
    ZkSyncCompressionForWrapperCircuit::CompressionMode1Circuit,
    RecursiveProofsTreeHasher
);

impl_compression_circuit!(
    CompressionMode5ForWrapper,
    5,
    true,
    ZkSyncCompressionForWrapperCircuit::CompressionMode5Circuit,
    <CompressionMode4 as ProofCompressionFunction>::ThisLayerHasher
);
