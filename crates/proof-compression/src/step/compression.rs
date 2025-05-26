use std::{io::Read, sync::Arc};

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
    pub precomputation: Option<<T as ProofSystemDefinition>::Precomputation>,
    pub vk: Option<<T as ProofSystemDefinition>::VK>,
    pub finalization_hint: Option<<T as ProofSystemDefinition>::FinalizationHint>,
    pub previous_vk: Option<VerificationKey<GoldilocksField, T::PreviousStepTreeHasher>>,
}

impl<T: CompressionStep> CompressionSetupData<T> {
    pub fn new() -> Self {
        Self {
            precomputation: None,
            vk: None,
            finalization_hint: None,
            previous_vk: None,
        }
    }
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
    ) -> <Self as ProofSystemDefinition>::FinalizationHint {
        serde_json::from_reader(reader).unwrap()
    }

    fn load_previous_vk(
        reader: Box<dyn Read>,
    ) -> VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher> {
        serde_json::from_reader(reader).unwrap()
    }

    fn load_this_vk(reader: Box<dyn Read>) -> <Self as ProofSystemDefinition>::VK {
        serde_json::from_reader(reader).unwrap()
    }

    fn get_precomputation(
        reader: Box<dyn Read>,
    ) -> <Self as ProofSystemDefinition>::Precomputation {
        let start = std::time::Instant::now();
        let precomputation =
            <<Self as ProofSystemDefinition>::Precomputation>::read_from_buffer(reader).unwrap();
        println!(
            "Compression device setup loading takes {}s",
            start.elapsed().as_secs()
        );
        precomputation
    }

    fn prove_compression_step<CI>(
        input_proof: Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
        setup_data_cache: &CompressionSetupData<Self>,
        context_handler: &CI,
    ) -> <Self as ProofSystemDefinition>::Proof
    where
        CI: ContextManagerInterface,
    {
        let input_vk = setup_data_cache.previous_vk.as_ref().unwrap();
        let vk = setup_data_cache.vk.as_ref().unwrap();
        let precomputation = setup_data_cache.precomputation.as_ref().unwrap();
        let finalization_hint = setup_data_cache.finalization_hint.as_ref().unwrap();
        let ctx_config = Self::get_context_config_from_hint(&finalization_hint);
        let ctx = context_handler.init_compression_context::<Self>(ctx_config);
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
            &precomputation,
            &finalization_hint,
            &vk,
        );
        assert!(<Self as ProofSystemDefinition>::verify(&proof, &vk));

        proof
    }

    // CompressionLayerCircuit is unified type for both compression circuits
    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> CompressionLayerCircuit<Self>;
}

pub(crate) trait CompressionStepExt: CompressionProofSystemExt + CompressionStep {
    fn precompute_and_store_compression_circuits<CM>(
        setup_data_cache: CompressionSetupData<Self>,
        context_manager: &CM,
    ) -> CompressionSetupData<Self>
    where
        CM: ContextManagerInterface,
    {
        let input_vk = setup_data_cache.previous_vk.clone().unwrap();
        let circuit = Self::build_circuit(input_vk, None);
        // Workaround: trace length is not known at this point, so thats totally fine
        // to use a hardcoded trace length
        let ctx_config = Self::get_context_config();
        let ctx = context_manager.init_compression_context::<Self>(ctx_config);
        let (finalization_hint, setup_assembly) =
            <Self as CompressionProofSystemExt>::synthesize_for_setup(circuit);
        let (precomputation, vk) =
            <Self as CompressionProofSystemExt>::generate_precomputation_and_vk(
                ctx,
                setup_assembly,
                &finalization_hint,
            );

        CompressionSetupData {
            precomputation: Some(precomputation),
            vk: Some(vk),
            finalization_hint: Some(finalization_hint),
            previous_vk: setup_data_cache.previous_vk,
        }

        // let (precompuatation_writer, vk_writer, hint_writer) = if Self::IS_WRAPPER {
        //     (
        //         blob_storage.write_compression_wrapper_precomputation(Self::MODE),
        //         blob_storage.write_compression_wrapper_vk(Self::MODE),
        //         blob_storage.write_compression_wrapper_finalization_hint(Self::MODE),
        //     )
        // } else {
        //     (
        //         blob_storage.write_compression_layer_precomputation(Self::MODE),
        //         blob_storage.write_compression_layer_vk(Self::MODE),
        //         blob_storage.write_compression_layer_finalization_hint(Self::MODE),
        //     )
        // };
        // precomputation
        //     .write_into_buffer(precompuatation_writer)
        //     .unwrap();
        // serde_json::to_writer_pretty(vk_writer, &vk).unwrap();
        // serde_json::to_writer_pretty(hint_writer, &finalization_hint).unwrap();
        // println!(
        //     "Precomputation and vk of compression circuit {} saved into blob storage",
        //     Self::MODE
        // );
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
