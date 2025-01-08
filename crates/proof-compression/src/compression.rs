use super::*;

use boojum::cs::{
    implementations::{
        fast_serialization::MemcopySerializable, proof::Proof, verifier::VerificationKey,
    },
    oracle::TreeHasher,
};
use circuit_definitions::circuit_definitions::{
    aux_layer::{
        compression::{CompressionLayerCircuit, ProofCompressionFunction},
        ZkSyncCompressionForWrapperCircuit, ZkSyncCompressionLayerCircuit,
    },
    recursion_layer::RecursiveProofsTreeHasher,
};

pub trait CompressionStep: CompressionProofSystem {
    type PreviousStepTreeHasher: TreeHasher<
        GoldilocksField,
        Output: serde::Serialize + serde::de::DeserializeOwned,
    >;

    const MODE: u8;
    const IS_WRAPPER: bool;
    fn load_finalization_hint<AL>(
        artifact_loader: &AL,
    ) -> <Self as ProofSystemDefinition>::FinalizationHint
    where
        AL: ArtifactLoader,
    {
        let reader = if Self::IS_WRAPPER {
            artifact_loader.read_compression_wrapper_finalization_hint(Self::MODE)
        } else {
            artifact_loader.read_compression_layer_finalization_hint(Self::MODE)
        };
        serde_json::from_reader(reader).unwrap()
    }

    fn load_previous_vk<AL>(
        artifact_loader: &AL,
    ) -> VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>
    where
        AL: ArtifactLoader,
    {
        assert!(Self::MODE >= 1);

        let reader = if Self::MODE == 1 {
            artifact_loader.read_scheduler_vk()
        } else if Self::IS_WRAPPER {
            artifact_loader.read_compression_wrapper_vk(Self::MODE)
        } else {
            artifact_loader.read_compression_layer_vk(Self::MODE - 1)
        };

        serde_json::from_reader(reader).unwrap()
    }

    fn load_this_vk<AL>(artifact_loader: &AL) -> <Self as ProofSystemDefinition>::VK
    where
        AL: ArtifactLoader,
    {
        let reader = if Self::IS_WRAPPER {
            artifact_loader.read_compression_wrapper_vk(Self::MODE)
        } else {
            artifact_loader.read_compression_layer_vk(Self::MODE)
        };

        serde_json::from_reader(reader).unwrap()
    }

    fn get_precomputation<AL>(
        artifact_loader: &AL,
    ) -> AsyncHandler<<Self as ProofSystemDefinition>::Precomputation>
    where
        AL: ArtifactLoader,
    {
        // TODO
        let reader = if Self::IS_WRAPPER {
            artifact_loader.get_compression_layer_precomputation(Self::MODE)
        } else {
            artifact_loader.get_compression_layer_precomputation(Self::MODE)
        };
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();

            let precomputation =
                <<Self as ProofSystemDefinition>::Precomputation as MemcopySerializable>::read_from_buffer(
                    reader,
                )
                .unwrap();

            sender.send(precomputation).unwrap();
            receiver
        };

        AsyncHandler::spawn(f)
    }

    fn prove_compression_step<AL, CI>(
        input_proof: Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
        artifact_loader: &AL,
    ) -> <Self as ProofSystemDefinition>::Proof
    where
        AL: ArtifactLoader,
        CI: ContextInitializator,
    {
        let input_vk = Self::load_previous_vk(artifact_loader);
        let vk = Self::load_this_vk(artifact_loader);
        let precomputation = Self::get_precomputation(artifact_loader);
        let config = <Self as ProofSystemDefinition>::get_context_config();
        let ctx = CI::init::<Self>(config);
        let finalization_hint = Self::load_finalization_hint(artifact_loader);
        let circuit = Self::build_circuit(input_vk, Some(input_proof));
        let proving_assembly = <Self as CompressionProofSystem>::synthesize_for_proving(
            circuit,
            finalization_hint.clone(),
        );
        let aux_config =
            <Self as CompressionProofSystem>::aux_config_from_assembly(&proving_assembly);
        <Self as CompressionProofSystem>::prove(
            ctx,
            proving_assembly,
            aux_config,
            precomputation,
            finalization_hint,
            vk,
        )
    }

    // CompressionLayerCircuit is unified type for both compression circuits
    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> CompressionLayerCircuit<Self>;
}

pub trait CompressionStepExt: CompressionProofSystemExt + CompressionStep {
    fn run_precomputation_for_compression<AL>(
        artifact_loader: &AL,
    ) -> (
        <Self as ProofSystemDefinition>::Precomputation,
        <Self as ProofSystemDefinition>::VK,
    )
    where
        AL: ArtifactLoader,
    {
        let input_vk = Self::load_previous_vk(artifact_loader);
        let circuit = Self::build_circuit(input_vk, None);
        let (finalization_hint, setup_assembly) =
            <Self as CompressionProofSystemExt>::synthesize_for_setup(circuit);
        let data = <Self as CompressionProofSystemExt>::generate_precomputation_and_vk(
            setup_assembly,
            finalization_hint,
        );
        data.wait()
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
