use circuit_definitions::circuit_definitions::aux_layer::{
    compression::ProofCompressionFunction,
    compression_modes::{CompressionMode1ForWrapper, CompressionMode5ForWrapper},
    wrapper::ZkSyncCompressionWrapper,
};
use franklin_crypto::boojum::cs::{
    implementations::{
        fast_serialization::MemcopySerializable, proof::Proof, verifier::VerificationKey,
    },
    oracle::TreeHasher,
};

use super::*;

pub(crate) trait SnarkWrapperStep: SnarkWrapperProofSystem {
    const IS_PLONK: bool;
    const IS_FFLONK: bool;
    const PREVIOUS_COMPRESSION_MODE: u8;
    type PreviousStepTreeHasher: TreeHasher<
        GoldilocksField,
        Output: serde::Serialize + serde::de::DeserializeOwned,
    >;
    fn load_finalization_hint<BS>(
        _blob_storage: &BS,
    ) -> <Self as ProofSystemDefinition>::FinalizationHint
    where
        BS: BlobStorage,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let hint = if Self::IS_PLONK {
            (1 << <PlonkProverDeviceMemoryManagerConfig as gpu_prover::ManagerConfigs>::FULL_SLOT_SIZE_LOG).to_string()
        } else {
            (1 << fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG).to_string()
        };
        serde_json::from_str(&hint).unwrap()
    }

    fn load_previous_vk<BS>(
        blob_storage: &BS,
    ) -> VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>
    where
        BS: BlobStorage,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let previous_compression_mode = Self::PREVIOUS_COMPRESSION_MODE;
        let reader = blob_storage.read_compression_wrapper_vk(previous_compression_mode);
        serde_json::from_reader(reader).unwrap()
    }

    fn load_this_vk<BS>(blob_storage: &BS) -> <Self as ProofSystemDefinition>::VK
    where
        BS: BlobStorage,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let reader = if Self::IS_FFLONK {
            assert_eq!(Self::IS_PLONK, false);
            blob_storage.read_fflonk_vk()
        } else {
            assert_eq!(Self::IS_PLONK, true);
            blob_storage.read_plonk_vk()
        };

        serde_json::from_reader(reader).unwrap()
    }

    fn load_compact_raw_crs<BS>(blob_storage: &BS) -> AsyncHandler<Self::CRS>
    where
        BS: BlobStorage,
        // Self::Allocator: Send + Sync + 'static,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let reader = blob_storage.read_compact_raw_crs();
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let start = std::time::Instant::now();
            let compact_raw_crs = <Self as SnarkWrapperProofSystem>::load_compact_raw_crs(reader);
            println!(
                "Compact raw CRS loading takes {}s",
                start.elapsed().as_secs()
            );
            sender.send(compact_raw_crs).unwrap();
            receiver
        };

        AsyncHandler::spawn(f)
    }

    fn get_precomputation<BS>(
        blob_storage: &BS,
    ) -> AsyncHandler<<Self as ProofSystemDefinition>::Precomputation>
    where
        BS: BlobStorage,
    {
        let reader = if Self::IS_FFLONK {
            blob_storage.read_fflonk_precomputation()
        } else {
            blob_storage.read_plonk_precomputation()
        };
        let f = move || {
            let (sender, receiver) = std::sync::mpsc::channel();
            let start = std::time::Instant::now();
            let precomputation =
                <<Self as ProofSystemDefinition>::Precomputation as MemcopySerializable>::read_from_buffer(
                    reader,
                )
                .unwrap();
            println!(
                "Snark wrapper device setup loading takes {}s",
                start.elapsed().as_secs()
            );
            sender.send(precomputation).unwrap();
            receiver
        };

        AsyncHandler::spawn(f)
    }

    fn run_pre_initialization_tasks() {
        Self::pre_init();
    }

    fn prove_snark_wrapper_step<BS, CI>(
        ctx_config: AsyncHandler<Self::CRS>,
        precomputation: AsyncHandler<Self::Precomputation>,
        input_proof: Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
        blob_storage: &BS,
        context_handler: &CI,
    ) -> <Self as ProofSystemDefinition>::Proof
    where
        BS: BlobStorage,
        CI: ContextManagerInterface,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let input_vk = Self::load_previous_vk(blob_storage);

        let ctx = context_handler.init_snark_context::<Self>(ctx_config);
        let finalization_hint = Self::load_finalization_hint(blob_storage);
        let circuit = Self::build_circuit(input_vk, Some(input_proof));
        let proving_assembly = <Self as SnarkWrapperProofSystem>::synthesize_for_proving(circuit);
        let vk = Self::load_this_vk(blob_storage);

        let proof = <Self as SnarkWrapperProofSystem>::prove(
            ctx,
            proving_assembly,
            precomputation,
            finalization_hint,
        );

        assert!(<Self as ProofSystemDefinition>::verify(&proof, &vk));

        proof
    }

    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> Self::Circuit;
}

pub(crate) trait SnarkWrapperStepExt: SnarkWrapperProofSystemExt + SnarkWrapperStep {
    fn precompute_and_store_snark_wrapper_circuit<BS, CM>(
        compact_raw_crs: AsyncHandler<Self::CRS>,
        blob_storage: &BS,
        context_manager: &CM,
    ) where
        BS: BlobStorageExt,
        CM: ContextManagerInterface,
        <Self as ProofSystemDefinition>::VK: 'static,
    {
        let input_vk = Self::load_previous_vk(blob_storage);
        let finalization_hint = Self::load_finalization_hint(blob_storage);
        let circuit = Self::build_circuit(input_vk, None);
        let ctx = context_manager.init_snark_context::<Self>(compact_raw_crs);
        let setup_assembly = <Self as SnarkWrapperProofSystemExt>::synthesize_for_setup(circuit);

        let (precomputation, vk) =
            <Self as SnarkWrapperProofSystemExt>::generate_precomputation_and_vk(
                ctx,
                setup_assembly,
                finalization_hint,
            );
        let (precompuatation_writer, vk_writer) = if Self::IS_FFLONK {
            (
                blob_storage.write_fflonk_precomputation(),
                blob_storage.write_fflonk_vk(),
            )
        } else {
            (
                blob_storage.write_plonk_precomputation(),
                blob_storage.write_plonk_vk(),
            )
        };
        precomputation
            .write_into_buffer(precompuatation_writer)
            .unwrap();
        serde_json::to_writer_pretty(vk_writer, &vk).unwrap();
        println!("Pecomputation and vk of snark wrapper circuit saved into blob storage");
    }
}

impl SnarkWrapperStep for FflonkSnarkWrapper {
    const IS_PLONK: bool = false;
    const IS_FFLONK: bool = true;
    const PREVIOUS_COMPRESSION_MODE: u8 = 5;
    type PreviousStepTreeHasher =
        <CompressionMode5ForWrapper as ProofCompressionFunction>::ThisLayerHasher;
    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> Self::Circuit {
        let fixed_parameters = input_vk.fixed_parameters.clone();
        FflonkSnarkVerifierCircuit {
            witness: input_proof,
            vk: input_vk,
            fixed_parameters,
            transcript_params: (),
            wrapper_function: ZkSyncCompressionWrapper::from_numeric_circuit_type(
                Self::PREVIOUS_COMPRESSION_MODE,
            ),
        }
    }
}
impl SnarkWrapperStepExt for FflonkSnarkWrapper {}

impl SnarkWrapperStep for PlonkSnarkWrapper {
    const IS_PLONK: bool = true;
    const IS_FFLONK: bool = false;

    const PREVIOUS_COMPRESSION_MODE: u8 = 1;
    type PreviousStepTreeHasher =
        <CompressionMode1ForWrapper as ProofCompressionFunction>::ThisLayerHasher;
    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> Self::Circuit {
        let fixed_parameters = input_vk.fixed_parameters.clone();
        PlonkSnarkVerifierCircuit {
            witness: input_proof,
            vk: input_vk,
            fixed_parameters,
            transcript_params: (),
            wrapper_function: ZkSyncCompressionWrapper::from_numeric_circuit_type(
                Self::PREVIOUS_COMPRESSION_MODE,
            ),
        }
    }
}
impl SnarkWrapperStepExt for PlonkSnarkWrapper {}
