use std::io::Read;

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

pub struct SnarkWrapperSetupData<T: SnarkWrapperStep> {
    pub precomputation: <T as ProofSystemDefinition>::Precomputation,
    pub vk: <T as ProofSystemDefinition>::VK,
    pub finalization_hint: <T as ProofSystemDefinition>::FinalizationHint,
    pub previous_vk: VerificationKey<GoldilocksField, T::PreviousStepTreeHasher>,
    pub ctx: Option<<T as SnarkWrapperProofSystem>::Context>,
}

pub trait SnarkWrapperStep: SnarkWrapperProofSystem {
    const IS_PLONK: bool;
    const IS_FFLONK: bool;
    const PREVIOUS_COMPRESSION_MODE: u8;
    type PreviousStepTreeHasher: TreeHasher<
        GoldilocksField,
        Output: serde::Serialize + serde::de::DeserializeOwned,
    >;
    fn load_finalization_hint() -> <Self as ProofSystemDefinition>::FinalizationHint {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let hint = if Self::IS_PLONK {
            (1 << <PlonkProverDeviceMemoryManagerConfig as gpu_prover::ManagerConfigs>::FULL_SLOT_SIZE_LOG).to_string()
        } else {
            (1 << ::fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG).to_string()
        };
        serde_json::from_str(&hint).unwrap()
    }

    fn load_previous_vk(
        reader: Box<dyn Read>,
    ) -> VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher> {
        serde_json::from_reader(reader).unwrap()
    }

    fn load_this_vk(reader: Box<dyn Read>) -> <Self as ProofSystemDefinition>::VK {
        serde_json::from_reader(reader).unwrap()
    }

    fn load_compact_raw_crs(reader: Box<dyn Read>) -> Self::CRS {
        let start = std::time::Instant::now();
        let compact_raw_crs = <Self as SnarkWrapperProofSystem>::load_compact_raw_crs(reader);
        println!(
            "Compact raw CRS loading takes {}s",
            start.elapsed().as_secs()
        );
        compact_raw_crs
    }

    fn get_precomputation(
        reader: Box<dyn Read>,
    ) -> <Self as ProofSystemDefinition>::Precomputation {
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

        precomputation
    }

    fn run_pre_initialization_tasks() {
        Self::pre_init();
    }

    fn prove_snark_wrapper_step(
        input_proof: Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
        setup_data_cache: &SnarkWrapperSetupData<Self>,
    ) -> <Self as ProofSystemDefinition>::Proof
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let input_vk = &setup_data_cache.previous_vk;

        let ctx = setup_data_cache.ctx.as_ref().unwrap();
        // let ctx = context_handler.init_snark_context::<Self>(ctx.clone());
        let finalization_hint = &setup_data_cache.finalization_hint;
        let circuit = Self::build_circuit(input_vk.clone(), Some(input_proof));
        let proving_assembly = <Self as SnarkWrapperProofSystem>::synthesize_for_proving(circuit);
        let vk = &setup_data_cache.vk;
        let precomputation = &setup_data_cache.precomputation;

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
    fn precompute_and_store_snark_wrapper_circuit(
        setup_data_cache: &SnarkWrapperSetupData<Self>,
    ) -> SnarkWrapperSetupData<Self>
    where
        <Self as ProofSystemDefinition>::VK: 'static,
    {
        let input_vk = setup_data_cache.previous_vk.clone();
        let finalization_hint = setup_data_cache.finalization_hint.clone();
        let circuit = Self::build_circuit(input_vk, None);
        let ctx = setup_data_cache.ctx.as_ref().unwrap();
        let setup_assembly = <Self as SnarkWrapperProofSystemExt>::synthesize_for_setup(circuit);

        let (precomputation, vk) =
            <Self as SnarkWrapperProofSystemExt>::generate_precomputation_and_vk(
                ctx,
                setup_assembly,
                finalization_hint,
            );

        let setup_data = SnarkWrapperSetupData {
            previous_vk: setup_data_cache.previous_vk.clone(),
            vk,
            precomputation,
            finalization_hint: setup_data_cache.finalization_hint.clone(),
            ctx: None,
        };

        setup_data

        // let (precompuatation_writer, vk_writer) = if Self::IS_FFLONK {
        //     (
        //         blob_storage.write_fflonk_precomputation(),
        //         blob_storage.write_fflonk_vk(),
        //     )
        // } else {
        //     (
        //         blob_storage.write_plonk_precomputation(),
        //         blob_storage.write_plonk_vk(),
        //     )
        // };
        // precomputation
        //     .write_into_buffer(precompuatation_writer)
        //     .unwrap();
        // serde_json::to_writer_pretty(vk_writer, &vk).unwrap();
        // println!("Pecomputation and vk of snark wrapper circuit saved into blob storage");
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
