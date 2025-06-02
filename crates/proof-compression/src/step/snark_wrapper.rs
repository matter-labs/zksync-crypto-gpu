use std::io::{Read, Write};

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
    pub ctx: <T as SnarkWrapperProofSystem>::Context,
}

pub trait SnarkWrapperStep: SnarkWrapperProofSystem {
    const IS_PLONK: bool;
    const IS_FFLONK: bool;
    const PREVIOUS_COMPRESSION_MODE: u8;
    type PreviousStepTreeHasher: TreeHasher<
        GoldilocksField,
        Output: serde::Serialize + serde::de::DeserializeOwned,
    >;
    fn load_finalization_hint() -> anyhow::Result<<Self as ProofSystemDefinition>::FinalizationHint> {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let hint = if Self::IS_PLONK {
            (1 << <PlonkProverDeviceMemoryManagerConfig as gpu_prover::ManagerConfigs>::FULL_SLOT_SIZE_LOG).to_string()
        } else {
            (1 << ::fflonk::fflonk::L1_VERIFIER_DOMAIN_SIZE_LOG).to_string()
        };
        Ok(serde_json::from_str(&hint)?)
    }

    fn load_previous_vk(
        reader: Box<dyn Read>,
    ) -> anyhow::Result<VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>> {
        Ok(serde_json::from_reader(reader)?)
    }

    fn load_this_vk(reader: Box<dyn Read>) -> anyhow::Result<<Self as ProofSystemDefinition>::VK> {
        Ok(serde_json::from_reader(reader)?)
    }

    fn load_compact_raw_crs(reader: Box<dyn Read>) -> anyhow::Result<<Self as SnarkWrapperProofSystem>::CRS> {
        <Self as SnarkWrapperProofSystem>::load_compact_raw_crs(reader)
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

    fn run_pre_initialization_tasks() {
        Self::pre_init();
    }

    fn prove_snark_wrapper_step(
        input_proof: Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
        setup_data_cache: &SnarkWrapperSetupData<Self>,
    ) -> anyhow::Result<<Self as ProofSystemDefinition>::Proof> {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let input_vk = &setup_data_cache.previous_vk;
        let ctx = &setup_data_cache.ctx;
        let finalization_hint = &setup_data_cache.finalization_hint;
        let circuit = Self::build_circuit(input_vk.clone(), Some(input_proof));
        let proving_assembly = <Self as SnarkWrapperProofSystem>::synthesize_for_proving(circuit);
        let vk = &setup_data_cache.vk;
        let precomputation = &setup_data_cache.precomputation;

        let proof = <Self as SnarkWrapperProofSystem>::prove(
            &ctx,
            proving_assembly,
            precomputation,
            finalization_hint,
        )?;

        assert!(<Self as ProofSystemDefinition>::verify(&proof, &vk));

        Ok(proof)
    }

    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> Self::Circuit;
}

pub trait SnarkWrapperStepExt: SnarkWrapperProofSystemExt + SnarkWrapperStep {
    fn store_precomputation(
        precomputation: &<Self as ProofSystemDefinition>::Precomputation,
        writer: Box<dyn Write>,
    ) -> anyhow::Result<()> {
        <Self as ProofSystemDefinition>::Precomputation::write_into_buffer(precomputation, writer)
            .map_err(|e| anyhow::anyhow!("Failed to write precomputation: {}", e))?;
        Ok(())
    }

    fn store_vk(vk: &<Self as ProofSystemDefinition>::VK, writer: Box<dyn Write>) -> anyhow::Result<()> {
        serde_json::to_writer_pretty(writer, vk)?;
        Ok(())
    }

    fn precompute_snark_wrapper_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        finalization_hint: <Self as ProofSystemDefinition>::FinalizationHint,
        ctx: <Self as SnarkWrapperProofSystem>::Context,
    ) -> anyhow::Result<(
        <Self as ProofSystemDefinition>::Precomputation,
        <Self as ProofSystemDefinition>::VK,
    )> {
        let circuit = Self::build_circuit(input_vk, None);
        let setup_assembly = <Self as SnarkWrapperProofSystemExt>::synthesize_for_setup(circuit);

        let (precomputation, vk) =
            <Self as SnarkWrapperProofSystemExt>::generate_precomputation_and_vk(
                &ctx,
                setup_assembly,
                finalization_hint,
            )?;

        Ok((precomputation, vk))
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
