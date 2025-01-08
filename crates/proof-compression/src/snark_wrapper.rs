use boojum::cs::{
    implementations::{proof::Proof, verifier::VerificationKey},
    oracle::TreeHasher,
};
use circuit_definitions::circuit_definitions::aux_layer::{
    compression::ProofCompressionFunction, wrapper::ZkSyncCompressionWrapper,
};

use super::*;

pub trait SnarkWrapperStep: SnarkWrapperProofSystem {
    const IS_PLONK: bool;
    const IS_FFLONK: bool;
    const PREVIOUS_COMPRESSION_MODE: u8;
    type PreviousStepTreeHasher: TreeHasher<
        GoldilocksField,
        Output: serde::Serialize + serde::de::DeserializeOwned,
    >;
    fn load_finalization_hint<AL>(
        artifact_loader: &AL,
    ) -> <Self as ProofSystemDefinition>::FinalizationHint
    where
        AL: ArtifactLoader,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let hint = if Self::IS_PLONK { &[26u8] } else { &[24] };
        serde_json::from_reader(&hint[..]).unwrap()
    }

    fn load_previous_vk<AL>(
        artifact_loader: &AL,
    ) -> VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>
    where
        AL: ArtifactLoader,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let previous_compression_mode = Self::PREVIOUS_COMPRESSION_MODE;
        let reader = artifact_loader.read_compression_wrapper_vk(previous_compression_mode);
        serde_json::from_reader(reader).unwrap()
    }

    fn load_this_vk<AL>(artifact_loader: &AL) -> <Self as ProofSystemDefinition>::VK
    where
        AL: ArtifactLoader,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let reader = if Self::IS_FFLONK {
            assert_eq!(Self::IS_PLONK, false);
            artifact_loader.read_fflonk_vk()
        } else {
            assert_eq!(Self::IS_PLONK, true);
            artifact_loader.read_plonk_vk()
        };

        serde_json::from_reader(reader).unwrap()
    }

    fn get_precomputation<AL>(
        artifact_loader: &AL,
    ) -> AsyncHandler<<Self as ProofSystemDefinition>::Precomputation>
    where
        AL: ArtifactLoader,
    {
        todo!()
    }

    fn prove_snark_wrapper_step<AL, CI>(
        input_proof: Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>,
        artifact_loader: &AL,
    ) -> <Self as ProofSystemDefinition>::Proof
    where
        AL: ArtifactLoader,
        CI: ContextInitializator,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let input_vk = Self::load_previous_vk(artifact_loader);
        let precomputation = Self::get_precomputation(artifact_loader);
        let config = <Self as ProofSystemDefinition>::get_context_config();
        let ctx = CI::init::<Self>(config);
        let finalization_hint = Self::load_finalization_hint(artifact_loader);
        let circuit = Self::build_circuit(input_vk, Some(input_proof));
        let proving_assembly = <Self as SnarkWrapperProofSystem>::synthesize_for_proving(circuit);
        <Self as SnarkWrapperProofSystem>::prove(
            ctx,
            proving_assembly,
            precomputation,
            finalization_hint,
        )
    }

    fn build_circuit(
        input_vk: VerificationKey<GoldilocksField, Self::PreviousStepTreeHasher>,
        input_proof: Option<Proof<GoldilocksField, Self::PreviousStepTreeHasher, GoldilocksExt2>>,
    ) -> Self::Circuit;
}

pub trait SnarkWrapperStepExt: SnarkWrapperProofSystemExt + SnarkWrapperStep {
    fn run_precomputation_for_compression<AL>(
        artifact_loader: &AL,
    ) -> (
        <Self as ProofSystemDefinition>::Precomputation,
        <Self as ProofSystemDefinition>::VK,
    )
    where
        AL: ArtifactLoader,
        <Self as ProofSystemDefinition>::VK: 'static,
    {
        let input_vk = Self::load_previous_vk(artifact_loader);
        let finalization_hint = Self::load_finalization_hint(artifact_loader);
        let circuit = Self::build_circuit(input_vk, None);
        let setup_assembly = <Self as SnarkWrapperProofSystemExt>::synthesize_for_setup(circuit);
        let data = <Self as SnarkWrapperProofSystemExt>::generate_precomputation_and_vk(
            setup_assembly,
            finalization_hint,
        );

        data.wait()
    }
}

pub struct FflonkSnarkWrapper;
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
pub struct PlonkSnarkWrapper;
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
