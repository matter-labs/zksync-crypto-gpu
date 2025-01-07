use circuit_definitions::circuit_definitions::aux_layer::compression_modes::CompressionTreeHasherForWrapper;

use super::*;

pub trait SnarkWrapperStep: StepDefinition<ThisProofSystem: SnarkWrapperProofSystem> {
    const IS_PLONK: bool;
    const IS_FFLONK: bool;
    const LAST_COMPRESSION_MODE: u8;

    fn load_finalization_hint<AL>(
        artifact_loader: &AL,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::FinalizationHint
    where
        AL: ArtifactLoader,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let hint = if Self::IS_PLONK { &[26u8] } else { &[24] };
        serde_json::from_reader(&hint[..]).unwrap()
    }

    fn load_previous_vk<AL>(
        artifact_loader: &AL,
    ) -> <Self::PreviousProofSystem as ProofSystemDefinition>::VK
    where
        AL: ArtifactLoader,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let reader = artifact_loader.read_compression_wrapper_vk(Self::LAST_COMPRESSION_MODE);
        serde_json::from_reader(reader).unwrap()
    }

    fn load_this_vk<AL>(
        artifact_loader: &AL,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::VK
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
    ) -> AsyncHandler<<Self::ThisProofSystem as ProofSystemDefinition>::Precomputation>
    where
        AL: ArtifactLoader,
    {
        todo!()
    }

    fn prove_snark_wrapper_step<AL, CI>(
        input_proof: <Self::PreviousProofSystem as ProofSystemDefinition>::Proof,
        artifact_loader: &AL,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::Proof
    where
        AL: ArtifactLoader,
        CI: ContextInitializator,
    {
        assert!(Self::IS_FFLONK ^ Self::IS_PLONK);
        let input_vk = Self::load_previous_vk(artifact_loader);
        let precomputation = Self::get_precomputation(artifact_loader);
        let config = <Self::ThisProofSystem as ProofSystemDefinition>::get_context_config();
        let ctx = CI::init::<Self::ThisProofSystem>(config);
        let finalization_hint = Self::load_finalization_hint(artifact_loader);
        let proving_assembly =
            <Self::ThisProofSystem as SnarkWrapperProofSystem>::synthesize_for_proving::<
                Self::PreviousProofSystem,
            >(input_vk, input_proof);
        let proof_config = <Self::ThisProofSystem as SnarkWrapperProofSystem>::proof_config();
        Self::prove_step(
            ctx,
            proving_assembly,
            proof_config,
            precomputation,
            finalization_hint,
        )
    }
}

pub trait SnarkWrapperStepExt:
    SnarkWrapperStep<ThisProofSystem: SnarkWrapperProofSystemExt>
{
    fn run_precomputation_for_compression<AL>(
        artifact_loader: &AL,
    ) -> (
        <Self::ThisProofSystem as ProofSystemDefinition>::Precomputation,
        <Self::ThisProofSystem as ProofSystemDefinition>::VK,
    )
    where
        AL: ArtifactLoader,
    {
        let input_vk = Self::load_previous_vk(artifact_loader);
        let finalization_hint = Self::load_finalization_hint(artifact_loader);
        let setup_assembly =
            <Self::ThisProofSystem as SnarkWrapperProofSystemExt>::synthesize_for_setup::<
                Self::PreviousProofSystem,
            >(input_vk);
        let (precomputation, vk) =
            <Self::ThisProofSystem as ProofSystemExt>::generate_precomputation_and_vk(
                setup_assembly,
                finalization_hint,
            );
        (precomputation.into_inner(), vk)
    }
}

pub struct FflonkSnarkWrapper;

impl StepDefinition for FflonkSnarkWrapper {
    type PreviousProofSystem = BoojumProofSystem<CompressionTreeHasherForWrapper>;
    type ThisProofSystem = FflonkProofSystem;
}

impl SnarkWrapperStep for FflonkSnarkWrapper {
    const IS_PLONK: bool = false;
    const IS_FFLONK: bool = true;
    const LAST_COMPRESSION_MODE: u8 = 5;
}
pub struct PlonkSnarkWrapper;

impl StepDefinition for PlonkSnarkWrapper {
    type PreviousProofSystem = BoojumProofSystem<CompressionTreeHasherForWrapper>;
    type ThisProofSystem = PlonkProofSystem;
}

impl SnarkWrapperStep for PlonkSnarkWrapper {
    const IS_PLONK: bool = true;
    const IS_FFLONK: bool = false;
    const LAST_COMPRESSION_MODE: u8 = 1;
}
