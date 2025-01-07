use super::*;

use circuit_definitions::circuit_definitions::{
    aux_layer::{
        compression::ProofCompressionFunction,
        compression_modes::{
            CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
            CompressionMode4, CompressionMode5ForWrapper, CompressionTreeHasherForWrapper,
        },
        CompressionProofsTreeHasher,
    },
    recursion_layer::RecursiveProofsTreeHasher,
};

pub trait CompressionStep:
    ProofCompressionFunction + StepDefinition<ThisProofSystem: CompressionProofSystem>
{
    const MODE: u8;
    const IS_WRAPPER: bool;
    fn load_finalization_hint<AL>(
        artifact_loader: &AL,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::FinalizationHint
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
    ) -> <Self::PreviousProofSystem as ProofSystemDefinition>::VK
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

    fn load_this_vk<AL>(
        artifact_loader: &AL,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::VK
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
    ) -> AsyncHandler<<Self::ThisProofSystem as ProofSystemDefinition>::Precomputation>
    where
        AL: ArtifactLoader,
    {
        todo!()
    }

    fn prove_compression_step<AL, CI>(
        input_proof: <Self::PreviousProofSystem as ProofSystemDefinition>::Proof,
        artifact_loader: &AL,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::Proof
    where
        AL: ArtifactLoader,
        CI: ContextInitializator,
    {
        let input_vk = Self::load_previous_vk(artifact_loader);
        let precomputation = Self::get_precomputation(artifact_loader);
        let config = <Self::ThisProofSystem as ProofSystemDefinition>::get_context_config();
        let ctx = CI::init::<Self::ThisProofSystem>(config);
        let finalization_hint = Self::load_finalization_hint(artifact_loader);
        let proving_assembly =
            <Self::ThisProofSystem as CompressionProofSystem>::synthesize_for_proving::<
                Self::PreviousProofSystem,
            >(input_vk, input_proof, Self::MODE);
        let proof_config =
            <Self::ThisProofSystem as CompressionProofSystem>::proof_config_for_compression_step::<
                Self,
            >();
        Self::prove_step(
            ctx,
            proving_assembly,
            proof_config,
            precomputation,
            finalization_hint,
        )
    }
}

pub trait CompressionStepExt: CompressionStep<ThisProofSystem: CompressionProofSystemExt> {
    fn run_precomputation_for_compression<AL>(
        artifact_loader: &AL,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::Proof
    where
        AL: ArtifactLoader,
    {
        let input_vk = Self::load_previous_vk(artifact_loader);
        let finalization_hint = Self::load_finalization_hint(artifact_loader);
        let setup_assembly =
            <Self::ThisProofSystem as CompressionProofSystemExt>::synthesize_for_setup::<
                Self::PreviousProofSystem,
            >(input_vk, Self::MODE);
        <Self::ThisProofSystem as ProofSystemExt>::generate_precomputation_and_vk(
            setup_assembly,
            finalization_hint,
        );
        todo!()
    }
}

impl StepDefinition for CompressionMode1 {
    type PreviousProofSystem = BoojumProofSystem<RecursiveProofsTreeHasher>;
    type ThisProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
}

impl CompressionStep for CompressionMode1 {
    const MODE: u8 = 1;
    const IS_WRAPPER: bool = false;
}

impl StepDefinition for CompressionMode2 {
    type PreviousProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
    type ThisProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
}

impl CompressionStep for CompressionMode2 {
    const MODE: u8 = 2;
    const IS_WRAPPER: bool = false;
}

impl StepDefinition for CompressionMode3 {
    type PreviousProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
    type ThisProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
}

impl CompressionStep for CompressionMode3 {
    const MODE: u8 = 3;
    const IS_WRAPPER: bool = false;
}

impl StepDefinition for CompressionMode4 {
    type PreviousProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
    type ThisProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
}

impl CompressionStep for CompressionMode4 {
    const MODE: u8 = 4;
    const IS_WRAPPER: bool = false;
}

impl StepDefinition for CompressionMode1ForWrapper {
    type PreviousProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
    type ThisProofSystem = BoojumProofSystem<CompressionTreeHasherForWrapper>;
}

impl CompressionStep for CompressionMode1ForWrapper {
    const MODE: u8 = 1;
    const IS_WRAPPER: bool = true;
}

impl StepDefinition for CompressionMode5ForWrapper {
    type PreviousProofSystem = BoojumProofSystem<CompressionProofsTreeHasher>;
    type ThisProofSystem = BoojumProofSystem<CompressionTreeHasherForWrapper>;
}

impl CompressionStep for CompressionMode5ForWrapper {
    const MODE: u8 = 5;
    const IS_WRAPPER: bool = true;
}
