use super::*;

pub trait StepDefinition: Sized {
    type PreviousProofSystem: ProofSystemDefinition;
    type ThisProofSystem: ProofSystemDefinition;

    fn prove_step(
        ctx: AsyncHandler<<Self::ThisProofSystem as ProofSystemDefinition>::Context>,
        proving_assembly: <Self::ThisProofSystem as ProofSystemDefinition>::ProvingAssembly,
        proof_config: <Self::ThisProofSystem as ProofSystemDefinition>::ProofConfig,
        precomputation: AsyncHandler<
            <Self::ThisProofSystem as ProofSystemDefinition>::Precomputation,
        >,
        finalization_hint: <Self::ThisProofSystem as ProofSystemDefinition>::FinalizationHint,
    ) -> <Self::ThisProofSystem as ProofSystemDefinition>::Proof {
        let _ = ctx.into_inner();
        let proof = <Self::ThisProofSystem as ProofSystemDefinition>::prove(
            proving_assembly,
            precomputation.into_inner(),
            finalization_hint,
            proof_config,
        );

        proof
    }
}

pub trait StepDefinitionExt: StepDefinition<ThisProofSystem: ProofSystemExt> {
    fn generate_precomputation_and_vk(
        ctx: AsyncHandler<<Self::ThisProofSystem as ProofSystemDefinition>::Context>,
        setup_assembly: <Self::ThisProofSystem as ProofSystemExt>::SetupAssembly,
        proof_config: <Self::ThisProofSystem as ProofSystemDefinition>::ProofConfig,
        finalization_hint: <Self::ThisProofSystem as ProofSystemDefinition>::FinalizationHint,
    ) -> (
        AsyncHandler<<Self::ThisProofSystem as ProofSystemDefinition>::Precomputation>,
        <Self::ThisProofSystem as ProofSystemDefinition>::VK,
    ) {
        let _ = ctx.into_inner();
        let (precomputation, vk) =
            <Self::ThisProofSystem as ProofSystemExt>::generate_precomputation_and_vk(
                setup_assembly,
                finalization_hint,
            );
        (precomputation, vk)
    }
}
