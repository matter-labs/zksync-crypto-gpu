use crate::CompressionSetupData;
use crate::CompressionStep;
use crate::FflonkSnarkWrapper;
use crate::PlonkSnarkWrapper;
use crate::ProofSystemDefinition;
use crate::SnarkWrapperProofSystem;
use crate::SnarkWrapperSetupData;
use crate::SnarkWrapperStep;
use circuit_definitions::boojum::{
    cs::implementations::verifier::VerificationKey, field::goldilocks::GoldilocksField,
};
use circuit_definitions::circuit_definitions::aux_layer::compression_modes::{
    CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
    CompressionMode4, CompressionMode5ForWrapper,
};

pub trait CompressorBlobStorage: Send + Sync + 'static {
    fn get_compression_mode1_setup_data(&self) -> &CompressionSetupData<CompressionMode1>;
    fn get_compression_mode2_setup_data(&self) -> &CompressionSetupData<CompressionMode2>;
    fn get_compression_mode3_setup_data(&self) -> &CompressionSetupData<CompressionMode3>;
    fn get_compression_mode4_setup_data(&self) -> &CompressionSetupData<CompressionMode4>;
    fn get_compression_mode5_for_wrapper_setup_data(
        &self,
    ) -> &CompressionSetupData<CompressionMode5ForWrapper>;
    fn get_compression_mode1_for_wrapper_setup_data(
        &self,
    ) -> &CompressionSetupData<CompressionMode1ForWrapper>;

    fn get_plonk_snark_wrapper_setup_data(&self) -> &SnarkWrapperSetupData<PlonkSnarkWrapper>;
    fn get_fflonk_snark_wrapper_setup_data(&self) -> &SnarkWrapperSetupData<FflonkSnarkWrapper>;
}

pub trait CompressorBlobStorageExt: CompressorBlobStorage {
    fn get_compression_mode1_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode1 as CompressionStep>::PreviousStepTreeHasher,
        >,
    >;
    fn get_compression_mode2_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode2 as CompressionStep>::PreviousStepTreeHasher,
        >,
    >;
    fn get_compression_mode3_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode3 as CompressionStep>::PreviousStepTreeHasher,
        >,
    >;
    fn get_compression_mode4_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode4 as CompressionStep>::PreviousStepTreeHasher,
        >,
    >;
    fn get_compression_mode5_for_wrapper_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode5ForWrapper as CompressionStep>::PreviousStepTreeHasher,
        >,
    >;
    fn get_compression_mode1_for_wrapper_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode1ForWrapper as CompressionStep>::PreviousStepTreeHasher,
        >,
    >;
    fn get_plonk_snark_wrapper_previous_vk_finalization_hint_and_ctx(
        &self,
    ) -> anyhow::Result<(
        VerificationKey<
            GoldilocksField,
            <PlonkSnarkWrapper as SnarkWrapperStep>::PreviousStepTreeHasher,
        >,
        <PlonkSnarkWrapper as ProofSystemDefinition>::FinalizationHint,
        <PlonkSnarkWrapper as SnarkWrapperProofSystem>::Context,
    )>;
    fn get_fflonk_snark_wrapper_previous_vk_finalization_hint_and_ctx(
        &self,
    ) -> anyhow::Result<(
        VerificationKey<
            GoldilocksField,
            <FflonkSnarkWrapper as SnarkWrapperStep>::PreviousStepTreeHasher,
        >,
        <FflonkSnarkWrapper as ProofSystemDefinition>::FinalizationHint,
        <FflonkSnarkWrapper as SnarkWrapperProofSystem>::Context,
    )>;

    fn set_compression_mode1_setup_data(
        &self,
        precomputation: &<CompressionMode1 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode1 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode1 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()>;
    fn set_compression_mode2_setup_data(
        &self,
        precomputation: &<CompressionMode2 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode2 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode2 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()>;
    fn set_compression_mode3_setup_data(
        &self,
        precomputation: &<CompressionMode3 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode3 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode3 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()>;
    fn set_compression_mode4_setup_data(
        &self,
        precomputation: &<CompressionMode4 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode4 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode4 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()>;
    fn set_compression_mode5_for_wrapper_setup_data(
        &self,
        precomputation: &<CompressionMode5ForWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode5ForWrapper as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode5ForWrapper as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()>;
    fn set_compression_mode1_for_wrapper_setup_data(
        &self,
        precomputation: &<CompressionMode1ForWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode1ForWrapper as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode1ForWrapper as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()>;
    fn set_plonk_snark_wrapper_setup_data(
        &self,
        precomputation: &<PlonkSnarkWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<PlonkSnarkWrapper as ProofSystemDefinition>::VK,
    ) -> anyhow::Result<()>;
    fn set_fflonk_snark_wrapper_setup_data(
        &self,
        precomputation: &<FflonkSnarkWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<FflonkSnarkWrapper as ProofSystemDefinition>::VK,
    ) -> anyhow::Result<()>;
}

//TODO: Add impl of CompressorBlobStorage and CompressorBlobStorageExt for FileSystemBlobStorage for tests
