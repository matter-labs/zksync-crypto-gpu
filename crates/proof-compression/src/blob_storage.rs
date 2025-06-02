use crate::CompressionSetupData;
use crate::CompressionStep;
use crate::CompressionStepExt;
use crate::FflonkSnarkWrapper;
use crate::PlonkSnarkWrapper;
use crate::ProofSystemDefinition;
use crate::SnarkWrapperProofSystem;
use crate::SnarkWrapperSetupData;
use crate::SnarkWrapperStep;
use crate::SnarkWrapperStepExt;
use anyhow::Context;
use circuit_definitions::boojum::{
    cs::implementations::verifier::VerificationKey, field::goldilocks::GoldilocksField,
};
use circuit_definitions::circuit_definitions::aux_layer::compression_modes::{
    CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
    CompressionMode4, CompressionMode5ForWrapper,
};

pub trait CompressorBlobStorage: Send + Sync + 'static {
    fn get_compression_mode1_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode1>>;
    fn get_compression_mode2_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode2>>;
    fn get_compression_mode3_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode3>>;
    fn get_compression_mode4_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode4>>;
    fn get_compression_mode5_for_wrapper_setup_data(
        &self,
    ) -> anyhow::Result<&CompressionSetupData<CompressionMode5ForWrapper>>;
    fn get_compression_mode1_for_wrapper_setup_data(
        &self,
    ) -> anyhow::Result<&CompressionSetupData<CompressionMode1ForWrapper>>;

    fn get_plonk_snark_wrapper_setup_data(&self) -> anyhow::Result<SnarkWrapperSetupData<PlonkSnarkWrapper>>;
    fn get_fflonk_snark_wrapper_setup_data(&self) -> anyhow::Result<&SnarkWrapperSetupData<FflonkSnarkWrapper>>;
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

use std::io::{Read, Write};
pub(crate) struct FileSystemBlobStorage {
    compression_mode1_setup_data: Option<CompressionSetupData<CompressionMode1>>,
    compression_mode2_setup_data: Option<CompressionSetupData<CompressionMode2>>,
    compression_mode3_setup_data: Option<CompressionSetupData<CompressionMode3>>,
    compression_mode4_setup_data: Option<CompressionSetupData<CompressionMode4>>,
    compression_mode5_for_wrapper_setup_data:
        Option<CompressionSetupData<CompressionMode5ForWrapper>>,
    compression_mode1_for_wrapper_setup_data:
        Option<CompressionSetupData<CompressionMode1ForWrapper>>,
    plonk_snark_wrapper_setup_data: Option<SnarkWrapperSetupData<PlonkSnarkWrapper>>,
    fflonk_snark_wrapper_setup_data: Option<SnarkWrapperSetupData<FflonkSnarkWrapper>>,
}

impl FileSystemBlobStorage {
    const DATA_DIR_PATH: &str = "./data";
    const SCHEDULER_PREFIX: &str = "scheduler_recursive";
    const COMPRESSION_LAYER_PREFIX: &str = "compression";
    const COMPRESSION_WRAPPER_PREFIX: &str = "compression_wrapper";
    const FFLONK_PREFIX: &str = "fflonk";
    const PLONK_PREFIX: &str = "plonk";

    fn open_file(path: &str) -> anyhow::Result<Box<dyn Read + Send + Sync>> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open file at path: {}", path))?;
        Ok(Box::new(file))
    }

    fn create_file(path: &str) -> anyhow::Result<Box<dyn Write>> {
        let file = std::fs::File::create(path)
            .with_context(|| format!("Failed to create file at path: {}", path))?;
        Ok(Box::new(file))
    }
}

impl FileSystemBlobStorage {
    pub fn new() -> Self {
        Self {
            compression_mode1_setup_data: None,
            compression_mode2_setup_data: None,
            compression_mode3_setup_data: None,
            compression_mode4_setup_data: None,
            compression_mode5_for_wrapper_setup_data: None,
            compression_mode1_for_wrapper_setup_data: None,
            plonk_snark_wrapper_setup_data: None,
            fflonk_snark_wrapper_setup_data: None,
        }
    }

    pub fn load_all_resources_for_fflonk(&self) -> anyhow::Result<Self> {
        Ok(Self {
            compression_mode1_setup_data: Some(
                self.load_compression_setup_data::<CompressionMode1>()
                    .context("Failed to get compression mode 1 setup data")?,
            ),
            compression_mode2_setup_data: Some(
                self.load_compression_setup_data::<CompressionMode2>()
                    .context("Failed to get compression mode 2 setup data")?,
            ),
            compression_mode3_setup_data: Some(
                self.load_compression_setup_data::<CompressionMode3>()
                    .context("Failed to get compression mode 3 setup data")?,
            ),
            compression_mode4_setup_data: Some(
                self.load_compression_setup_data::<CompressionMode4>()
                    .context("Failed to get compression mode 4 setup data")?,
            ),
            compression_mode5_for_wrapper_setup_data: Some(
                self.load_compression_setup_data::<CompressionMode5ForWrapper>()
                    .context("Failed to get compression mode 5 for wrapper setup data")?,
            ),
            fflonk_snark_wrapper_setup_data: Some(
                self.load_snark_wrapper_setup_data::<FflonkSnarkWrapper>()
                    .context("Failed to get Fflonk snark wrapper setup data")?,
            ),
            compression_mode1_for_wrapper_setup_data: None,
            plonk_snark_wrapper_setup_data: None,
        })
    }

    pub fn load_all_resources_for_plonk(&self) -> anyhow::Result<Self> {
        Ok(Self {
            compression_mode1_setup_data: None,
            compression_mode2_setup_data: None,
            compression_mode3_setup_data: None,
            compression_mode4_setup_data: None,
            compression_mode5_for_wrapper_setup_data: None,
            compression_mode1_for_wrapper_setup_data: Some(
                self.load_compression_setup_data::<CompressionMode1ForWrapper>()
                    .context("Failed to get compression mode 1 for wrapper setup data")?,
            ),
            // We will load the Plonk snark wrapper setup data later in-place
            plonk_snark_wrapper_setup_data: None,
            fflonk_snark_wrapper_setup_data: None,
        })
    }

    fn load_compression_vk<CS: CompressionStep>(&self) -> anyhow::Result<CS::VK> {
        let prefix = if CS::IS_WRAPPER {
            Self::COMPRESSION_WRAPPER_PREFIX
        } else {
            Self::COMPRESSION_LAYER_PREFIX
        };
        let path = format!("{}/{}_{}_vk.json", Self::DATA_DIR_PATH, prefix, CS::MODE);
        println!("Reading compression vk at path {}", path);
        let reader = Self::open_file(&path)?;
        let vk = CS::load_this_vk(reader);
        vk
    }
    fn load_compression_precomputation<CS: CompressionStep>(
        &self,
    ) -> anyhow::Result<CS::Precomputation> {
        let prefix = if CS::IS_WRAPPER {
            Self::COMPRESSION_WRAPPER_PREFIX
        } else {
            Self::COMPRESSION_LAYER_PREFIX
        };
        let path = format!("{}/{}_{}_setup.bin", Self::DATA_DIR_PATH, prefix, CS::MODE);
        println!("Reading compression precomputation at path {}", path);
        let reader = Self::open_file(&path)?;
        let precomputation = CS::get_precomputation(reader);
        precomputation
    }
    fn load_compression_finalization_hint<CS: CompressionStep>(
        &self,
    ) -> anyhow::Result<CS::FinalizationHint> {
        let prefix = if CS::IS_WRAPPER {
            Self::COMPRESSION_WRAPPER_PREFIX
        } else {
            Self::COMPRESSION_LAYER_PREFIX
        };
        let path = format!("{}/{}_{}_hint.json", Self::DATA_DIR_PATH, prefix, CS::MODE);
        println!("Reading compression finalization hint at path {}", path);
        let reader = Self::open_file(&path)?;
        let finalization_hint = CS::load_finalization_hint(reader);
        finalization_hint
    }
    fn load_compression_previous_vk<CS: CompressionStep>(
        &self,
    ) -> anyhow::Result<
        VerificationKey<GoldilocksField, <CS as CompressionStep>::PreviousStepTreeHasher>,
    > {
        assert!(CS::MODE >= 1);
        let path = if CS::MODE == 1 {
            format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::SCHEDULER_PREFIX,)
        } else {
            format!(
                "{}/{}_{}_vk.json",
                Self::DATA_DIR_PATH,
                Self::COMPRESSION_LAYER_PREFIX,
                CS::MODE - 1
            )
        };
        println!("Reading compression previous vk at path {}", path);
        let reader = Self::open_file(&path)?;
        let previous_vk = CS::load_previous_vk(reader);
        previous_vk
    }
    fn load_compression_setup_data<CS: CompressionStep>(
        &self,
    ) -> anyhow::Result<CompressionSetupData<CS>> {
        let vk = self.load_compression_vk::<CS>()?;
        let previous_vk = self.load_compression_previous_vk::<CS>()?;
        let precomputation = self.load_compression_precomputation::<CS>()?;
        let finalization_hint = self.load_compression_finalization_hint::<CS>()?;
        Ok(CompressionSetupData {
            vk,
            previous_vk,
            precomputation,
            finalization_hint,
        })
    }

    fn load_snark_wrapper_vk<SW: SnarkWrapperStep>(&self) -> anyhow::Result<SW::VK> {
        let prefix = if SW::IS_FFLONK {
            Self::FFLONK_PREFIX
        } else {
            Self::PLONK_PREFIX
        };
        let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, prefix);
        println!("Reading snark wrapper vk at path {}", path);
        let reader = Self::open_file(&path)?;
        let vk = SW::load_this_vk(reader);
        vk
    }
    fn load_snark_wrapper_precomputation<SW: SnarkWrapperStep>(
        &self,
    ) -> anyhow::Result<SW::Precomputation> {
        let prefix = if SW::IS_FFLONK {
            Self::FFLONK_PREFIX
        } else {
            Self::PLONK_PREFIX
        };
        let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, prefix);
        println!("Reading snark wrapper precomputation at path {}", path);
        let reader = Self::open_file(&path)?;
        let precomputation = SW::get_precomputation(reader);
        precomputation
    }
    fn load_snark_wrapper_finalization_hint<SW: SnarkWrapperStep>(
        &self,
    ) -> anyhow::Result<SW::FinalizationHint> {
        let finalization_hint = SW::load_finalization_hint();
        finalization_hint
    }
    fn load_snark_wrapper_previous_vk<SW: SnarkWrapperStep>(
        &self,
    ) -> anyhow::Result<
        VerificationKey<GoldilocksField, <SW as SnarkWrapperStep>::PreviousStepTreeHasher>,
    > {
        let previous_compression_mode = SW::PREVIOUS_COMPRESSION_MODE;
        let path = format!(
            "{}/{}_{}_vk.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_WRAPPER_PREFIX,
            previous_compression_mode
        );
        println!("Reading snark wrapper previous vk at path {}", path);
        let reader = Self::open_file(&path)?;
        let previous_vk = SW::load_previous_vk(reader);
        previous_vk
    }
    fn load_snark_wrapper_ctx<SW: SnarkWrapperStep>(&self) -> anyhow::Result<SW::Context> {
        let path = format!("{}/compact_raw_crs.key", Self::DATA_DIR_PATH,);
        println!("Reading CRS at path {}", path);
        let reader = Self::open_file(&path)?;
        let crs = <SW as SnarkWrapperStep>::load_compact_raw_crs(reader)?;
        let ctx = <SW as SnarkWrapperProofSystem>::init_context(crs);
        ctx
    }
    fn load_snark_wrapper_setup_data<SW: SnarkWrapperStep>(
        &self,
    ) -> anyhow::Result<SnarkWrapperSetupData<SW>> {
        let vk = self.load_snark_wrapper_vk::<SW>()?;
        let previous_vk = self.load_snark_wrapper_previous_vk::<SW>()?;
        let precomputation = self.load_snark_wrapper_precomputation::<SW>()?;
        let finalization_hint = self.load_snark_wrapper_finalization_hint::<SW>()?;
        let ctx = self.load_snark_wrapper_ctx::<SW>()?;
        Ok(SnarkWrapperSetupData {
            vk,
            previous_vk,
            precomputation,
            finalization_hint,
            ctx,
        })
    }

    fn store_compression_setup_data<CS: CompressionStepExt>(
        &self,
        precomputation: &CS::Precomputation,
        vk: &CS::VK,
        finalization_hint: &CS::FinalizationHint,
    ) -> anyhow::Result<()> {
        let prefix = if CS::IS_WRAPPER {
            Self::COMPRESSION_WRAPPER_PREFIX
        } else {
            Self::COMPRESSION_LAYER_PREFIX
        };
        let path = format!("{}/{}_{}_setup.bin", Self::DATA_DIR_PATH, prefix, CS::MODE);
        println!("Writing compression setup data at path {}", path);
        let writer = Self::create_file(&path)?;
        <CS as CompressionStepExt>::store_precomputation(precomputation, writer)?;

        let path = format!("{}/{}_{}_vk.json", Self::DATA_DIR_PATH, prefix, CS::MODE);
        println!("Writing compression vk at path {}", path);
        let writer = Self::create_file(&path)?;
        <CS as CompressionStepExt>::store_vk(vk, writer)?;

        let path = format!("{}/{}_{}_hint.json", Self::DATA_DIR_PATH, prefix, CS::MODE);
        println!("Writing compression finalization hint at path {}", path);
        let writer = Self::create_file(&path)?;
        <CS as CompressionStepExt>::store_finalization_hint(finalization_hint, writer)?;

        Ok(())
    }

    fn store_snark_wrapper_setup_data<WS: SnarkWrapperStepExt>(
        &self,
        precomputation: &<WS as ProofSystemDefinition>::Precomputation,
        vk: &<WS as ProofSystemDefinition>::VK,
    ) -> anyhow::Result<()> {
        let prefix = if WS::IS_FFLONK {
            Self::FFLONK_PREFIX
        } else {
            Self::PLONK_PREFIX
        };
        let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, prefix);
        println!("Writing snark wrapper precomputation at path {}", path);
        let writer = Self::create_file(&path)?;
        WS::store_precomputation(precomputation, writer)?;

        let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, prefix);
        println!("Writing snark wrapper vk at path {}", path);
        let writer = Self::create_file(&path)?;
        WS::store_vk(vk, writer)?;

        Ok(())
    }

    pub fn write_compact_raw_crs(&self) -> anyhow::Result<Box<dyn Write>> {
        let path = format!("{}/compact_raw_crs.key", Self::DATA_DIR_PATH);
        println!("Writing compact raw CRS at path {}", path);
        Self::create_file(&path)
    }
}

impl CompressorBlobStorage for FileSystemBlobStorage {
    fn get_compression_mode1_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode1>> {
        self.compression_mode1_setup_data
            .as_ref()
            .context("Compression mode 1 setup data should be initialized")
    }

    fn get_compression_mode2_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode2>> {
        self.compression_mode2_setup_data
            .as_ref()
            .context("Compression mode 2 setup data should be initialized")
    }

    fn get_compression_mode3_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode3>> {
        self.compression_mode3_setup_data
            .as_ref()
            .context("Compression mode 3 setup data should be initialized")
    }

    fn get_compression_mode4_setup_data(&self) -> anyhow::Result<&CompressionSetupData<CompressionMode4>> {
        self.compression_mode4_setup_data
            .as_ref()
            .context("Compression mode 4 setup data should be initialized")
    }

    fn get_compression_mode5_for_wrapper_setup_data(
        &self,
    ) -> anyhow::Result<&CompressionSetupData<CompressionMode5ForWrapper>> {
        self.compression_mode5_for_wrapper_setup_data
            .as_ref()
            .context("Compression mode 5 for wrapper setup data should be initialized")
    }

    fn get_compression_mode1_for_wrapper_setup_data(
        &self,
    ) -> anyhow::Result<&CompressionSetupData<CompressionMode1ForWrapper>> {
        self.compression_mode1_for_wrapper_setup_data
            .as_ref()
            .context("Compression mode 1 for wrapper setup data should be initialized")
    }

    fn get_plonk_snark_wrapper_setup_data(&self) -> anyhow::Result<SnarkWrapperSetupData<PlonkSnarkWrapper>> {
        // We load the Plonk snark wrapper setup data in-place
        self.load_snark_wrapper_setup_data::<PlonkSnarkWrapper>()
            .context("Failed to get Plonk snark wrapper setup data")
    }

    fn get_fflonk_snark_wrapper_setup_data(&self) -> anyhow::Result<&SnarkWrapperSetupData<FflonkSnarkWrapper>> {
        self.fflonk_snark_wrapper_setup_data
            .as_ref()
            .context("Fflonk snark wrapper setup data should be initialized")
    }
}

impl CompressorBlobStorageExt for FileSystemBlobStorage {
    fn get_compression_mode1_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode1 as CompressionStep>::PreviousStepTreeHasher,
        >,
    > {
        self.load_compression_previous_vk::<CompressionMode1>()
    }

    fn get_compression_mode2_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode2 as CompressionStep>::PreviousStepTreeHasher,
        >,
    > {
        self.load_compression_previous_vk::<CompressionMode2>()
    }

    fn get_compression_mode3_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode3 as CompressionStep>::PreviousStepTreeHasher,
        >,
    > {
        self.load_compression_previous_vk::<CompressionMode3>()
    }

    fn get_compression_mode4_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode4 as CompressionStep>::PreviousStepTreeHasher,
        >,
    > {
        self.load_compression_previous_vk::<CompressionMode4>()
    }

    fn get_compression_mode5_for_wrapper_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode5ForWrapper as CompressionStep>::PreviousStepTreeHasher,
        >,
    > {
        self.load_compression_previous_vk::<CompressionMode5ForWrapper>()
    }

    fn get_compression_mode1_for_wrapper_previous_vk(
        &self,
    ) -> anyhow::Result<
        VerificationKey<
            GoldilocksField,
            <CompressionMode1ForWrapper as CompressionStep>::PreviousStepTreeHasher,
        >,
    > {
        self.load_compression_previous_vk::<CompressionMode1ForWrapper>()
    }

    fn get_plonk_snark_wrapper_previous_vk_finalization_hint_and_ctx(
        &self,
    ) -> anyhow::Result<(
        VerificationKey<
            GoldilocksField,
            <PlonkSnarkWrapper as SnarkWrapperStep>::PreviousStepTreeHasher,
        >,
        <PlonkSnarkWrapper as ProofSystemDefinition>::FinalizationHint,
        <PlonkSnarkWrapper as SnarkWrapperProofSystem>::Context,
    )> {
        let vk = self.load_snark_wrapper_previous_vk::<PlonkSnarkWrapper>()?;
        let finalization_hint = PlonkSnarkWrapper::load_finalization_hint()?;
        let ctx = self.load_snark_wrapper_ctx::<PlonkSnarkWrapper>()?;
        Ok((vk, finalization_hint, ctx))
    }

    fn get_fflonk_snark_wrapper_previous_vk_finalization_hint_and_ctx(
        &self,
    ) -> anyhow::Result<(
        VerificationKey<
            GoldilocksField,
            <FflonkSnarkWrapper as SnarkWrapperStep>::PreviousStepTreeHasher,
        >,
        <FflonkSnarkWrapper as ProofSystemDefinition>::FinalizationHint,
        <FflonkSnarkWrapper as SnarkWrapperProofSystem>::Context,
    )> {
        let vk = self.load_snark_wrapper_previous_vk::<FflonkSnarkWrapper>()?;
        let finalization_hint = FflonkSnarkWrapper::load_finalization_hint()?;
        let ctx = self.load_snark_wrapper_ctx::<FflonkSnarkWrapper>()?;
        Ok((vk, finalization_hint, ctx))
    }

    fn set_compression_mode1_setup_data(
        &self,
        precomputation: &<CompressionMode1 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode1 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode1 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()> {
        self.store_compression_setup_data::<CompressionMode1>(precomputation, vk, finalization_hint)
    }

    fn set_compression_mode2_setup_data(
        &self,
        precomputation: &<CompressionMode2 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode2 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode2 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()> {
        self.store_compression_setup_data::<CompressionMode2>(precomputation, vk, finalization_hint)
    }

    fn set_compression_mode3_setup_data(
        &self,
        precomputation: &<CompressionMode3 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode3 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode3 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()> {
        self.store_compression_setup_data::<CompressionMode3>(precomputation, vk, finalization_hint)
    }
    fn set_compression_mode4_setup_data(
        &self,
        precomputation: &<CompressionMode4 as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode4 as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode4 as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()> {
        self.store_compression_setup_data::<CompressionMode4>(precomputation, vk, finalization_hint)
    }
    fn set_compression_mode5_for_wrapper_setup_data(
        &self,
        precomputation: &<CompressionMode5ForWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode5ForWrapper as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode5ForWrapper as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()> {
        self.store_compression_setup_data::<CompressionMode5ForWrapper>(
            precomputation,
            vk,
            finalization_hint,
        )
    }
    fn set_compression_mode1_for_wrapper_setup_data(
        &self,
        precomputation: &<CompressionMode1ForWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<CompressionMode1ForWrapper as ProofSystemDefinition>::VK,
        finalization_hint: &<CompressionMode1ForWrapper as ProofSystemDefinition>::FinalizationHint,
    ) -> anyhow::Result<()> {
        self.store_compression_setup_data::<CompressionMode1ForWrapper>(
            precomputation,
            vk,
            finalization_hint,
        )
    }
    fn set_plonk_snark_wrapper_setup_data(
        &self,
        precomputation: &<PlonkSnarkWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<PlonkSnarkWrapper as ProofSystemDefinition>::VK,
    ) -> anyhow::Result<()> {
        self.store_snark_wrapper_setup_data::<PlonkSnarkWrapper>(precomputation, vk)
    }
    fn set_fflonk_snark_wrapper_setup_data(
        &self,
        precomputation: &<FflonkSnarkWrapper as ProofSystemDefinition>::Precomputation,
        vk: &<FflonkSnarkWrapper as ProofSystemDefinition>::VK,
    ) -> anyhow::Result<()> {
        self.store_snark_wrapper_setup_data::<FflonkSnarkWrapper>(precomputation, vk)
    }
}
