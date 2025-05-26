use crate::CompressionSetupData;
use crate::CompressionStep;
use crate::FflonkSnarkWrapper;
use crate::PlonkSnarkWrapper;
use crate::SnarkWrapperSetupData;
use crate::SnarkWrapperStep;
use circuit_definitions::boojum::{
    cs::implementations::verifier::VerificationKey, field::goldilocks::GoldilocksField,
};
use circuit_definitions::circuit_definitions::aux_layer::compression_modes::{
    CompressionMode1, CompressionMode1ForWrapper, CompressionMode2, CompressionMode3,
    CompressionMode4, CompressionMode5ForWrapper,
};
use circuit_definitions::zkevm_circuits::bn254::ec_add::input;
use std::io::Read;
use std::sync::Arc;

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
    fn set_compression_mode1_setup_data(
        &self,
        input: CompressionSetupData<CompressionMode1>,
    ) -> anyhow::Result<()>;
    fn set_compression_mode2_setup_data(
        &self,
        input: CompressionSetupData<CompressionMode2>,
    ) -> anyhow::Result<()>;
    fn set_compression_mode3_setup_data(
        &self,
        input: CompressionSetupData<CompressionMode3>,
    ) -> anyhow::Result<()>;
    fn set_compression_mode4_setup_data(
        &self,
        input: CompressionSetupData<CompressionMode4>,
    ) -> anyhow::Result<()>;
    fn set_compression_mode5_for_wrapper_setup_data(
        &self,
        input: CompressionSetupData<CompressionMode5ForWrapper>,
    ) -> anyhow::Result<()>;
    fn set_compression_mode1_for_wrapper_setup_data(
        &self,
        input: CompressionSetupData<CompressionMode1ForWrapper>,
    ) -> anyhow::Result<()>;
    fn set_plonk_snark_wrapper_setup_data(
        &self,
        input: SnarkWrapperSetupData<PlonkSnarkWrapper>,
    ) -> anyhow::Result<()>;
    fn set_fflonk_snark_wrapper_setup_data(
        &self,
        input: SnarkWrapperSetupData<FflonkSnarkWrapper>,
    ) -> anyhow::Result<()>;
}

// pub trait CompressorBlobStorage{
//     // fn read_file_for_compression(
//     //     &self,
//     //     key: ProverServiceDataKey,
//     //     service_data_type: ProverServiceDataType,
//     // ) -> Box<dyn Read>;

//     // fn read_compact_raw_crs(&self) -> Box<dyn Read>;

//     fn get_compression_vk<CS: CompressionStep>(&self) -> anyhow::Result<CS::VK>;
//     fn get_compression_precomputation<CS: CompressionStep>(
//         &self,
//     ) -> anyhow::Result<CS::Precomputation>;
//     fn get_compression_finalization_hint<CS: CompressionStep>(
//         &self,
//     ) -> anyhow::Result<CS::FinalizationHint>;
//     fn get_compression_previous_vk<CS: CompressionStep>(
//         &self,
//     ) -> anyhow::Result<VerificationKey<GoldilocksField, CS::PreviousStepTreeHasher>>;
//     // fn get_compression_setup_data<CS: CompressionStep>(
//     //     &self,
//     // ) -> anyhow::Result<CompressionSetupData<CS>> {
//     //     let vk = self.get_compression_vk::<CS>()?;
//     //     let previous_vk = self.get_compression_previous_vk::<CS>()?;
//     //     let precomputation = self.get_compression_precomputation::<CS>()?;
//     //     let finalization_hint = self.get_compression_finalization_hint::<CS>()?;

//     //     Ok(CompressionSetupData {
//     //         vk,
//     //         previous_vk,
//     //         precomputation,
//     //         finalization_hint,
//     //     })
//     // }

//     fn get_snark_wrapper_precomputation<WS: SnarkWrapperStep>(
//         &self,
//     ) -> anyhow::Result<WS::Precomputation>;
//     fn get_snark_wrapper_vk<WS: SnarkWrapperStep>(&self) -> anyhow::Result<WS::VK>;
//     fn get_snark_wrapper_finalization_hint<WS: SnarkWrapperStep>(
//         &self,
//     ) -> anyhow::Result<WS::FinalizationHint>;
//     fn get_snark_wrapper_ctx<WS: SnarkWrapperStep>(&self) -> anyhow::Result<WS::Context>;
//     fn get_snark_wrapper_previous_vk<WS: SnarkWrapperStep>(
//         &self,
//     ) -> anyhow::Result<VerificationKey<GoldilocksField, WS::PreviousStepTreeHasher>>;
//     // fn get_snark_wrapper_setup_data<WS: SnarkWrapperStep>(
//     //     &self,
//     // ) -> anyhow::Result<SnarkWrapperSetupData<WS>> {
//     //     let vk = self.get_snark_wrapper_vk::<WS>()?;
//     //     let previous_vk = self.get_snark_wrapper_previous_vk::<WS>()?;
//     //     let precomputation = self.get_snark_wrapper_precomputation::<WS>()?;
//     //     let ctx = self.get_snark_wrapper_ctx::<WS>()?;
//     //     let finalization_hint = self.get_snark_wrapper_finalization_hint::<WS>()?;

//     //     Ok(SnarkWrapperSetupData {
//     //         vk,
//     //         previous_vk,
//     //         precomputation,
//     //         finalization_hint,
//     //         ctx: Some(ctx),
//     //     })
//     // }

//     // fn get_full_fflonk_setup_data(&self) -> anyhow::Result<FflonkSetupData> {
//     //     let compression_mode1_setup_data = self.get_compression_setup_data::<CompressionMode1>()?;
//     //     let compression_mode2_setup_data = self.get_compression_setup_data::<CompressionMode2>()?;
//     //     let compression_mode3_setup_data = self.get_compression_setup_data::<CompressionMode3>()?;
//     //     let compression_mode4_setup_data = self.get_compression_setup_data::<CompressionMode4>()?;
//     //     let compression_mode5_for_wrapper_setup_data =
//     //         self.get_compression_setup_data::<CompressionMode5ForWrapper>()?;
//     //     let fflonk_snark_wrapper_setup_data =
//     //         self.get_snark_wrapper_setup_data::<FflonkSnarkWrapper>()?;

//     //     Ok(FflonkSetupData {
//     //         compression_mode1_setup_data,
//     //         compression_mode2_setup_data,
//     //         compression_mode3_setup_data,
//     //         compression_mode4_setup_data,
//     //         compression_mode5_for_wrapper_setup_data,
//     //         fflonk_snark_wrapper_setup_data,
//     //     })
//     // }

//     // fn get_full_plonk_setup_data(&self) -> anyhow::Result<PlonkSetupData> {
//     //     let compression_mode1_for_wrapper_setup_data =
//     //         self.get_compression_setup_data::<CompressionMode1ForWrapper>()?;
//     //     let plonk_snark_wrapper_setup_data =
//     //         self.get_snark_wrapper_setup_data::<PlonkSnarkWrapper>()?;

//     //     Ok(PlonkSetupData {
//     //         compression_mode1_for_wrapper_setup_data,
//     //         plonk_snark_wrapper_setup_data,
//     //     })
//     // }
// }

// pub trait CompressorBlobStorageExt: CompressorBlobStorage {
//     fn set_compression_vk<CS: CompressionStep>(&self, input: CS::VK) -> anyhow::Result<()>;
//     fn set_compression_precomputation<CS: CompressionStep>(
//         &self,
//         input: CS::Precomputation,
//     ) -> anyhow::Result<()>;
//     fn set_compression_finalization_hint<CS: CompressionStep>(
//         &self,
//         input: CS::FinalizationHint,
//     ) -> anyhow::Result<()>;
//     fn set_snark_wrapper_precomputation<WS: SnarkWrapperStep>(
//         &self,
//         input: WS::Precomputation,
//     ) -> anyhow::Result<()>;
//     fn set_snark_wrapper_vk<WS: SnarkWrapperStep>(&self, input: WS::VK) -> anyhow::Result<()>;
//     fn set_snark_wrapper_ctx<WS: SnarkWrapperStep>(&self, input: WS::Context) -> anyhow::Result<()>;
// }

// use std::io::{Read, Write};

// pub trait BlobStorage: Send + Sync {
//     fn read_scheduler_vk(&self) -> Box<dyn Read>;
//     fn read_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read>;
//     fn read_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Read>;
//     fn read_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Read + Send + Sync>;

//     fn read_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read>;
//     fn read_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Read>;
//     fn read_compression_wrapper_precomputation(
//         &self,
//         circuit_id: u8,
//     ) -> Box<dyn Read + Send + Sync>;

//     fn read_fflonk_vk(&self) -> Box<dyn Read>;
//     fn read_fflonk_precomputation(&self) -> Box<dyn Read + Send + Sync>;

//     fn read_plonk_vk(&self) -> Box<dyn Read>;
//     fn read_plonk_precomputation(&self) -> Box<dyn Read + Send + Sync>;
//     fn read_compact_raw_crs(&self) -> Box<dyn Read + Send + Sync>;
// }

// pub trait BlobStorageExt: BlobStorage {
//     fn write_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write>;
//     fn write_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Write>;
//     fn write_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Write>;

//     fn write_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write>;
//     fn write_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Write>;
//     fn write_compression_wrapper_precomputation(&self, circuit_id: u8) -> Box<dyn Write>;

//     fn write_fflonk_vk(&self) -> Box<dyn Write>;
//     fn write_fflonk_precomputation(&self) -> Box<dyn Write>;

//     fn write_plonk_vk(&self) -> Box<dyn Write>;
//     fn write_plonk_precomputation(&self) -> Box<dyn Write>;

//     fn write_compact_raw_crs(&self) -> Box<dyn Write>;
// }

// pub(crate) struct FileSystemBlobStorage;

// impl FileSystemBlobStorage {
//     const DATA_DIR_PATH: &str = "./data";
//     const SCHEDULER_PREFIX: &str = "scheduler_recursive";
//     const COMPRESSION_LAYER_PREFIX: &str = "compression";
//     const COMPRESSION_WRAPPER_PREFIX: &str = "compression_wrapper";
//     const FFLONK_PREFIX: &str = "fflonk";
//     const PLONK_PREFIX: &str = "plonk";

//     fn open_file(path: &str) -> Box<dyn Read + Send + Sync> {
//         let file = std::fs::File::open(path).unwrap();
//         Box::new(file)
//     }

//     fn create_file(path: &str) -> Box<dyn Write> {
//         let file = std::fs::File::create(path).unwrap();
//         Box::new(file)
//     }
// }

// impl BlobStorage for FileSystemBlobStorage {
//     fn read_scheduler_vk(&self) -> Box<dyn Read> {
//         let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::SCHEDULER_PREFIX,);
//         println!("Reading scheduler vk at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read> {
//         let path = format!(
//             "{}/{}_{}_hint.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_LAYER_PREFIX,
//             circuit_id
//         );
//         println!("Reading compression layer finalization at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Read> {
//         let path = format!(
//             "{}/{}_{}_vk.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_LAYER_PREFIX,
//             circuit_id
//         );
//         println!("Reading compression layer vk at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Read + Send + Sync> {
//         let path = format!(
//             "{}/{}_{}_setup.bin",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_LAYER_PREFIX,
//             circuit_id
//         );
//         println!("Reading compression layer precomputation at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read> {
//         let path = format!(
//             "{}/{}_{}_hint.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_WRAPPER_PREFIX,
//             circuit_id
//         );
//         println!("Reading compression wrapper finalization at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Read> {
//         let path = format!(
//             "{}/{}_{}_vk.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_WRAPPER_PREFIX,
//             circuit_id
//         );
//         println!("Reading compression wrapper vk at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_compression_wrapper_precomputation(
//         &self,
//         circuit_id: u8,
//     ) -> Box<dyn Read + Send + Sync> {
//         let path = format!(
//             "{}/{}_{}_setup.bin",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_WRAPPER_PREFIX,
//             circuit_id
//         );
//         println!(
//             "Reading compression wrapper precomputation at path {}",
//             path
//         );
//         Self::open_file(&path)
//     }

//     fn read_fflonk_vk(&self) -> Box<dyn Read> {
//         let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
//         println!("Reading fflonk vk at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_fflonk_precomputation(&self) -> Box<dyn Read + Send + Sync> {
//         let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
//         println!("Reading fflonk precomputation at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_plonk_precomputation(&self) -> Box<dyn Read + Send + Sync> {
//         let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
//         println!("Reading plonk precomputation at path {}", path);
//         Self::open_file(&path)
//     }

//     fn read_plonk_vk(&self) -> Box<dyn Read> {
//         let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
//         println!("Reading plonk vk at path {}", path);
//         Self::open_file(&path)
//     }
//     fn read_compact_raw_crs(&self) -> Box<dyn Read + Send + Sync> {
//         let path = format!("{}/compact_raw_crs.key", Self::DATA_DIR_PATH,);
//         println!("Reading CRS at path {}", path);
//         Self::open_file(&path)
//     }
// }

// impl BlobStorageExt for FileSystemBlobStorage {
//     fn write_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write> {
//         let path = format!(
//             "{}/{}_{}_hint.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_LAYER_PREFIX,
//             circuit_id
//         );
//         println!("Writing compression layer finalization at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Write> {
//         let path = format!(
//             "{}/{}_{}_vk.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_LAYER_PREFIX,
//             circuit_id
//         );
//         println!("Writeing compression layer vk at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Write> {
//         let path = format!(
//             "{}/{}_{}_setup.bin",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_LAYER_PREFIX,
//             circuit_id
//         );
//         println!("Writeing compression layer precomputation at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write> {
//         let path = format!(
//             "{}/{}_{}_hint.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_WRAPPER_PREFIX,
//             circuit_id
//         );
//         println!("Writeing compression wrapper finalization at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Write> {
//         let path = format!(
//             "{}/{}_{}_vk.json",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_WRAPPER_PREFIX,
//             circuit_id
//         );
//         println!("Writeing compression wrapper vk at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_compression_wrapper_precomputation(&self, circuit_id: u8) -> Box<dyn Write> {
//         let path = format!(
//             "{}/{}_{}_setup.bin",
//             Self::DATA_DIR_PATH,
//             Self::COMPRESSION_WRAPPER_PREFIX,
//             circuit_id
//         );
//         println!(
//             "Writeing compression wrapper precomputation at path {}",
//             path
//         );
//         Self::create_file(&path)
//     }

//     fn write_fflonk_vk(&self) -> Box<dyn Write> {
//         let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
//         println!("Writeing fflonk vk at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_fflonk_precomputation(&self) -> Box<dyn Write> {
//         let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
//         println!("Writeing fflonk precomputation at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_plonk_precomputation(&self) -> Box<dyn Write> {
//         let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
//         println!("Writeing plonk precomputation at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_plonk_vk(&self) -> Box<dyn Write> {
//         let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
//         println!("Writeing plonk vk at path {}", path);
//         Self::create_file(&path)
//     }

//     fn write_compact_raw_crs(&self) -> Box<dyn Write> {
//         let path = format!("{}/compact_raw_crs.key", Self::DATA_DIR_PATH);
//         println!("Writeing compact raw CRS at path {}", path);
//         Self::create_file(&path)
//     }
// }

// pub(crate) struct AsyncHandler<T: Send + Sync + 'static> {
//     receiver: std::thread::JoinHandle<std::sync::mpsc::Receiver<T>>,
// }

// impl<T> AsyncHandler<T>
// where
//     T: Send + Sync + 'static,
// {
//     pub(crate) fn spawn<F>(f: F) -> Self
//     where
//         F: FnOnce() -> std::sync::mpsc::Receiver<T> + Send + Sync + 'static,
//     {
//         let receiver = std::thread::spawn(f);

//         Self { receiver }
//     }

//     pub(crate) fn wait(self) -> T {
//         self.receiver.join().unwrap().recv().unwrap()
//     }
// }

// unsafe impl<T> Send for AsyncHandler<T> where T: Send + Sync {}
// unsafe impl<T> Sync for AsyncHandler<T> where T: Send + Sync {}
