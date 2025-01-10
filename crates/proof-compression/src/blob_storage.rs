use std::io::{Read, Write};

pub trait BlobStorage: Send + Sync {
    fn read_scheduler_vk(&self) -> Box<dyn Read>;
    fn read_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read>;
    fn read_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Read>;
    fn read_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Read + Send + Sync>;

    fn read_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read>;
    fn read_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Read>;
    fn read_compression_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> Box<dyn Read + Send + Sync>;

    fn read_fflonk_vk(&self) -> Box<dyn Read>;
    fn read_fflonk_precomputation(&self) -> Box<dyn Read + Send + Sync>;
    fn read_fflonk_crs(&self) -> Box<dyn Read>;

    fn read_plonk_vk(&self) -> Box<dyn Read>;
    fn read_plonk_precomputation(&self) -> Box<dyn Read + Send + Sync>;
    fn read_plonk_crs(&self) -> Box<dyn Read>;
}

pub trait BlobStorageExt: BlobStorage {
    fn write_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write>;
    fn write_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Write>;
    fn write_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Write>;

    fn write_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write>;
    fn write_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Write>;
    fn write_compression_wrapper_precomputation(&self, circuit_id: u8) -> Box<dyn Write>;

    fn write_fflonk_vk(&self) -> Box<dyn Write>;
    fn write_fflonk_precomputation(&self) -> Box<dyn Write>;
    fn write_fflonk_crs(&self) -> Box<dyn Write>;

    fn write_plonk_vk(&self) -> Box<dyn Write>;
    fn write_plonk_precomputation(&self) -> Box<dyn Write>;
    fn write_plonk_crs(&self) -> Box<dyn Write>;
}

pub struct FileSystemBlobStorage;

impl FileSystemBlobStorage {
    const DATA_DIR_PATH: &str = "./data";
    const SCHEDULER_PREFIX: &str = "scheduler_recursive";
    const COMPRESSION_LAYER_PREFIX: &str = "compression";
    const COMPRESSION_WRAPPER_PREFIX: &str = "compression_wrapper";
    const FFLONK_PREFIX: &str = "fflonk";
    const PLONK_PREFIX: &str = "plonk";

    fn open_file(path: &str) -> Box<dyn Read + Send + Sync> {
        let file = std::fs::File::open(path).unwrap();
        Box::new(file)
    }

    fn create_file(path: &str) -> Box<dyn Write> {
        let file = std::fs::File::create(path).unwrap();
        Box::new(file)
    }
}

impl BlobStorage for FileSystemBlobStorage {
    fn read_scheduler_vk(&self) -> Box<dyn Read> {
        let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::SCHEDULER_PREFIX,);
        println!("Reading scheduler vk at path {}", path);
        Self::open_file(&path)
    }

    fn read_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read> {
        let path = format!(
            "{}/{}_{}_hint.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_LAYER_PREFIX,
            circuit_id
        );
        println!("Reading compression layer finalization at path {}", path);
        Self::open_file(&path)
    }

    fn read_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Read> {
        let path = format!(
            "{}/{}_{}_vk.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_LAYER_PREFIX,
            circuit_id
        );
        println!("Reading compression layer vk at path {}", path);
        Self::open_file(&path)
    }

    fn read_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Read + Send + Sync> {
        let path = format!(
            "{}/{}_{}_setup.bin",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_LAYER_PREFIX,
            circuit_id
        );
        println!("Reading compression layer precomputation at path {}", path);
        Self::open_file(&path)
    }

    fn read_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Read> {
        let path = format!(
            "{}/{}_{}_hint.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_WRAPPER_PREFIX,
            circuit_id
        );
        println!("Reading compression wrapper finalization at path {}", path);
        Self::open_file(&path)
    }

    fn read_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Read> {
        let path = format!(
            "{}/{}_{}_vk.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_WRAPPER_PREFIX,
            circuit_id
        );
        println!("Reading compression wrapper vk at path {}", path);
        Self::open_file(&path)
    }

    fn read_compression_wrapper_precomputation(
        &self,
        circuit_id: u8,
    ) -> Box<dyn Read + Send + Sync> {
        let path = format!(
            "{}/{}_{}_setup.bin",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_WRAPPER_PREFIX,
            circuit_id
        );
        println!(
            "Reading compression wrapper precomputation at path {}",
            path
        );
        Self::open_file(&path)
    }

    fn read_fflonk_vk(&self) -> Box<dyn Read> {
        let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
        println!("Reading fflonk vk at path {}", path);
        Self::open_file(&path)
    }

    fn read_fflonk_precomputation(&self) -> Box<dyn Read + Send + Sync> {
        let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
        println!("Reading fflonk precomputation at path {}", path);
        Self::open_file(&path)
    }

    fn read_plonk_precomputation(&self) -> Box<dyn Read + Send + Sync> {
        let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
        println!("Reading plonk precomputation at path {}", path);
        Self::open_file(&path)
    }

    fn read_plonk_vk(&self) -> Box<dyn Read> {
        let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
        println!("Reading plonk vk at path {}", path);
        Self::open_file(&path)
    }

    fn read_fflonk_crs(&self) -> Box<dyn Read> {
        let path = format!(
            "{}/{}_compact_crs.key.raw",
            Self::DATA_DIR_PATH,
            Self::FFLONK_PREFIX
        );
        println!("Reading fflonk CRS at path {}", path);
        Self::open_file(&path)
    }

    fn read_plonk_crs(&self) -> Box<dyn Read> {
        let path = format!(
            "{}/{}_compact_crs.key.raw",
            Self::DATA_DIR_PATH,
            Self::PLONK_PREFIX
        );
        println!("Reading fflonk CRS at path {}", path);
        Self::open_file(&path)
    }
}

impl BlobStorageExt for FileSystemBlobStorage {
    fn write_compression_layer_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_{}_hint.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_LAYER_PREFIX,
            circuit_id
        );
        println!("Writing compression layer finalization at path {}", path);
        Self::create_file(&path)
    }

    fn write_compression_layer_vk(&self, circuit_id: u8) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_{}_vk.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_LAYER_PREFIX,
            circuit_id
        );
        println!("Writeing compression layer vk at path {}", path);
        Self::create_file(&path)
    }

    fn write_compression_layer_precomputation(&self, circuit_id: u8) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_{}_setup.bin",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_LAYER_PREFIX,
            circuit_id
        );
        println!("Writeing compression layer precomputation at path {}", path);
        Self::create_file(&path)
    }

    fn write_compression_wrapper_finalization_hint(&self, circuit_id: u8) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_{}_hint.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_WRAPPER_PREFIX,
            circuit_id
        );
        println!("Writeing compression wrapper finalization at path {}", path);
        Self::create_file(&path)
    }

    fn write_compression_wrapper_vk(&self, circuit_id: u8) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_{}_vk.json",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_WRAPPER_PREFIX,
            circuit_id
        );
        println!("Writeing compression wrapper vk at path {}", path);
        Self::create_file(&path)
    }

    fn write_compression_wrapper_precomputation(&self, circuit_id: u8) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_{}_setup.bin",
            Self::DATA_DIR_PATH,
            Self::COMPRESSION_WRAPPER_PREFIX,
            circuit_id
        );
        println!(
            "Writeing compression wrapper precomputation at path {}",
            path
        );
        Self::create_file(&path)
    }

    fn write_fflonk_vk(&self) -> Box<dyn Write> {
        let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
        println!("Writeing fflonk vk at path {}", path);
        Self::create_file(&path)
    }

    fn write_fflonk_precomputation(&self) -> Box<dyn Write> {
        let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::FFLONK_PREFIX);
        println!("Writeing fflonk precomputation at path {}", path);
        Self::create_file(&path)
    }

    fn write_plonk_precomputation(&self) -> Box<dyn Write> {
        let path = format!("{}/{}_setup.bin", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
        println!("Writeing plonk precomputation at path {}", path);
        Self::create_file(&path)
    }

    fn write_plonk_vk(&self) -> Box<dyn Write> {
        let path = format!("{}/{}_vk.json", Self::DATA_DIR_PATH, Self::PLONK_PREFIX);
        println!("Writeing plonk vk at path {}", path);
        Self::create_file(&path)
    }

    fn write_fflonk_crs(&self) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_compact_crs.key.raw",
            Self::DATA_DIR_PATH,
            Self::PLONK_PREFIX
        );
        println!("Writeing fflonk CRS at path {}", path);
        Self::create_file(&path)
    }

    fn write_plonk_crs(&self) -> Box<dyn Write> {
        let path = format!(
            "{}/{}_compact_crs.key.raw",
            Self::DATA_DIR_PATH,
            Self::PLONK_PREFIX
        );
        println!("Writeing plonk CRS at path {}", path);
        Self::create_file(&path)
    }
}

pub struct AsyncHandler<T: Send + Sync + 'static> {
    receiver: std::thread::JoinHandle<std::sync::mpsc::Receiver<T>>,
}

impl<T> AsyncHandler<T>
where
    T: Send + Sync + 'static,
{
    pub fn spawn<F>(f: F) -> Self
    where
        F: FnOnce() -> std::sync::mpsc::Receiver<T> + Send + Sync + 'static,
    {
        let receiver = std::thread::spawn(f);

        Self { receiver }
    }

    pub fn wait(self) -> T {
        self.receiver.join().unwrap().recv().unwrap()
    }
}

unsafe impl<T> Send for AsyncHandler<T> where T: Send + Sync {}
unsafe impl<T> Sync for AsyncHandler<T> where T: Send + Sync {}
