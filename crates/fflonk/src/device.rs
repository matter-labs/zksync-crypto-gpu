use super::*;

#[derive(Default)]
pub struct DeviceConfig {
    pub msm_chunk_size: usize,
}
pub enum Device {
    L4(DeviceConfig),
    T4(DeviceConfig),
    A100_40(DeviceConfig),
    A100_80(DeviceConfig),
    Other(DeviceConfig),
}

impl Device {
    pub fn init() -> Self {
        let memory_info = gpu_ffi::device_info(0)
            .map_err(|_| CudaError::Error(format!("DeviceInfoErr")))
            .unwrap();
        let total_in_bytes = memory_info.total;
        dbg!(total_in_bytes);

        let mut config = DeviceConfig::default();

        if total_in_bytes < 24 * 1_000_000_000 {
            // T4
            dbg!("T4");
            config.msm_chunk_size = 1 << 23;
            Device::T4(config)
        } else if total_in_bytes < 40 * 1_000_000_000 {
            // L4
            dbg!("L4");
            config.msm_chunk_size = 1 << 23;
            Device::L4(config)
        } else if total_in_bytes < 80 * 1_000_000_000 {
            dbg!("A100 40");
            config.msm_chunk_size = 1 << 23;
            Device::A100_40(config)
        } else if total_in_bytes < 85 * 1_000_000_000 {
            config.msm_chunk_size = 1 << 23;
            Device::A100_80(config)
        } else {
            unimplemented!()
        }
    }

    pub fn config(&self) -> &DeviceConfig {
        match self {
            Device::L4(config)
            | Device::T4(config)
            | Device::A100_40(config)
            | Device::A100_80(config)
            | Device::Other(config) => config,
        }
    }
}
