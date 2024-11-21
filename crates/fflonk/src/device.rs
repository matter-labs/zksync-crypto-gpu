use super::*;

#[derive(Default, Debug)]
pub struct DeviceConfig {
    pub msm_chunk_size: usize,
    pub static_alloc_num_blocks: usize,
}
pub enum Device {
    L4(DeviceConfig),
    T4(DeviceConfig),
    A100_40(DeviceConfig),
    A100_80(DeviceConfig),
    Other(DeviceConfig),
}

impl Device {
    pub fn model() -> Self {
        let memory_info = gpu_ffi::device_info(0)
            .map_err(|_| CudaError::Error(format!("DeviceInfoErr")))
            .unwrap();
        let total_in_bytes = memory_info.total;
        dbg!(total_in_bytes);
        let free_in_bytes = memory_info.free;
        dbg!(free_in_bytes);

        let mut config = DeviceConfig::default();

        // Count production blocks
        let block_size_in_bytes = 32 << 23;
        let total_num_blocks = total_in_bytes as usize / block_size_in_bytes;
        dbg!(total_num_blocks);
        let free_num_blocks = free_in_bytes as usize / block_size_in_bytes;
        dbg!(free_num_blocks);
        let model = if total_num_blocks < 64 {
            todo!("T4 support is in progress");
            // T4
            dbg!("T4");
            config.msm_chunk_size = 1 << 23;
            Device::T4(config)
        } else if total_num_blocks < 96 {
            // L4
            dbg!("L4");
            config.msm_chunk_size = 1 << 23;
            config.static_alloc_num_blocks = 59;
            Device::L4(config)
        } else if total_num_blocks < 160 {
            dbg!("A100 40");
            config.msm_chunk_size = 1 << 23;
            config.static_alloc_num_blocks = 80;
            Device::A100_40(config)
        } else if total_num_blocks < 320 {
            dbg!("A100 80");
            config.msm_chunk_size = 1 << 23;
            config.static_alloc_num_blocks = 80;
            Device::A100_80(config)
        } else {
            unimplemented!()
        };
        dbg!(model.config());

        model
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

    pub fn static_alloc_num_blocks() -> usize {
        match Device::model() {
            Device::L4(device_config) => device_config.static_alloc_num_blocks,
            Device::T4(device_config) => device_config.static_alloc_num_blocks,
            Device::A100_40(device_config) => device_config.static_alloc_num_blocks,
            Device::A100_80(device_config) => device_config.static_alloc_num_blocks,
            Device::Other(device_config) => device_config.static_alloc_num_blocks,
        }
    }
}
