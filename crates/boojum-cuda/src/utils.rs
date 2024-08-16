use era_cudart::device::{device_get_attribute, get_device};
use era_cudart::execution::{Dim3, KernelFunction};
use era_cudart::occupancy::max_active_blocks_per_multiprocessor;
use era_cudart::result::CudaResult;
use era_cudart_sys::CudaDeviceAttr::MultiProcessorCount;
use std::cmp::min;

pub const WARP_SIZE: u32 = 32;

pub fn get_grid_block_dims_for_threads_count(
    threads_per_block: u32,
    threads_count: u32,
) -> (Dim3, Dim3) {
    let block_dim = min(threads_count, threads_per_block);
    let grid_dim = (threads_count + block_dim - 1) / block_dim;
    (grid_dim.into(), block_dim.into())
}

pub fn get_waves_count(
    kernel_function: &impl KernelFunction,
    block_size: u32,
    grid_size: u32,
    dynamic_smem_size: usize,
) -> CudaResult<u32> {
    let device_id = get_device()?;
    let mpc = device_get_attribute(MultiProcessorCount, device_id)?;
    let max = max_active_blocks_per_multiprocessor(
        kernel_function,
        block_size as i32,
        dynamic_smem_size,
    )?;
    Ok((grid_size - 1) / (mpc * max) as u32 + 1)
}
