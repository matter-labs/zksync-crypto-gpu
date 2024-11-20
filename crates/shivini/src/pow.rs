use super::*;
use boojum::blake2::Blake2s256;
use boojum::cs::implementations::pow::{NoPow, PoWRunner};
use boojum_cuda::blake2s::blake2s_pow;
use boojum_cuda::poseidon2::{poseidon2_bn_pow, BNHasher};
use era_cudart::slice::DeviceSlice;

pub trait GPUPoWRunner: PoWRunner {
    fn run(seed: &[F], pow_bits: u32) -> CudaResult<u64>;
}

impl GPUPoWRunner for NoPow {
    fn run(_seed: &[F], _pow_bits: u32) -> CudaResult<u64> {
        unreachable!()
    }
}

impl GPUPoWRunner for Blake2s256 {
    fn run(h_seed: &[F], pow_bits: u32) -> CudaResult<u64> {
        let mut d_seed = svec!(h_seed.len());
        let mut d_result = svec!(1);
        d_seed.copy_from_slice(h_seed)?;
        unsafe {
            let seed = DeviceSlice::from_slice(&d_seed).transmute();
            let result = &mut DeviceSlice::from_mut_slice(&mut d_result)[0];
            blake2s_pow(seed, pow_bits, u64::MAX, result, get_stream())?;
        }
        let h_result = d_result.to_vec()?;
        Ok(h_result[0])
    }
}

impl GPUPoWRunner for BNHasher {
    fn run(h_seed: &[F], pow_bits: u32) -> CudaResult<u64> {
        let mut d_seed = svec!(h_seed.len());
        let mut d_result = svec!(1);
        d_seed.copy_from_slice(h_seed)?;
        unsafe {
            let seed = DeviceSlice::from_slice(&d_seed);
            let result = &mut DeviceSlice::from_mut_slice(&mut d_result)[0];
            poseidon2_bn_pow(seed, pow_bits, u64::MAX, result, get_stream())?;
        }
        let h_result = d_result.to_vec()?;
        Ok(h_result[0])
    }
}
