#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use boojum::field::goldilocks::GoldilocksField;
use boojum_cuda::poseidon2::{poseidon2_bn_pow, BNHasher, GLHasher, GpuTreeHasher};
use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use era_criterion_cuda::CudaMeasurement;
use era_cudart::memory::{memory_copy, memory_set, DeviceAllocation};
use era_cudart::result::CudaResult;
use era_cudart::slice::CudaSlice;
use era_cudart::stream::CudaStream;
use rand::prelude::*;
use rayon::prelude::*;
use std::time::Duration;

fn bench_leafs<H: GpuTreeHasher, const VALUES_PER_LEAF: usize>(
    c: &mut Criterion<CudaMeasurement>,
    group_name: String,
) -> CudaResult<()> {
    const MIN_LOG_N: usize = 8;
    const MAX_LOG_N: usize = 23;
    let mut initialized = false;
    let mut values_device = DeviceAllocation::alloc(VALUES_PER_LEAF << MAX_LOG_N)?;
    let mut results_device = DeviceAllocation::alloc(H::CAPACITY << MAX_LOG_N)?;
    let stream = CudaStream::default();
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);
    for log_count in MIN_LOG_N..=MAX_LOG_N {
        // let bytes = (VALUES_PER_LEAF * size_of::<GL>()) << log_count;
        // group.throughput(Throughput::Bytes(bytes as u64));
        let elements = ((VALUES_PER_LEAF + 1) / (H::RATE * H::CHUNKING) - 1) << log_count;
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function(BenchmarkId::from_parameter(log_count), |b| {
            if !initialized {
                let values_host: Vec<GoldilocksField> = (0..values_device.len())
                    .into_par_iter()
                    .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
                    .collect();
                memory_copy(&mut values_device, &values_host).unwrap();
                initialized = true;
            }
            b.iter(|| {
                let values = &values_device[..VALUES_PER_LEAF << log_count];
                let results = &mut results_device[..H::CAPACITY << log_count];
                H::compute_leaf_hashes(values, results, 0, false, false, &stream).unwrap();
            })
        });
    }
    group.finish();
    stream.destroy()?;
    results_device.free()?;
    values_device.free()?;
    Ok(())
}

fn gl_leafs(c: &mut Criterion<CudaMeasurement>) {
    bench_leafs::<GLHasher, 64>(c, String::from("gl_leafs")).unwrap();
}

fn bn_leafs(c: &mut Criterion<CudaMeasurement>) {
    bench_leafs::<BNHasher, 47>(c, String::from("bn_leafs")).unwrap();
}

fn bench_nodes<H: GpuTreeHasher>(
    c: &mut Criterion<CudaMeasurement>,
    group_name: String,
) -> CudaResult<()> {
    const MIN_LOG_N: usize = 8;
    const MAX_LOG_N: usize = 23;
    let mut initialized = false;
    let mut values_device = DeviceAllocation::alloc(H::RATE << MAX_LOG_N)?;
    let mut results_device = DeviceAllocation::alloc(H::CAPACITY << MAX_LOG_N)?;
    let stream = CudaStream::default();
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_millis(2500));
    group.sampling_mode(SamplingMode::Flat);
    for log_count in MIN_LOG_N..=MAX_LOG_N {
        // let bytes = (H::RATE * size_of::<H::NodesType>()) << log_count;
        // group.throughput(Throughput::Bytes(bytes as u64));
        let elements = 1 << log_count;
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function(BenchmarkId::from_parameter(log_count), |b| {
            if !initialized {
                unsafe {
                    memory_set(values_device.transmute_mut(), 0).unwrap();
                }
                initialized = true;
            }
            b.iter(|| {
                let values = &values_device[..H::RATE << log_count];
                let results = &mut results_device[..H::CAPACITY << log_count];
                H::compute_node_hashes(values, results, &stream).unwrap();
            })
        });
    }
    group.finish();
    stream.destroy()?;
    results_device.free()?;
    values_device.free()?;
    Ok(())
}

fn gl_nodes(c: &mut Criterion<CudaMeasurement>) {
    bench_nodes::<GLHasher>(c, String::from("gl_nodes")).unwrap();
}

fn bn_nodes(c: &mut Criterion<CudaMeasurement>) {
    bench_nodes::<BNHasher>(c, String::from("bn_nodes")).unwrap();
}

fn bench_merkle_tree<H: GpuTreeHasher, const VALUES_PER_LEAF: usize>(
    c: &mut Criterion<CudaMeasurement>,
    group_name: String,
) -> CudaResult<()> {
    const MIN_LOG_N: usize = 8;
    const MAX_LOG_N: usize = 23;
    let mut initialized = false;
    let mut values_device = DeviceAllocation::alloc(VALUES_PER_LEAF << MAX_LOG_N)?;
    let mut results_device = DeviceAllocation::alloc(H::CAPACITY << (MAX_LOG_N + 1))?;
    let stream = CudaStream::default();
    let mut group = c.benchmark_group(group_name);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    group.sampling_mode(SamplingMode::Flat);
    for log_count in MIN_LOG_N..=MAX_LOG_N {
        let bytes = (VALUES_PER_LEAF * size_of::<GoldilocksField>()) << log_count;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_function(BenchmarkId::from_parameter(log_count), |b| {
            if !initialized {
                let values_host: Vec<GoldilocksField> = (0..values_device.len())
                    .into_par_iter()
                    .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
                    .collect();
                memory_copy(&mut values_device, &values_host).unwrap();
                initialized = true;
            }
            b.iter(|| {
                let values = &values_device[..VALUES_PER_LEAF << log_count];
                let results = &mut results_device[..H::CAPACITY << (log_count + 1)];
                H::build_merkle_tree(values, results, 0, &stream, log_count as u32).unwrap();
            })
        });
    }
    group.finish();
    stream.destroy()?;
    results_device.free()?;
    values_device.free()?;
    Ok(())
}

fn gl_merkle_tree(c: &mut Criterion<CudaMeasurement>) {
    bench_merkle_tree::<GLHasher, 64>(c, String::from("gl_merkle_tree")).unwrap();
}

fn bn_merkle_tree(c: &mut Criterion<CudaMeasurement>) {
    bench_merkle_tree::<BNHasher, 47>(c, String::from("bn_merkle_tree")).unwrap();
}

fn bn_pow(c: &mut Criterion<CudaMeasurement>) {
    const MIN_BITS_COUNT: u32 = 15;
    const MAX_BITS_COUNT: u32 = 26;
    let d_seed = DeviceAllocation::alloc(4).unwrap();
    let mut d_result = DeviceAllocation::alloc(1).unwrap();
    let stream = CudaStream::default();
    let mut group = c.benchmark_group("bn_pow");
    for bits_count in MIN_BITS_COUNT..=MAX_BITS_COUNT {
        let max_nonce = 1 << bits_count;
        group.throughput(Throughput::Elements(max_nonce));
        group.bench_function(BenchmarkId::from_parameter(bits_count), |b| {
            b.iter(|| {
                poseidon2_bn_pow(&d_seed, u32::MAX, max_nonce, &mut d_result[0], &stream).unwrap();
            })
        });
    }
    group.finish();
}

criterion_group!(
    name = bench_poseidon2;
    config = Criterion::default().with_measurement::<CudaMeasurement>(CudaMeasurement{});
    targets = gl_leafs, bn_leafs, gl_nodes, bn_nodes, gl_merkle_tree, bn_merkle_tree, bn_pow
);

criterion_main!(bench_poseidon2);
