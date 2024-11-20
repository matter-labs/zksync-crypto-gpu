use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, DeviceRepr, MutPtrAndStride, PtrAndStride,
};
use crate::utils::{get_grid_block_dims_for_threads_count, get_waves_count, WARP_SIZE};
use boojum::algebraic_props::round_function::AbsorptionModeOverwrite;
use boojum::algebraic_props::sponge::GoldilocksPoseidon2Sponge;
use boojum::cs::oracle::TreeHasher;
use boojum::field::goldilocks::GoldilocksField;
use era_cudart::device::{device_get_attribute, get_device};
use era_cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use era_cudart::memory::memory_set_async;
use era_cudart::occupancy::max_active_blocks_per_multiprocessor;
use era_cudart::paste::paste;
use era_cudart::result::CudaResult;
use era_cudart::slice::{DeviceSlice, DeviceVariable};
use era_cudart::stream::CudaStream;
use era_cudart::{
    cuda_kernel, cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function,
};
use era_cudart_sys::CudaDeviceAttr;
use snark_wrapper::franklin_crypto::bellman::bn256::{Bn256, Fr};
use snark_wrapper::implementations::poseidon2::tree_hasher::AbsorptionModeReplacement;
use snark_wrapper::rescue_poseidon::poseidon2::Poseidon2Sponge;
use std::ops::{Deref, DerefMut};

type GL = GoldilocksField;
type BN = Fr;

cuda_kernel_signature_arguments_and_function!(
    pub LeavesKernel<T: DeviceRepr>,
    values: *const GL,
    results: *mut <T as DeviceRepr>::Type,
    rows_count: u32,
    cols_count: u32,
    count: u32,
    load_intermediate: bool,
    store_intermediate: bool,
);

macro_rules! leaves_kernel {
    ($type:ty, $thread:ident) => {
        paste! {
            cuda_kernel_declaration!(
                [<poseidon2_ $type:lower _ $thread _leaves_kernel>](
                    values: *const GL,
                    results: *mut <$type as DeviceRepr>::Type,
                    rows_count: u32,
                    cols_count: u32,
                    count: u32,
                    load_intermediate: bool,
                    store_intermediate: bool,
                )
            );
        }
    };
}

leaves_kernel!(GL, st);
leaves_kernel!(GL, mt);
leaves_kernel!(BN, st);
leaves_kernel!(BN, mt);

cuda_kernel_signature_arguments_and_function!(
    pub NodesKernel<T: DeviceRepr>,
    values: *const <T as DeviceRepr>::Type,
    results: *mut <T as DeviceRepr>::Type,
    count: u32,
);

macro_rules! nodes_kernel {
    ($type:ty, $thread:ident) => {
        paste! {
            cuda_kernel_declaration!(
                [<poseidon2_ $type:lower _ $thread _nodes_kernel>](
                    values: *const <$type as DeviceRepr>::Type,
                    results: *mut <$type as DeviceRepr>::Type,
                    count: u32,
                )
            );
        }
    };
}

nodes_kernel!(GL, st);
nodes_kernel!(GL, mt);
nodes_kernel!(BN, st);
nodes_kernel!(BN, mt);

cuda_kernel_signature_arguments_and_function!(
    pub GatherRowsKernel,
    indexes: *const u32,
    indexes_count: u32,
    values: PtrAndStride<<GL as DeviceRepr>::Type>,
    results: MutPtrAndStride<<GL as DeviceRepr>::Type>,
);

macro_rules! gather_rows_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<poseidon2_ $type:lower _gather_rows_kernel>](
                    indexes: *const u32,
                    indexes_count: u32,
                    values: PtrAndStride<<GL as DeviceRepr>::Type>,
                    results: MutPtrAndStride<<GL as DeviceRepr>::Type>,
                )
            );
        }
    };
}

gather_rows_kernel!(GL);
gather_rows_kernel!(BN);

cuda_kernel_signature_arguments_and_function!(
    pub GatherMerklePathsKernel<T: DeviceRepr>,
    indexes: *const u32,
    indexes_count: u32,
    values: *const <T as DeviceRepr>::Type,
    log_leaves_count: u32,
    results: *mut <T as DeviceRepr>::Type,
);

macro_rules! gather_merkle_paths_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<poseidon2_ $type:lower _gather_merkle_paths_kernel>](
                    indexes: *const u32,
                    indexes_count: u32,
                    values: *const <$type as DeviceRepr>::Type,
                    log_leaves_count: u32,
                    results: *mut <$type as DeviceRepr>::Type,
                )
            );
        }
    };
}

gather_merkle_paths_kernel!(GL);
gather_merkle_paths_kernel!(BN);

fn select_function_and_grid_block_dims<F: KernelFunction>(
    (mt_function, mt_parallelism): (F, u32),
    st_function: F,
    count: u32,
) -> (F, Dim3, Dim3) {
    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE, count);
    let waves_count =
        get_waves_count(&mt_function, WARP_SIZE * mt_parallelism, grid_dim.x, 0).unwrap();
    if waves_count > 1 {
        let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count);
        (st_function, grid_dim, block_dim)
    } else {
        let block_dim = Dim3::from((block_dim.x, mt_parallelism));
        (mt_function, grid_dim, block_dim)
    }
}

pub trait GpuDigestElements<T: DeviceRepr + Copy + Default>:
    Sized + Copy + Default + Deref<Target = [T]> + DerefMut<Target = [T]>
{
}

pub trait GpuTreeHasher: TreeHasher<GL> {
    type DigestElementType: DeviceRepr + Copy + Default;
    type DigestElements: GpuDigestElements<Self::DigestElementType>
        + Into<Self::Output>
        + From<Self::Output>;
    const RATE: usize;
    const CAPACITY: usize;
    const CHUNKING: usize;
    const LEAVES_ST_FUNCTION: LeavesKernelFunction<Self::DigestElementType>;
    const LEAVES_MT_FUNCTION_AND_PARALLELISM: (LeavesKernelFunction<Self::DigestElementType>, u32);
    const NODES_ST_FUNCTION: NodesKernelFunction<Self::DigestElementType>;
    const NODES_MT_FUNCTION_AND_PARALLELISM: (NodesKernelFunction<Self::DigestElementType>, u32);
    const GATHER_ROWS_FUNCTION: GatherRowsKernelFunction;
    const GATHER_MERKLE_PATHS_FUNCTION: GatherMerklePathsKernelFunction<Self::DigestElementType>;

    fn compute_leaf_hashes(
        values: &DeviceSlice<GL>,
        results: &mut DeviceSlice<Self::DigestElementType>,
        log_rows_per_hash: u32,
        load_intermediate: bool,
        store_intermediate: bool,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let values_len = values.len();
        let results_len = results.len();
        assert_eq!(results_len % Self::CAPACITY, 0);
        let count = results_len / Self::CAPACITY;
        assert_eq!(values_len % (count << log_rows_per_hash), 0);
        let values = values.as_ptr();
        let results = results.as_mut_ptr() as *mut <Self::DigestElementType as DeviceRepr>::Type;
        let rows_count = 1 << log_rows_per_hash;
        let cols_count = values_len / (count << log_rows_per_hash);
        // If this launch computes an intermediate result for a partial set of columns,
        // the kernels assume we'll complete a permutation for a full state before writing
        // the result for the current columns. This imposes a restriction on the number
        // of columns we may include in the partial set.
        assert!(
            !store_intermediate || ((rows_count * cols_count * Self::CHUNKING) % Self::RATE == 0)
        );
        assert!(cols_count <= u32::MAX as usize);
        let cols_count = cols_count as u32;
        assert!(rows_count <= u32::MAX as usize);
        let rows_count = rows_count as u32;
        assert!(count <= u32::MAX as usize);
        let count = count as u32;
        let (function, grid_dim, block_dim) = select_function_and_grid_block_dims(
            Self::LEAVES_MT_FUNCTION_AND_PARALLELISM,
            Self::LEAVES_ST_FUNCTION,
            count,
        );
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = LeavesKernelArguments::<Self::DigestElementType>::new(
            values,
            results,
            rows_count,
            cols_count,
            count,
            load_intermediate,
            store_intermediate,
        );
        function.launch(&config, &args)
    }

    fn compute_node_hashes(
        values: &DeviceSlice<Self::DigestElementType>,
        results: &mut DeviceSlice<Self::DigestElementType>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(Self::RATE, Self::CAPACITY * 2);
        let values_len = values.len();
        let results_len = results.len();
        assert_eq!(values_len % Self::RATE, 0);
        assert_eq!(results_len % Self::CAPACITY, 0);
        assert_eq!(values_len, results_len * 2);
        let values = values.as_ptr() as *const <Self::DigestElementType as DeviceRepr>::Type;
        let results = results.as_mut_ptr() as *mut <Self::DigestElementType as DeviceRepr>::Type;
        assert!(results_len / Self::CAPACITY <= u32::MAX as usize);
        let count = (results_len / Self::CAPACITY) as u32;
        let (function, grid_dim, block_dim) = select_function_and_grid_block_dims(
            Self::NODES_MT_FUNCTION_AND_PARALLELISM,
            Self::NODES_ST_FUNCTION,
            count,
        );
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = NodesKernelArguments::<Self::DigestElementType>::new(values, results, count);
        function.launch(&config, &args)
    }

    fn build_merkle_tree_leaves(
        values: &DeviceSlice<GL>,
        results: &mut DeviceSlice<Self::DigestElementType>,
        log_rows_per_hash: u32,
        load_intermediate: bool,
        store_intermediate: bool,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let values_len = values.len();
        let results_len = results.len();
        assert_eq!(results_len % Self::CAPACITY, 0);
        let leaves_count = results_len / Self::CAPACITY;
        assert_eq!(values_len % leaves_count, 0);
        Self::compute_leaf_hashes(
            values,
            results,
            log_rows_per_hash,
            load_intermediate,
            store_intermediate,
            stream,
        )
    }

    fn build_merkle_tree_nodes(
        values: &DeviceSlice<Self::DigestElementType>,
        results: &mut DeviceSlice<Self::DigestElementType>,
        layers_count: u32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        if layers_count == 0 {
            Ok(())
        } else {
            let values_len = values.len();
            let results_len = results.len();
            assert_eq!(values_len % Self::RATE, 0);
            let layer = (values_len / Self::RATE).trailing_zeros();
            assert_eq!(values_len, Self::RATE << layer);
            assert_eq!(values_len, results_len);
            let (nodes, nodes_remaining) = results.split_at_mut(results_len >> 1);
            Self::compute_node_hashes(values, nodes, stream)?;
            Self::build_merkle_tree_nodes(nodes, nodes_remaining, layers_count - 1, stream)
        }
    }

    fn build_merkle_tree(
        values: &DeviceSlice<GL>,
        results: &mut DeviceSlice<Self::DigestElementType>,
        log_rows_per_hash: u32,
        stream: &CudaStream,
        layers_count: u32,
    ) -> CudaResult<()> {
        assert_ne!(layers_count, 0);
        let values_len = values.len();
        let results_len = results.len();
        assert_eq!(results_len % (2 * Self::CAPACITY), 0);
        let leaves_count = results_len / (2 * Self::CAPACITY);
        assert!(1 << (layers_count - 1) <= leaves_count);
        assert_eq!(values_len % leaves_count, 0);
        let (nodes, nodes_remaining) = results.split_at_mut(results.len() >> 1);
        Self::build_merkle_tree_leaves(values, nodes, log_rows_per_hash, false, false, stream)?;
        Self::build_merkle_tree_nodes(nodes, nodes_remaining, layers_count - 1, stream)
    }

    fn gather_rows(
        indexes: &DeviceSlice<u32>,
        log_rows_per_index: u32,
        values: &(impl DeviceMatrixChunkImpl<GL> + ?Sized),
        result: &mut (impl DeviceMatrixChunkMutImpl<GL> + ?Sized),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let indexes_len = indexes.len();
        let values_cols = values.cols();
        let result_rows = result.rows();
        let result_cols = result.cols();
        let rows_per_index = 1 << log_rows_per_index;
        assert!(log_rows_per_index < WARP_SIZE);
        assert_eq!(result_cols, values_cols);
        assert_eq!(result_rows, indexes_len << log_rows_per_index);
        assert!(indexes_len <= u32::MAX as usize);
        let indexes_count = indexes_len as u32;
        let (mut grid_dim, block_dim) =
            get_grid_block_dims_for_threads_count(WARP_SIZE >> log_rows_per_index, indexes_count);
        let block_dim = (rows_per_index, block_dim.x);
        assert!(result_cols <= u32::MAX as usize);
        grid_dim.y = result_cols as u32;
        let indexes = indexes.as_ptr();
        let values = values.as_ptr_and_stride();
        let result = result.as_mut_ptr_and_stride();
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = GatherRowsKernelArguments::new(indexes, indexes_count, values, result);
        Self::GATHER_ROWS_FUNCTION.launch(&config, &args)
    }

    fn gather_merkle_paths(
        indexes: &DeviceSlice<u32>,
        values: &DeviceSlice<Self::DigestElementType>,
        results: &mut DeviceSlice<Self::DigestElementType>,
        layers_count: u32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert!(indexes.len() <= u32::MAX as usize);
        let indexes_count = indexes.len() as u32;
        assert_eq!(values.len() % Self::CAPACITY, 0);
        let values_count = values.len() / Self::CAPACITY;
        assert!(values_count.is_power_of_two());
        let log_values_count = values_count.trailing_zeros();
        assert_ne!(log_values_count, 0);
        let log_leaves_count = log_values_count - 1;
        assert!(layers_count <= log_leaves_count);
        assert_eq!(
            indexes.len() * layers_count as usize * Self::CAPACITY,
            results.len()
        );
        let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE, indexes_count);
        let grid_dim = (grid_dim.x, Self::CAPACITY as u32, layers_count);
        let indexes = indexes.as_ptr();
        let values = values.as_ptr() as *const <Self::DigestElementType as DeviceRepr>::Type;
        let result = results.as_mut_ptr() as *mut <Self::DigestElementType as DeviceRepr>::Type;
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        let args = GatherMerklePathsKernelArguments::<Self::DigestElementType>::new(
            indexes,
            indexes_count,
            values,
            log_leaves_count,
            result,
        );
        Self::GATHER_MERKLE_PATHS_FUNCTION.launch(&config, &args)
    }
}

pub type GLHasher = GoldilocksPoseidon2Sponge<AbsorptionModeOverwrite>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct GLDigestElements([GL; 4]);

impl Deref for GLDigestElements {
    type Target = [GL];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GLDigestElements {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuDigestElements<GL> for GLDigestElements {}

// impl Into<[GL; 4]> for GLDigestElements {
//     fn into(self) -> [GL; 4] {
//         self.0
//     }
// }

impl From<[GL; 4]> for GLDigestElements {
    fn from(value: [GL; 4]) -> Self {
        Self(value)
    }
}

impl From<GLDigestElements> for [GL; 4] {
    fn from(value: GLDigestElements) -> Self {
        value.0
    }
}

impl GpuTreeHasher for GLHasher {
    type DigestElementType = GL;
    type DigestElements = GLDigestElements;
    const RATE: usize = 8;
    const CAPACITY: usize = 4;
    const CHUNKING: usize = 1;
    const LEAVES_ST_FUNCTION: LeavesKernelFunction<Self::DigestElementType> =
        LeavesKernelFunction(poseidon2_gl_st_leaves_kernel);
    const LEAVES_MT_FUNCTION_AND_PARALLELISM: (LeavesKernelFunction<Self::DigestElementType>, u32) =
        (LeavesKernelFunction(poseidon2_gl_mt_leaves_kernel), 3);
    const NODES_ST_FUNCTION: NodesKernelFunction<Self::DigestElementType> =
        NodesKernelFunction(poseidon2_gl_st_nodes_kernel);
    const NODES_MT_FUNCTION_AND_PARALLELISM: (NodesKernelFunction<Self::DigestElementType>, u32) =
        (NodesKernelFunction(poseidon2_gl_mt_nodes_kernel), 3);
    const GATHER_ROWS_FUNCTION: GatherRowsKernelFunction =
        GatherRowsKernelFunction(poseidon2_gl_gather_rows_kernel);
    const GATHER_MERKLE_PATHS_FUNCTION: GatherMerklePathsKernelFunction<Self::DigestElementType> =
        GatherMerklePathsKernelFunction(poseidon2_gl_gather_merkle_paths_kernel);
}

pub type BNHasher = Poseidon2Sponge<Bn256, GoldilocksField, AbsorptionModeReplacement<Fr>, 2, 3>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct BNDigestElements([BN; 1]);

impl Deref for BNDigestElements {
    type Target = [BN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for BNDigestElements {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuDigestElements<BN> for BNDigestElements {}

// impl Into<BN> for BNDigestElements {
//     fn into(self) -> BN {
//         self.0[0]
//     }
// }

impl From<BN> for BNDigestElements {
    fn from(value: BN) -> Self {
        Self([value])
    }
}

impl From<BNDigestElements> for BN {
    fn from(value: BNDigestElements) -> Self {
        value.0[0]
    }
}

impl GpuTreeHasher for BNHasher {
    type DigestElementType = BN;
    type DigestElements = BNDigestElements;
    const RATE: usize = 2;
    const CAPACITY: usize = 1;
    const CHUNKING: usize = 3;
    const LEAVES_ST_FUNCTION: LeavesKernelFunction<Self::DigestElementType> =
        LeavesKernelFunction(poseidon2_bn_st_leaves_kernel);
    const LEAVES_MT_FUNCTION_AND_PARALLELISM: (LeavesKernelFunction<Self::DigestElementType>, u32) =
        (LeavesKernelFunction(poseidon2_bn_mt_leaves_kernel), 3);
    const NODES_ST_FUNCTION: NodesKernelFunction<Self::DigestElementType> =
        NodesKernelFunction(poseidon2_bn_st_nodes_kernel);
    const NODES_MT_FUNCTION_AND_PARALLELISM: (NodesKernelFunction<Self::DigestElementType>, u32) =
        (NodesKernelFunction(poseidon2_bn_mt_nodes_kernel), 3);
    const GATHER_ROWS_FUNCTION: GatherRowsKernelFunction =
        GatherRowsKernelFunction(poseidon2_bn_gather_rows_kernel);
    const GATHER_MERKLE_PATHS_FUNCTION: GatherMerklePathsKernelFunction<Self::DigestElementType> =
        GatherMerklePathsKernelFunction(poseidon2_bn_gather_merkle_paths_kernel);
}

cuda_kernel!(Poseidon2Pow, poseidon2_bn_pow_kernel(seed: *const GL, bits_count: u32, max_nonce: u64, result: *mut u64));

pub fn poseidon2_bn_pow(
    seed: &DeviceSlice<GL>,
    bits_count: u32,
    max_nonce: u64,
    result: &mut DeviceVariable<u64>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(seed.len(), 4);
    unsafe {
        memory_set_async(result.transmute_mut(), 0xff, stream)?;
    }
    const BLOCK_SIZE: u32 = WARP_SIZE * 4;
    let device_id = get_device()?;
    let mpc = device_get_attribute(CudaDeviceAttr::MultiProcessorCount, device_id)?;
    let kernel_function = Poseidon2PowFunction::default();
    let max_blocks = max_active_blocks_per_multiprocessor(&kernel_function, BLOCK_SIZE as i32, 0)?;
    let num_blocks = (mpc * max_blocks) as u32;
    let config = CudaLaunchConfig::basic(num_blocks, BLOCK_SIZE, stream);
    let seed = seed.as_ptr();
    let result = result.as_mut_ptr();
    let args = Poseidon2PowArguments {
        seed,
        bits_count,
        max_nonce,
        result,
    };
    kernel_function.launch(&config, &args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use crate::ops_simple::set_to_zero;
    use crate::tests_helpers::RandomIterator;
    use boojum::cs::implementations::pow::PoWRunner;
    use era_cudart::memory::{memory_copy_async, DeviceAllocation};
    use itertools::Itertools;
    use rand::{thread_rng, Rng};
    use rand_04::Rand;
    use std::cmp;
    use std::fmt::Debug;

    trait TestableGpuTreeHasher: GpuTreeHasher
    where
        Self::DigestElementType: Copy + Eq + Debug + Default,
    {
        fn canonical_digest_element(value: &Self::DigestElementType) -> Self::DigestElementType;
        fn random_digest_element() -> Self::DigestElementType;

        fn verify_digest(
            expected: &[Self::DigestElementType],
            actual: &[Self::DigestElementType],
            offset: usize,
            stride: usize,
        ) {
            for i in 0..Self::CAPACITY {
                let expected = Self::canonical_digest_element(&expected[i]);
                let actual = Self::canonical_digest_element(&actual[i * stride + offset]);
                assert_eq!(expected, actual);
            }
        }

        fn verify_leaves(
            values: &[GL],
            results: &[Self::DigestElementType],
            log_rows_per_hash: u32,
        ) {
            let results_len = results.len();
            assert_eq!(results_len % Self::CAPACITY, 0);
            let count = results_len / Self::CAPACITY;
            let values_len = values.len();
            assert_eq!(values_len % (count << log_rows_per_hash), 0);
            let cols_count = values_len / (count << log_rows_per_hash);
            let rows_count = 1 << log_rows_per_hash;
            let values_per_hash = cols_count << log_rows_per_hash;
            for i in 0..count {
                let mut leaf_values = Vec::with_capacity(values_per_hash);
                for idx in 0..values_per_hash {
                    let row = idx % rows_count;
                    let col = idx / rows_count;
                    let value = values[(i << log_rows_per_hash) + row + col * rows_count * count];
                    leaf_values.push(value);
                }
                let digest = Self::hash_into_leaf(&leaf_values);
                let digest: Self::DigestElements = digest.into();
                Self::verify_digest(&digest, results, i, count);
            }
        }

        fn verify_nodes(values: &[Self::DigestElementType], results: &[Self::DigestElementType]) {
            let values_len = values.len();
            assert_eq!(values_len % (Self::CAPACITY * 2), 0);
            let count = values_len / (Self::CAPACITY * 2);
            assert_eq!(results.len(), count * Self::CAPACITY);
            for i in 0..count {
                let mut nodes = [Self::DigestElements::default(); 2];
                for (j, node) in nodes.iter_mut().enumerate() {
                    for (k, value) in node.iter_mut().enumerate() {
                        let value_offset = (k * count + i) * 2 + j;
                        *value = values[value_offset];
                    }
                }
                let digest = Self::hash_into_node(&nodes[0].into(), &nodes[1].into(), 0);
                let digest: Self::DigestElements = digest.into();
                Self::verify_digest(&digest, results, i, count);
            }
        }

        fn verify_tree_nodes(
            values: &[Self::DigestElementType],
            results: &[Self::DigestElementType],
            layers_count: u32,
        ) {
            assert_eq!(values.len(), results.len());
            if layers_count == 0 {
                assert!(results
                    .iter()
                    .all(|x| x == &Self::DigestElementType::default()));
            } else {
                let (nodes, nodes_remaining) = results.split_at(results.len() >> 1);
                Self::verify_nodes(values, nodes);
                Self::verify_tree_nodes(nodes, nodes_remaining, layers_count - 1);
            }
        }

        fn test_leaves<const LOG_N: u32, const CHECKPOINTED: bool>() -> CudaResult<()> {
            const VALUES_PER_ROW: usize = 9;
            const LOG_ROWS_PER_HASH: u32 = 1;
            let checkpointed_chunk = Self::RATE * Self::CHUNKING;
            let values_host = GL::get_random_iterator()
                .take(VALUES_PER_ROW << (LOG_N + LOG_ROWS_PER_HASH))
                .collect_vec();
            let mut results_host =
                vec![Self::DigestElementType::default(); Self::CAPACITY << LOG_N];
            let stream = CudaStream::default();
            let mut values_device = DeviceAllocation::<GL>::alloc(values_host.len())?;
            let mut results_device =
                DeviceAllocation::<Self::DigestElementType>::alloc(results_host.len())?;
            memory_copy_async(&mut values_device, &values_host, &stream)?;
            if CHECKPOINTED {
                for start_col in (0..VALUES_PER_ROW).step_by(checkpointed_chunk) {
                    let end_col =
                        start_col + cmp::min(checkpointed_chunk, VALUES_PER_ROW - start_col);
                    let start_mem_idx = start_col << (LOG_N + LOG_ROWS_PER_HASH);
                    let end_mem_idx = end_col << (LOG_N + LOG_ROWS_PER_HASH);
                    Self::compute_leaf_hashes(
                        &values_device[start_mem_idx..end_mem_idx],
                        &mut results_device,
                        LOG_ROWS_PER_HASH,
                        start_col != 0,
                        end_col != VALUES_PER_ROW,
                        &stream,
                    )?;
                }
            } else {
                Self::compute_leaf_hashes(
                    &values_device,
                    &mut results_device,
                    LOG_ROWS_PER_HASH,
                    false,
                    false,
                    &stream,
                )?;
            }
            memory_copy_async(&mut results_host, &results_device, &stream)?;
            stream.synchronize()?;
            Self::verify_leaves(&values_host, &results_host, LOG_ROWS_PER_HASH);
            Ok(())
        }

        fn test_nodes<const LOG_N: u32>() -> CudaResult<()> {
            let values_host = (0..Self::RATE << LOG_N)
                .map(|_| Self::random_digest_element())
                .collect_vec();
            let mut results_host =
                vec![Self::DigestElementType::default(); Self::CAPACITY << LOG_N];
            let stream = CudaStream::default();
            let mut values_device = DeviceAllocation::alloc(values_host.len())?;
            let mut results_device = DeviceAllocation::alloc(results_host.len())?;
            memory_copy_async(&mut values_device, &values_host, &stream)?;
            Self::compute_node_hashes(&values_device, &mut results_device, &stream)?;
            memory_copy_async(&mut results_host, &results_device, &stream)?;
            stream.synchronize()?;
            Self::verify_nodes(&values_host, &results_host);
            Ok(())
        }

        fn test_merkle_tree<const LOG_N: usize>() -> CudaResult<()> {
            const VALUES_PER_ROW: usize = 9;
            let n = 1 << LOG_N;
            let layers_count = (LOG_N + 1) as u32;
            let values_host = GL::get_random_iterator()
                .take(n * VALUES_PER_ROW)
                .collect_vec();
            let mut results_host = vec![Self::DigestElementType::default(); n * Self::CAPACITY * 2];
            let stream = CudaStream::default();
            let mut values_device = DeviceAllocation::alloc(values_host.len())?;
            let mut results_device = DeviceAllocation::alloc(results_host.len())?;
            set_to_zero(&mut results_device, &stream)?;
            memory_copy_async(&mut values_device, &values_host, &stream)?;
            Self::build_merkle_tree(
                &values_device,
                &mut results_device,
                0,
                &stream,
                layers_count,
            )?;
            memory_copy_async(&mut results_host, &results_device, &stream)?;
            stream.synchronize()?;
            let (nodes, nodes_remaining) = results_host.split_at(results_host.len() >> 1);
            Self::verify_leaves(&values_host, nodes, 0);
            Self::verify_tree_nodes(nodes, nodes_remaining, layers_count - 1);
            Ok(())
        }

        fn test_gather_rows() -> CudaResult<()> {
            const SRC_LOG_ROWS: usize = 12;
            const SRC_ROWS: usize = 1 << SRC_LOG_ROWS;
            const COLS: usize = 16;
            const INDEXES_COUNT: usize = 42;
            const LOG_ROWS_PER_INDEX: usize = 1;
            const DST_ROWS: usize = INDEXES_COUNT << LOG_ROWS_PER_INDEX;
            let mut rng = thread_rng();
            let mut indexes_host = vec![0; INDEXES_COUNT];
            indexes_host.fill_with(|| rng.gen_range(0..INDEXES_COUNT as u32));
            let values_host = GL::get_random_iterator()
                .take(SRC_ROWS * COLS)
                .collect_vec();
            let mut results_host = vec![GL::default(); DST_ROWS * COLS];
            let stream = CudaStream::default();
            let mut indexes_device = DeviceAllocation::alloc(indexes_host.len())?;
            let mut values_device = DeviceAllocation::alloc(values_host.len())?;
            let mut results_device = DeviceAllocation::alloc(results_host.len())?;
            memory_copy_async(&mut indexes_device, &indexes_host, &stream)?;
            memory_copy_async(&mut values_device, &values_host, &stream)?;
            Self::gather_rows(
                &indexes_device,
                LOG_ROWS_PER_INDEX as u32,
                &DeviceMatrix::new(&values_device, SRC_ROWS),
                &mut DeviceMatrixMut::new(&mut results_device, DST_ROWS),
                &stream,
            )?;
            memory_copy_async(&mut results_host, &results_device, &stream)?;
            stream.synchronize()?;
            for (i, index) in indexes_host.into_iter().enumerate() {
                let src_index = (index as usize) << LOG_ROWS_PER_INDEX;
                let dst_index = i << LOG_ROWS_PER_INDEX;
                for j in 0..1 << LOG_ROWS_PER_INDEX {
                    let src_index = src_index + j;
                    let dst_index = dst_index + j;
                    for k in 0..COLS {
                        let expected = values_host[(k << SRC_LOG_ROWS) + src_index];
                        let actual = results_host[(k * DST_ROWS) + dst_index];
                        assert_eq!(expected, actual);
                    }
                }
            }
            Ok(())
        }

        fn test_gather_merkle_paths() -> CudaResult<()>
        where
            [(); Self::CAPACITY]:,
        {
            const LOG_LEAVES_COUNT: usize = 12;
            const INDEXES_COUNT: usize = 42;
            const LAYERS_COUNT: usize = LOG_LEAVES_COUNT;
            let mut rng = thread_rng();
            let mut indexes_host = vec![0; INDEXES_COUNT];
            indexes_host.fill_with(|| rng.gen_range(0..1u32 << LOG_LEAVES_COUNT));
            let mut values_host =
                vec![Self::DigestElementType::default(); Self::CAPACITY << (LOG_LEAVES_COUNT + 1)];
            values_host.fill_with(|| Self::random_digest_element());
            let mut results_host = vec![
                Self::DigestElementType::default();
                Self::CAPACITY * INDEXES_COUNT * LAYERS_COUNT
            ];
            let stream = CudaStream::default();
            let mut indexes_device = DeviceAllocation::<u32>::alloc(indexes_host.len())?;
            let mut values_device = DeviceAllocation::alloc(values_host.len())?;
            let mut results_device = DeviceAllocation::alloc(results_host.len())?;
            memory_copy_async(&mut indexes_device, &indexes_host, &stream)?;
            memory_copy_async(&mut values_device, &values_host, &stream)?;
            Self::gather_merkle_paths(
                &indexes_device,
                &values_device,
                &mut results_device,
                LAYERS_COUNT as u32,
                &stream,
            )?;
            memory_copy_async(&mut results_host, &results_device, &stream)?;
            stream.synchronize()?;
            fn verify_merkle_path<T: Eq + Debug, const CAPACITY: usize>(
                indexes: &[u32],
                values: &[T],
                results: &[T],
            ) {
                let (values, values_next) = values.split_at(values.len() >> 1);
                let (results, results_next) = results.split_at(INDEXES_COUNT * CAPACITY);
                values
                    .chunks(values.len() / CAPACITY)
                    .zip(results.chunks(results.len() / CAPACITY))
                    .for_each(|(values, results)| {
                        for (row_index, &index) in indexes.iter().enumerate() {
                            let sibling_index = index ^ 1;
                            let expected = &values[sibling_index as usize];
                            let actual = &results[row_index];
                            assert_eq!(expected, actual);
                        }
                    });
                if !results_next.is_empty() {
                    let indexes_next = indexes.iter().map(|&x| x >> 1).collect_vec();
                    verify_merkle_path::<T, CAPACITY>(&indexes_next, values_next, results_next);
                }
            }
            verify_merkle_path::<Self::DigestElementType, { Self::CAPACITY }>(
                &indexes_host,
                &values_host,
                &results_host,
            );
            Ok(())
        }
    }

    impl TestableGpuTreeHasher for GLHasher {
        fn canonical_digest_element(value: &Self::DigestElementType) -> Self::DigestElementType {
            GL::from_nonreduced_u64(value.to_nonreduced_u64())
        }

        fn random_digest_element() -> Self::DigestElementType {
            GL::from_nonreduced_u64(thread_rng().gen())
        }
    }

    impl TestableGpuTreeHasher for BNHasher {
        fn canonical_digest_element(value: &Self::DigestElementType) -> Self::DigestElementType {
            *value
        }

        fn random_digest_element() -> Self::DigestElementType {
            BN::rand(&mut rand_04::thread_rng())
        }
    }

    #[test]
    fn gl_leaves_small() -> CudaResult<()> {
        GLHasher::test_leaves::<4, false>()
    }

    #[test]
    fn gl_leaves_large() -> CudaResult<()> {
        GLHasher::test_leaves::<12, false>()
    }

    #[test]
    fn gl_leaves_small_checkpointed() -> CudaResult<()> {
        GLHasher::test_leaves::<4, true>()
    }

    #[test]
    fn gl_leaves_large_checkpointed() -> CudaResult<()> {
        GLHasher::test_leaves::<12, true>()
    }

    #[test]
    fn bn_leaves_small() -> CudaResult<()> {
        BNHasher::test_leaves::<4, false>()
    }

    #[test]
    fn bn_leaves_large() -> CudaResult<()> {
        BNHasher::test_leaves::<12, false>()
    }

    #[test]
    fn bn_leaves_small_checkpointed() -> CudaResult<()> {
        BNHasher::test_leaves::<4, true>()
    }

    #[test]
    fn bn_leaves_large_checkpointed() -> CudaResult<()> {
        BNHasher::test_leaves::<12, true>()
    }

    #[test]
    fn gl_nodes_small() -> CudaResult<()> {
        GLHasher::test_nodes::<4>()
    }

    #[test]
    fn gl_nodes_large() -> CudaResult<()> {
        GLHasher::test_nodes::<12>()
    }

    #[test]
    fn bn_nodes_small() -> CudaResult<()> {
        BNHasher::test_nodes::<4>()
    }

    #[test]
    fn bn_nodes_large() -> CudaResult<()> {
        BNHasher::test_nodes::<12>()
    }

    #[test]
    fn gl_merkle_tree() -> CudaResult<()> {
        GLHasher::test_merkle_tree::<12>()
    }

    #[test]
    fn bn_merkle_tree() -> CudaResult<()> {
        BNHasher::test_merkle_tree::<6>()
    }

    #[test]
    fn gl_gather_rows() -> CudaResult<()> {
        GLHasher::test_gather_rows()
    }

    #[test]
    fn bn_gather_rows() -> CudaResult<()> {
        BNHasher::test_gather_rows()
    }

    #[test]
    fn gl_gather_merkle_paths() -> CudaResult<()> {
        GLHasher::test_gather_merkle_paths()
    }

    #[test]
    fn bn_gather_merkle_paths() -> CudaResult<()> {
        BNHasher::test_gather_merkle_paths()
    }

    #[test]
    fn poseidon2_bn_pow() {
        const BITS_COUNT: u32 = 26;
        let seed = GL::get_random_iterator().take(4).collect_vec();
        let mut h_result = [0u64; 1];
        let mut d_seed = DeviceAllocation::alloc(4).unwrap();
        let mut d_result = DeviceAllocation::alloc(1).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_seed, &seed, &stream).unwrap();
        super::poseidon2_bn_pow(&d_seed, BITS_COUNT, u64::MAX, &mut d_result[0], &stream).unwrap();
        memory_copy_async(&mut h_result, &d_result, &stream).unwrap();
        stream.synchronize().unwrap();
        let challenge = h_result[0];
        assert!(BNHasher::verify_from_field_elements(
            seed, BITS_COUNT, challenge
        ));
    }
}
