use super::*;

pub trait DigestType: 'static + Copy + Clone + PartialEq + Eq + Default + Debug {}

impl<T: 'static + Copy + Clone + Eq + Default + Debug> DigestType for T {}

pub trait GpuTreeHasher:
    boojum_cuda::poseidon2::GpuTreeHasher<
    DigestElements: DigestType,
    DigestElementType: DigestType,
    Output: DigestType + Serialize + for<'a> Deserialize<'a>,
>
{
}

impl GpuTreeHasher for GLHasher {}
impl GpuTreeHasher for BNHasher {}

pub fn build_tree<H: GpuTreeHasher>(
    leaf_sources: &[F],
    result: &mut [H::DigestElementType],
    source_len: usize,
    cap_size: usize,
    num_elems_per_leaf: usize,
) -> CudaResult<()> {
    assert!(!leaf_sources.is_empty());
    let num_leafs = source_len / num_elems_per_leaf;
    let log_cap = cap_size.trailing_zeros();
    let depth = num_leafs.trailing_zeros();
    let num_layers = depth - log_cap + 1;
    let (leaf_sources, result) = unsafe {
        (
            DeviceSlice::from_slice(leaf_sources),
            DeviceSlice::from_mut_slice(result),
        )
    };
    if_not_dry_run! {
        H::build_merkle_tree(
            leaf_sources,
            result,
            num_elems_per_leaf.trailing_zeros(),
            get_stream(),
            num_layers,
        )
    }
}

#[allow(dead_code)]
pub fn build_leaves_from_chunk<H: GpuTreeHasher>(
    leaf_sources: &[F],
    result: &mut [H::DigestElementType],
    _domain_size: usize,
    load_intermediate: bool,
    store_intermediate: bool,
) -> CudaResult<()> {
    let (d_values, d_result) = unsafe {
        (
            DeviceSlice::from_slice(leaf_sources),
            DeviceSlice::from_mut_slice(result),
        )
    };
    if_not_dry_run! {
        H::build_merkle_tree_leaves(
            d_values,
            d_result,
            0,
            load_intermediate,
            store_intermediate,
            get_stream(),
        )
    }
}

#[allow(dead_code)]
pub fn build_tree_nodes<H: GpuTreeHasher>(
    leaf_hashes: &[H::DigestElementType],
    result: &mut [H::DigestElementType],
    domain_size: usize,
    cap_size: usize,
) -> CudaResult<()> {
    assert!(!leaf_hashes.is_empty());
    let _num_sources = leaf_hashes.len() / domain_size;
    let num_leafs = domain_size;
    let log_cap = cap_size.trailing_zeros();
    let depth = num_leafs.trailing_zeros();
    let num_layers = depth - log_cap + 1;
    let (leaf_sources, result) = unsafe {
        (
            DeviceSlice::from_slice(leaf_hashes),
            DeviceSlice::from_mut_slice(result),
        )
    };
    if_not_dry_run! {
        H::build_merkle_tree_nodes(
            leaf_sources,
            result,
            num_layers,
            get_stream(),
        )
    }
}
