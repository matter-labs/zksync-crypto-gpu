use std::rc::Rc;

use super::*;
use crate::primitives::tree::build_tree;

// We can use move trees back to the cpu while proof is generated.
// This will allow us to leave gpu earlier and do fri queries on the cpu
#[derive(Clone)]
pub struct SubTree<H: GpuTreeHasher> {
    pub nodes: Rc<DVec<H::DigestElementType>>,
    pub num_leafs: usize,
    pub cap_size: usize,
}

impl<H: GpuTreeHasher> SubTree<H> {
    pub fn new(nodes: Rc<DVec<H::DigestElementType>>, num_leafs: usize, cap_size: usize) -> Self {
        assert!(num_leafs.is_power_of_two());
        assert!(cap_size.is_power_of_two());
        assert_eq!(nodes.len(), 2 * num_leafs * H::CAPACITY);
        SubTree {
            nodes,
            num_leafs,
            cap_size,
        }
    }
}

pub trait OracleData<T> {
    fn domain_size(&self) -> usize;
    fn as_single_slice(&self) -> &[T];
}

impl<H: GpuTreeHasher> OracleData<H::DigestElementType> for SubTree<H> {
    fn domain_size(&self) -> usize {
        self.num_leafs
    }

    fn as_single_slice(&self) -> &[H::DigestElementType] {
        &self.nodes
    }
}

pub fn compute_tree_cap<H: GpuTreeHasher>(
    leaf_sources: &[F],
    result: &mut [H::DigestElementType],
    source_len: usize,
    cap_size: usize,
    num_elems_per_leaf: usize,
) -> CudaResult<Vec<H::Output>> {
    build_tree::<H>(
        leaf_sources,
        result,
        source_len,
        cap_size,
        num_elems_per_leaf,
    )?;
    let tree_cap = get_tree_cap_from_nodes::<H>(result, cap_size)?;
    // TODO: transfer subtree to the host
    Ok(tree_cap)
}

pub fn get_tree_cap_from_nodes<H: GpuTreeHasher>(
    result: &[H::DigestElementType],
    cap_size: usize,
) -> CudaResult<Vec<H::Output>> {
    let result_len = result.len();
    let actual_cap_len = H::CAPACITY * cap_size;
    let cap_start_pos = result_len - 2 * actual_cap_len;
    let cap_end_pos = cap_start_pos + actual_cap_len;
    let range = cap_start_pos..cap_end_pos;
    let len = range.len();

    let mut layer_nodes = vec![H::DigestElementType::default(); len];
    mem::d2h(&result[range], &mut layer_nodes)?;

    let mut cap_values = vec![];
    for node_idx in 0..cap_size {
        let mut actual = H::DigestElements::default();
        for col_idx in 0..H::CAPACITY {
            let idx = col_idx * cap_size + node_idx;
            actual[col_idx] = layer_nodes[idx];
        }
        cap_values.push(actual.into());
    }
    assert_eq!(cap_values.len(), cap_size);

    Ok(cap_values)
}

pub fn build_subtree<H: GpuTreeHasher>(
    d_leaf_sources: &impl AsSingleSlice,
    cap_size: usize,
    num_elems_per_leaf: usize,
    mut nodes: DVec<H::DigestElementType>,
) -> CudaResult<(SubTree<H>, Vec<H::Output>)> {
    let domain_size = d_leaf_sources.domain_size();
    let leaf_sources = d_leaf_sources.as_single_slice();
    build_tree::<H>(
        leaf_sources,
        &mut nodes,
        domain_size,
        cap_size,
        num_elems_per_leaf,
    )?;
    let subtree_root = get_tree_cap_from_nodes::<H>(&nodes, cap_size)?;
    let num_leafs = domain_size / num_elems_per_leaf;
    let subtree = SubTree::new(Rc::new(nodes), num_leafs, cap_size);
    Ok((subtree, subtree_root))
}

#[allow(clippy::too_many_arguments)]
pub fn batch_query<H: GpuTreeHasher, A: GoodAllocator>(
    d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
    d_leaf_sources: &impl AsSingleSlice,
    num_cols: usize,
    d_oracle_data: &impl OracleData<H::DigestElementType>,
    cap_size: usize,
    num_rows: usize,
    num_elems_per_leaf: usize,
    h_all_leaf_elems: &mut Vec<F, A>,
    h_all_proofs: &mut Vec<H::DigestElementType, A>,
) -> CudaResult<()> {
    batch_query_leaf_sources::<H, A>(
        d_indexes,
        d_leaf_sources,
        num_cols,
        num_rows,
        num_elems_per_leaf,
        h_all_leaf_elems,
    )?;
    batch_query_tree::<H, A>(
        d_indexes,
        d_oracle_data,
        cap_size,
        num_rows,
        num_elems_per_leaf,
        h_all_proofs,
    )?;

    Ok(())
}

pub fn batch_query_tree<H: GpuTreeHasher, A: GoodAllocator>(
    d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
    d_oracle_data: &impl OracleData<H::DigestElementType>,
    cap_size: usize,
    num_rows: usize,
    num_elems_per_leaf: usize,
    h_all_proofs: &mut Vec<H::DigestElementType, A>,
) -> CudaResult<()> {
    use era_cudart::slice::DeviceSlice;
    let num_queries = d_indexes.len();
    assert!(num_rows.is_power_of_two());
    assert!(cap_size.is_power_of_two());
    assert!(num_elems_per_leaf.is_power_of_two());
    let num_leafs = num_rows / num_elems_per_leaf;
    assert_eq!(num_leafs, d_oracle_data.domain_size());
    let num_layers = (num_leafs.trailing_zeros() - cap_size.trailing_zeros()) as usize;
    if num_layers == 0 {
        return Ok(());
    }
    let mut d_all_proofs = dvec!(num_queries * H::CAPACITY * num_layers);
    assert!(h_all_proofs.capacity() >= d_all_proofs.len());
    unsafe { h_all_proofs.set_len(d_all_proofs.len()) };
    let (d_indexes_ref, d_oracle_data, d_all_proof_elems_ref) = unsafe {
        (
            DeviceSlice::from_slice(d_indexes),
            DeviceSlice::from_slice(d_oracle_data.as_single_slice()),
            DeviceSlice::from_mut_slice(&mut d_all_proofs),
        )
    };
    if_not_dry_run!(H::gather_merkle_paths(
        d_indexes_ref,
        d_oracle_data,
        d_all_proof_elems_ref,
        num_layers as u32,
        get_stream(),
    ))?;
    mem::d2h(&d_all_proofs, &mut h_all_proofs[..])?;

    Ok(())
}

pub fn batch_query_leaf_sources<H: GpuTreeHasher, A: GoodAllocator>(
    d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
    d_leaf_sources: &impl AsSingleSlice,
    num_cols: usize,
    num_rows: usize,
    num_elems_per_leaf: usize,
    h_all_leaf_elems: &mut Vec<F, A>,
) -> CudaResult<()> {
    use boojum_cuda::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use era_cudart::slice::DeviceSlice;
    let num_queries = d_indexes.len();
    assert!(num_rows.is_power_of_two());
    assert_eq!(num_rows, d_leaf_sources.domain_size());
    assert!(num_elems_per_leaf.is_power_of_two());
    assert_eq!(d_leaf_sources.len() % num_rows, 0);
    let num_polys = d_leaf_sources.len() / num_rows;
    // assert_eq!(d_leaf_sources.num_polys(), num_polys);
    assert_eq!(num_polys, num_cols);
    let mut d_all_leaf_elems = dvec!(num_queries * num_polys * num_elems_per_leaf);
    assert!(h_all_leaf_elems.capacity() >= d_all_leaf_elems.len());
    unsafe { h_all_leaf_elems.set_len(d_all_leaf_elems.len()) };
    let (d_indexes_ref, d_leaf_sources, mut d_all_leaf_elems_ref) = unsafe {
        (
            DeviceSlice::from_slice(d_indexes),
            DeviceMatrix::new(
                DeviceSlice::from_slice(d_leaf_sources.as_single_slice()),
                num_rows,
            ),
            DeviceMatrixMut::new(
                DeviceSlice::from_mut_slice(&mut d_all_leaf_elems),
                num_queries * num_elems_per_leaf,
            ),
        )
    };
    let log_rows_per_index = num_elems_per_leaf.trailing_zeros();
    if_not_dry_run!(H::gather_rows(
        d_indexes_ref,
        log_rows_per_index,
        &d_leaf_sources,
        &mut d_all_leaf_elems_ref,
        get_stream(),
    ))?;
    mem::d2h(&d_all_leaf_elems, &mut h_all_leaf_elems[..])?;

    Ok(())
}
