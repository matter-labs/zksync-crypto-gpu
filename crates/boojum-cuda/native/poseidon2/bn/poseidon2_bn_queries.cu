#include "../queries.cuh"
#include "poseidon2_bn.cuh"

namespace poseidon2::bn254 {

using namespace memory;

typedef goldilocks::base_field gl;
typedef ::bn254::fr::storage bn;

EXTERN __global__ void poseidon2_bn_gather_rows_kernel(const unsigned *indexes, const unsigned indexes_count, const matrix_getter<gl, ld_modifier::cs> values,
                                                       const matrix_setter<gl, st_modifier::cs> results) {
  gather_rows<gl>(indexes, indexes_count, values, results);
}

EXTERN __global__ void poseidon2_bn_gather_merkle_paths_kernel(const unsigned *indexes, const unsigned indexes_count, const bn *values,
                                                               const unsigned log_leaves_count, bn *results) {
  gather_merkle_paths<bn, CAPACITY>(indexes, indexes_count, values, log_leaves_count, results);
}

} // namespace poseidon2::bn254