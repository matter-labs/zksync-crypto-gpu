#include "../queries.cuh"
#include "poseidon2_gl.cuh"

namespace poseidon2::goldilocks {

using namespace memory;

typedef ::goldilocks::base_field gl;

EXTERN __global__ void poseidon2_gl_gather_rows_kernel(const unsigned *indexes, const unsigned indexes_count, const matrix_getter<gl, ld_modifier::cs> values,
                                                       const matrix_setter<gl, st_modifier::cs> results) {
  gather_rows<gl>(indexes, indexes_count, values, results);
}

EXTERN __global__ void poseidon2_gl_gather_merkle_paths_kernel(const unsigned *indexes, const unsigned indexes_count, const gl *values,
                                                               const unsigned log_leaves_count, gl *results) {
  gather_merkle_paths<gl, CAPACITY>(indexes, indexes_count, values, log_leaves_count, results);
}

} // namespace poseidon2::goldilocks
