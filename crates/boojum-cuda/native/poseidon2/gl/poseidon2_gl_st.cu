#include "poseidon2_gl_st.cuh"

namespace poseidon2::goldilocks {

using namespace ::goldilocks;
using namespace memory;

// https://eprint.iacr.org/2023/323.pdf Fig. 1
static DEVICE_FORCEINLINE void permutation(poseidon_state &state) {
  apply_M_eps_matrix(state);
#pragma unroll
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    if (round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS) {
      apply_round_constants<true>(state, round);
      apply_non_linearity<true>(state);
      apply_M_eps_matrix(state);
    } else {
      apply_round_constants<false>(state, round);
      apply_non_linearity<false>(state);
      apply_M_I_matrix(state);
    }
  }
}

EXTERN __global__ void poseidon2_gl_st_leaves_kernel(const base_field *values, base_field *results, const unsigned rows_count, const unsigned cols_count,
                                                     const unsigned count, const bool load_intermediate, const bool store_intermediate) {
  static_assert(RATE == 8);
  static_assert(CAPACITY == 4);
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  poseidon_state state{};
  if (load_intermediate) {
    auto intermediate_results = results + gid;
#pragma unroll
    for (unsigned i = STATE_WIDTH - CAPACITY; i < STATE_WIDTH; i++, intermediate_results += count)
      state[i] = base_field::into<3>(load_cs(intermediate_results));
  }
  values += gid * rows_count;
  for (unsigned offset = 0; offset < rows_count * cols_count;) {
#pragma unroll
    for (unsigned i = 0; i < RATE; i++, offset++) {
      const unsigned row = offset % rows_count;
      const unsigned col = offset / rows_count;
      state[i] = col < cols_count ? base_field::into<3>(load_cs(values + row + col * rows_count * count)) : field<3>{};
    }
    permutation(state);
  }
  results += gid;
  if (store_intermediate) {
#pragma unroll
    for (unsigned i = STATE_WIDTH - CAPACITY; i < STATE_WIDTH; i++, results += count)
      store_cs(results, base_field::field3_to_field2(state[i]));
  } else {
#pragma unroll
    for (unsigned i = 0; i < CAPACITY; i++, results += count)
      store_cs(results, base_field::field3_to_field2(state[i]));
  }
}

EXTERN __global__ void poseidon2_gl_st_nodes_kernel(const field<4> *values, base_field *results, const unsigned count) {
  static_assert(RATE == 8);
  static_assert(CAPACITY == 4);
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  poseidon_state state{};
  values += gid;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++, values += count) {
    const auto value = load_cs(values);
    const auto v2 = reinterpret_cast<const base_field *>(&value);
#pragma unroll
    for (unsigned j = 0; j < 2; j++)
      state[j * CAPACITY + i] = base_field::into<3>(v2[j]);
  }
  permutation(state);
  results += gid;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++, results += count)
    store_cs(results, base_field::field3_to_field2(state[i]));
}

} // namespace poseidon2::goldilocks
