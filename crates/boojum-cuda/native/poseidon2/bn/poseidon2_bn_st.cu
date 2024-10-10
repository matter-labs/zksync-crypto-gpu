#include "poseidon2_bn.cuh"

namespace poseidon2::bn254 {

typedef bn poseidon_state[STATE_WIDTH];

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_round_constants(poseidon_state &state, const unsigned round) {
  const auto rc = ROUND_CONSTANTS[round];
#pragma unroll
  for (unsigned i = 0; i < (IS_FULL_ROUND ? STATE_WIDTH : 1); i++)
    state[i] = fr::add(state[i], rc[i]);
}

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_non_linearity(poseidon_state &state) {
#pragma unroll
  for (unsigned i = 0; i < (IS_FULL_ROUND ? STATE_WIDTH : 1); i++) {
    state[i] = fr::mul(fr::sqr(fr::sqr(state[i])), state[i]);
  }
}

static DEVICE_FORCEINLINE void apply_M_eps_matrix(poseidon_state &state) {
  static_assert(STATE_WIDTH == 3);
  // Matrix circ(2, 1, 1)
  const bn sum = fr::add(fr::add(state[0], state[1]), state[2]);
  state[0] = fr::add(state[0], sum);
  state[1] = fr::add(state[1], sum);
  state[2] = fr::add(state[2], sum);
}

static DEVICE_FORCEINLINE void apply_M_I_matrix(poseidon_state &state) {
  static_assert(STATE_WIDTH == 3);
  // [2, 1, 1]
  // [1, 2, 1]
  // [1, 1, 3]
  const bn sum = fr::add(fr::add(state[0], state[1]), state[2]);
  state[0] = fr::add(state[0], sum);
  state[1] = fr::add(state[1], sum);
  state[2] = fr::add(fr::dbl(state[2]), sum);
}
// https://eprint.iacr.org/2023/323.pdf Fig. 1
static DEVICE_FORCEINLINE void permutation(poseidon_state &state) {
  apply_M_eps_matrix(state);
#pragma unroll
  for (unsigned round = 0; round < NUM_ALL_ROUNDS; round++) {
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

EXTERN __global__ void poseidon2_bn_st_leaves_kernel(const gl *values, bn *results, const unsigned rows_count, const unsigned cols_count, const unsigned count,
                                                     const bool load_intermediate, const bool store_intermediate) {
  static_assert(RATE == 2);
  static_assert(CAPACITY == 1);
  static_assert(CHUNK_BY == 3);
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  poseidon_state state{};
  if (load_intermediate) {
    auto intermediate_results = results + gid;
#pragma unroll
    for (unsigned i = STATE_WIDTH - CAPACITY; i < STATE_WIDTH; i++, intermediate_results += count)
      state[i] = load_cs(intermediate_results);
  }
  values += gid * rows_count;
  const unsigned values_count = rows_count * cols_count;
  for (unsigned offset = 0; offset < (store_intermediate ? values_count : values_count + 1);) {
#pragma unroll
    for (unsigned i = 0; i < RATE; i++) {
      auto s = state[i];
#pragma unroll
      for (unsigned j = 0; j < CHUNK_BY; j++, offset++) {
        const unsigned row = offset % rows_count;
        const unsigned col = offset / rows_count;
        reinterpret_cast<gl *>(&s)[j] = col < cols_count ? load_cs(values + row + col * rows_count * count) : (offset == values_count ? gl{1} : gl{});
      }
#pragma unroll
      for (unsigned j = CHUNK_BY * 2; j < fr::TLC; j++)
        s.limbs[j] = 0;
      state[i] = fr::to_montgomery(s);
    }
    permutation(state);
  }
  results += gid;
  if (store_intermediate) {
#pragma unroll
    for (unsigned i = STATE_WIDTH - CAPACITY; i < STATE_WIDTH; i++, results += count)
      store_cs(results, state[i]);
  } else {
#pragma unroll
    for (unsigned i = 0; i < CAPACITY; i++, results += count)
      store_cs(results, state[i]);
  }
}

EXTERN __global__ void poseidon2_bn_st_nodes_kernel(const bn *values, bn *results, const unsigned count) {
  static_assert(RATE == 2);
  static_assert(CAPACITY == 1);
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  poseidon_state state{};
  values += gid * 2;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++, values += count * 2) {
    state[i] = load_cs(values);
    state[i + CAPACITY] = load_cs(values + 1);
  }
  permutation(state);
  results += gid;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++, results += count)
    store_cs(results, state[i]);
}

} // namespace poseidon2::bn254
