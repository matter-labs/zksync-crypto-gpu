#include "poseidon2_bn.cuh"

namespace poseidon2::bn254 {

typedef bn poseidon_state[STATE_WIDTH];
typedef bn poseidon_shared_state[STATE_WIDTH][32];

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_round_constants(bn &state, const unsigned round, const unsigned wid) {
  if (IS_FULL_ROUND || wid == 0)
    state = fr::add(state, ROUND_CONSTANTS[round][wid]);
}

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_non_linearity(bn &state, const unsigned wid) {
  if (IS_FULL_ROUND || wid == 0)
    state = fr::mul(fr::sqr(fr::sqr(state)), state);
}

static DEVICE_FORCEINLINE void apply_M_eps_matrix(bn &state, const unsigned tid, const unsigned wid) {
  static_assert(STATE_WIDTH == 3);
  __shared__ poseidon_shared_state shared_state;
  __syncthreads();
  shared_state[wid][tid] = state;
  __syncthreads();
  // Matrix circ(2, 1, 1)
  const bn sum = fr::add(fr::add(shared_state[0][tid], shared_state[1][tid]), shared_state[2][tid]);
  state = fr::add(state, sum);
}

static DEVICE_FORCEINLINE void apply_M_I_matrix(bn &state, const unsigned tid, const unsigned wid) {
  static_assert(STATE_WIDTH == 3);
  __shared__ poseidon_shared_state shared_state;
  __syncthreads();
  shared_state[wid][tid] = state;
  __syncthreads();
  // [2, 1, 1]
  // [1, 2, 1]
  // [1, 1, 3]
  const bn sum = fr::add(fr::add(shared_state[0][tid], shared_state[1][tid]), shared_state[2][tid]);
  if (wid == 2)
    state = fr::dbl(state);
  state = fr::add(state, sum);
}

// https://eprint.iacr.org/2023/323.pdf Fig. 1
static DEVICE_FORCEINLINE void permutation(bn &state, const unsigned tid, const unsigned wid) {
  apply_M_eps_matrix(state, tid, wid);
#pragma unroll
  for (unsigned round = 0; round < NUM_ALL_ROUNDS; round++) {
    if (round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS) {
      apply_round_constants<true>(state, round, wid);
      apply_non_linearity<true>(state, wid);
      apply_M_eps_matrix(state, tid, wid);
    } else {
      apply_round_constants<false>(state, round, wid);
      apply_non_linearity<false>(state, wid);
      apply_M_I_matrix(state, tid, wid);
    }
  }
}

EXTERN __global__ void poseidon2_bn_mt_leaves_kernel(const gl *values, bn *results, const unsigned rows_count, const unsigned cols_count, const unsigned count,
                                                     const bool load_intermediate, const bool store_intermediate) {
  static_assert(RATE == 2);
  static_assert(CAPACITY == 1);
  static_assert(CHUNK_BY == 3);
  const unsigned int tid = threadIdx.x;
  const unsigned wid = threadIdx.y;
  const unsigned gid = tid + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  bn state{};
  if (wid >= STATE_WIDTH - CAPACITY && load_intermediate) {
    const auto intermediate_results = results + gid + (wid - (STATE_WIDTH - CAPACITY)) * count;
    state = load_cs(intermediate_results);
  }
  values += gid * rows_count;
  const unsigned values_count = rows_count * cols_count;
  for (unsigned offset = 0; offset < (store_intermediate ? values_count : values_count + 1);) {
    if (wid < RATE) {
      offset += wid * CHUNK_BY;
#pragma unroll
      for (unsigned j = 0; j < CHUNK_BY; j++, offset++) {
        const unsigned row = offset % rows_count;
        const unsigned col = offset / rows_count;
        reinterpret_cast<gl *>(&state)[j] = col < cols_count ? load_cs(values + row + col * rows_count * count) : (offset == values_count ? gl{1} : gl{});
      }
#pragma unroll
      for (unsigned j = CHUNK_BY * 2; j < fr::TLC; j++)
        state.limbs[j] = 0;
      state = fr::to_montgomery(state);
      offset += (RATE - wid - 1) * CHUNK_BY;
    } else
      offset += RATE * CHUNK_BY;
    permutation(state, tid, wid);
  }
  results += gid;
  if (store_intermediate) {
    if (wid >= STATE_WIDTH - CAPACITY)
      store_cs(results + (wid - (STATE_WIDTH - CAPACITY)) * count, state);
  } else {
    if (wid < CAPACITY)
      store_cs(results + wid * count, state);
  }
}

EXTERN __global__ void poseidon2_bn_mt_nodes_kernel(const bn *values, bn *results, const unsigned count) {
  static_assert(RATE == 2);
  static_assert(CAPACITY == 1);
  const unsigned int tid = threadIdx.x;
  const unsigned wid = threadIdx.y;
  const unsigned gid = tid + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  bn state{};
  values += gid * 2;
  if (wid < RATE)
    state = load_cs(values + wid % 2 + wid / 2 * count * 2);
  permutation(state, tid, wid);
  results += gid;
  if (wid < CAPACITY)
    store_cs(results + wid * count, state);
}

} // namespace poseidon2::bn254
