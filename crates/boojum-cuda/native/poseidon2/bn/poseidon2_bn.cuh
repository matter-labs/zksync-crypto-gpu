#pragma once
#include "../../bn254.cuh"
#include "../../goldilocks.cuh"
#include "poseidon2_bn_constants.cuh"

namespace poseidon2::bn254 {

using namespace ::bn254;
using namespace ::goldilocks;
using namespace memory;

typedef base_field gl;
typedef fr::storage bn;

constexpr unsigned STATE_WIDTH = RATE + CAPACITY;
constexpr unsigned NUM_FULL_ROUNDS = 2 * HALF_NUM_FULL_ROUNDS;
constexpr unsigned NUM_ALL_ROUNDS = NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS;

__constant__ constexpr bn ROUND_CONSTANTS[NUM_ALL_ROUNDS][STATE_WIDTH] = ROUND_CONSTANTS_VALUES;

} // namespace poseidon2::bn254
