#pragma once

#include "../../goldilocks.cuh"
#include "poseidon2_gl_constants.cuh"

namespace poseidon2::goldilocks {

using namespace ::goldilocks;

// https://eprint.iacr.org/2023/323.pdf Appendix B
static DEVICE_FORCEINLINE void m4_times_tile(field<3> *tile) {
  typedef field<3> f;
  const f t0 = f::add_limbs(tile[0], tile[1]);       //  t0 = x[0] + x[1]
  const f t1 = f::add_limbs(tile[2], tile[3]);       //  t1 = x[2] + x[3]
  const f t2 = f::add_limbs(f::shl(tile[1], 1), t1); //  t2 = 2 * x[1] + t1
  const f t3 = f::add_limbs(f::shl(tile[3], 1), t0); //  t3 = 2 * x[3] + t0
  const f t4 = f::add_limbs(f::shl(t1, 2), t3);      //  t4 = 4 * t1 + t3
  const f t5 = f::add_limbs(f::shl(t0, 2), t2);      //  t5 = 4 * t0 + t2
  const f t6 = f::add_limbs(t3, t5);                 //  t6 = t3 + t5
  const f t7 = f::add_limbs(t2, t4);                 //  t7 = t2 + t4
  tile[0] = t6;
  tile[1] = t5;
  tile[2] = t7;
  tile[3] = t4;
}

} // namespace poseidon2::goldilocks