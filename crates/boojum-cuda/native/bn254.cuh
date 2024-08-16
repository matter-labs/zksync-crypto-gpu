#pragma once

#include "carry_chain.cuh"
#include "common.cuh"

namespace bn254 {

#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))
#define ALIGN __align__(LIMBS_ALIGNMENT(LIMBS_COUNT))

template <unsigned LIMBS_COUNT> struct ALIGN storage {
  static constexpr unsigned LC = LIMBS_COUNT;
  uint32_t limbs[LIMBS_COUNT];
};

template <unsigned LIMBS_COUNT> struct ALIGN storage_wide {
  static_assert(LIMBS_COUNT ^ 1);
  static constexpr unsigned LC = LIMBS_COUNT;
  static constexpr unsigned LC2 = LIMBS_COUNT * 2;
  uint32_t limbs[LC2];

  void DEVICE_FORCEINLINE set_lo(const storage<LIMBS_COUNT> &in) {
#pragma unroll
    for (unsigned i = 0; i < LC; i++)
      limbs[i] = in.limbs[i];
  }

  void DEVICE_FORCEINLINE set_hi(const storage<LIMBS_COUNT> &in) {
#pragma unroll
    for (unsigned i = 0; i < LC; i++)
      limbs[i + LC].x = in.limbs[i];
  }

  storage<LC> DEVICE_FORCEINLINE get_lo() {
    storage<LC> out{};
#pragma unroll
    for (unsigned i = 0; i < LC; i++)
      out.limbs[i] = limbs[i];
    return out;
  }

  storage<LC> DEVICE_FORCEINLINE get_hi() {
    storage<LC> out{};
#pragma unroll
    for (unsigned i = 0; i < LC; i++)
      out.limbs[i] = limbs[i + LC].x;
    return out;
  }
};

struct config_fr {
  // field structure size = 8 * 32 bit
  static constexpr unsigned limbs_count = 8;
  // modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
  static constexpr storage<limbs_count> modulus = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
  // modulus*2 = 43776485743678550444492811490514550177096728800832068687396408373151616991234
  static constexpr storage<limbs_count> modulus_2 = {0xe0000002, 0x87c3eb27, 0xf372e122, 0x5067d090, 0x0302b0ba, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
  // modulus*4 = 87552971487357100888985622981029100354193457601664137374792816746303233982468
  static constexpr storage<limbs_count> modulus_4 = {0xc0000004, 0x0f87d64f, 0xe6e5c245, 0xa0cfa121, 0x06056174, 0xe14116da, 0x84c680a6, 0xc19139cb};
  // modulus^2
  static constexpr storage_wide<limbs_count> modulus_squared = {0xe0000001, 0x08c3eb27, 0xdcb34000, 0xc7f26223, 0x68c9bb7f, 0xffe9a62c, 0xe821ddb0, 0xa6ce1975,
                                                                0x47b62fe7, 0x2c77527b, 0xd379d3df, 0x85f73bb0, 0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
  // 2*modulus^2
  static constexpr storage_wide<limbs_count> modulus_squared_2 = {0xc0000002, 0x1187d64f, 0xb9668000, 0x8fe4c447, 0xd19376ff, 0xffd34c58,
                                                                  0xd043bb61, 0x4d9c32eb, 0x8f6c5fcf, 0x58eea4f6, 0xa6f3a7be, 0x0bee7761,
                                                                  0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
  // 4*modulus^2
  static constexpr storage_wide<limbs_count> modulus_squared_4 = {0x80000004, 0x230fac9f, 0x72cd0000, 0x1fc9888f, 0xa326edff, 0xffa698b1,
                                                                  0xa08776c3, 0x9b3865d7, 0x1ed8bf9e, 0xb1dd49ed, 0x4de74f7c, 0x17dceec3,
                                                                  0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
  // r2 = 944936681149208446651664254269745548490766851729442924617792859073125903783
  static constexpr storage<limbs_count> r2 = {0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1, 0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1};
  // inv
  static constexpr uint32_t inv = 0xefffffff;
  // 1 in montgomery form
  static constexpr storage<limbs_count> one = {0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
  static constexpr unsigned modulus_bits_count = 254;
  // log2 of order of omega
  static constexpr unsigned omega_log_order = 28;
  // k=(modulus-1)/(2^omega_log_order) = (21888242871839275222246405745257275088548364400416034343698204186575808495617-1)/(2^28) =
  // 81540058820840996586704275553141814055101440848469862132140264610111
  // omega generator is 7
  static constexpr unsigned omega_generator = 7;
  // omega = generator^k mod P = 7^81540058820840996586704275553141814055101440848469862132140264610111 mod
  // 21888242871839275222246405745257275088548364400416034343698204186575808495617 =
  // 1748695177688661943023146337482803886740723238769601073607632802312037301404 =
  // omega in montgomery form
  static constexpr storage<limbs_count> omega = {0xb639feb8, 0x9632c7c5, 0x0d0ff299, 0x985ce340, 0x01b0ecd8, 0xb2dd8800, 0x6d98ce29, 0x1d69070d};
  // inverse of 2 in montgomery form
  static constexpr storage<limbs_count> two_inv = {0x1ffffffe, 0x783c14d8, 0x0c8d1edd, 0xaf982f6f, 0xfcfd4f45, 0x8f5f7492, 0x3d9cbfac, 0x1f37631a};
};

// Can't make this a member of config_fr. nvcc does not allow __constant__ on members.
extern __device__ __constant__ uint32_t inv_fr;

template <class FF_CONFIG, const uint32_t &INV> struct ff {
  // allows consumers to access the underlying config (e.g., "fd_q::CONFIG") if needed
  using CONFIG = FF_CONFIG;

  static constexpr int LPT = CONFIG::limbs_count;
  static constexpr int TPF = 1;
  static constexpr unsigned TLC = CONFIG::limbs_count;

  typedef storage<TLC> storage;
  typedef storage_wide<TLC> storage_wide;

  // return number of bits in modulus
  static constexpr unsigned MBC = CONFIG::modulus_bits_count;

  // return modulus
  template <unsigned MULTIPLIER = 1> static constexpr DEVICE_FORCEINLINE storage get_modulus() {
    switch (MULTIPLIER) {
    case 1:
      return CONFIG::modulus;
    case 2:
      return CONFIG::modulus_2;
    case 4:
      return CONFIG::modulus_4;
    default:
      return {};
    }
  }

  // return modulus^2, helpful for ab +/- cd
  template <unsigned MULTIPLIER = 1> static constexpr DEVICE_FORCEINLINE storage_wide get_modulus_squared() {
    switch (MULTIPLIER) {
    case 1:
      return CONFIG::modulus_squared;
    case 2:
      return CONFIG::modulus_squared_2;
    case 4:
      return CONFIG::modulus_squared_4;
    default:
      return {};
    }
  }

  // return r^2
  static constexpr DEVICE_FORCEINLINE storage get_r2() { return CONFIG::r2; }

  // return one in montgomery form
  static constexpr DEVICE_FORCEINLINE storage get_one() { return CONFIG::one; }

  template <bool SUBTRACT, bool CARRY_OUT> static DEVICE_FORCEINLINE uint32_t add_sub_limbs(const storage &xs, const storage &ys, storage &rs) {
    const uint32_t *x = xs.limbs;
    const uint32_t *y = ys.limbs;
    uint32_t *r = rs.limbs;
    carry_chain<CARRY_OUT ? TLC + 1 : TLC> chain;
#pragma unroll
    for (unsigned i = 0; i < TLC; i++)
      r[i] = SUBTRACT ? chain.sub(x[i], y[i]) : chain.add(x[i], y[i]);
    if (!CARRY_OUT)
      return 0;
    return SUBTRACT ? chain.sub(0, 0) : chain.add(0, 0);
  }

  // If we want, we could make "2*TLC" a template parameter to deduplicate with "storage" overload, but that's a minor issue.
  template <bool SUBTRACT, bool CARRY_OUT> static DEVICE_FORCEINLINE uint32_t add_sub_limbs(const storage_wide &xs, const storage_wide &ys, storage_wide &rs) {
    const uint32_t *x = xs.limbs;
    const uint32_t *y = ys.limbs;
    uint32_t *r = rs.limbs;
    carry_chain<CARRY_OUT ? 2 * TLC + 1 : 2 * TLC> chain;
#pragma unroll
    for (unsigned i = 0; i < 2 * TLC; i++) {
      r[i] = SUBTRACT ? chain.sub(x[i], y[i]) : chain.add(x[i], y[i]);
    }
    if (!CARRY_OUT)
      return 0;
    return SUBTRACT ? chain.sub(0, 0) : chain.add(0, 0);
  }

  template <bool CARRY_OUT, typename T> static DEVICE_FORCEINLINE uint32_t add_limbs(const T &xs, const T &ys, T &rs) {
    return add_sub_limbs<false, CARRY_OUT>(xs, ys, rs);
  }

  template <bool CARRY_OUT, typename T> static DEVICE_FORCEINLINE uint32_t sub_limbs(const T &xs, const T &ys, T &rs) {
    return add_sub_limbs<true, CARRY_OUT>(xs, ys, rs);
  }

  // return xs == 0 with field operands
  static DEVICE_FORCEINLINE bool is_zero(const storage &xs) {
    const uint32_t *x = xs.limbs;
    uint32_t limbs_or = x[0];
#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
      limbs_or |= x[i];
    return limbs_or == 0;
  }

  // return xs == ys with field operands
  static DEVICE_FORCEINLINE bool eq(const storage &xs, const storage &ys) {
    const uint32_t *x = xs.limbs;
    const uint32_t *y = ys.limbs;
    uint32_t limbs_or = x[0] ^ y[0];
#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
      limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
  }

  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage reduce(const storage &xs) {
    if (REDUCTION_SIZE == 0)
      return xs;
    const storage modulus = get_modulus<REDUCTION_SIZE>();
    storage rs = {};
    return sub_limbs<true>(xs, modulus, rs) ? xs : rs;
  }

  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage_wide reduce_wide(const storage_wide &xs) {
    if (REDUCTION_SIZE == 0)
      return xs;
    const storage_wide modulus_squared = get_modulus_squared<REDUCTION_SIZE>();
    storage_wide rs = {};
    return sub_limbs<true>(xs, modulus_squared, rs) ? xs : rs;
  }

  // return xs + ys with field operands
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage add(const storage &xs, const storage &ys) {
    storage rs = {};
    add_limbs<false>(xs, ys, rs);
    return reduce<REDUCTION_SIZE>(rs);
  }

  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage_wide add_wide(const storage_wide &xs, const storage_wide &ys) {
    storage_wide rs = {};
    add_limbs<false>(xs, ys, rs);
    return reduce_wide<REDUCTION_SIZE>(rs);
  }

  // return xs - ys with field operands
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage sub(const storage &xs, const storage &ys) {
    storage rs = {};
    if (REDUCTION_SIZE == 0) {
      sub_limbs<false>(xs, ys, rs);
    } else {
      uint32_t carry = sub_limbs<true>(xs, ys, rs);
      if (carry == 0)
        return rs;
      const storage modulus = get_modulus<REDUCTION_SIZE>();
      add_limbs<false>(rs, modulus, rs);
    }
    return rs;
  }

  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage_wide sub_wide(const storage_wide &xs, const storage_wide &ys) {
    storage_wide rs = {};
    if (REDUCTION_SIZE == 0) {
      sub_limbs<false>(xs, ys, rs);
    } else {
      uint32_t carry = sub_limbs<true>(xs, ys, rs);
      if (carry == 0)
        return rs;
      const storage_wide modulus_squared = get_modulus_squared<REDUCTION_SIZE>();
      add_limbs<false>(rs, modulus_squared, rs);
    }
    return rs;
  }

  // The following algorithms are adaptations of
  // http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf,
  // taken from https://github.com/z-prize/test-msm-gpu (under Apache 2.0 license)
  // and modified to use our datatypes.
  // We had our own implementation of http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf,
  // but the sppark versions achieved lower instruction count thanks to clever carry handling,
  // so we decided to just use theirs.

  static DEVICE_FORCEINLINE void mul_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
#pragma unroll
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  static DEVICE_FORCEINLINE void cmad_n(uint32_t *acc, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
#pragma unroll
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
    // return carry flag
  }

  static DEVICE_FORCEINLINE void madc_n_rshift(uint32_t *odd, const uint32_t *a, uint32_t bi) {
    constexpr uint32_t n = TLC;
#pragma unroll
    for (size_t i = 0; i < n - 2; i += 2) {
      odd[i] = ptx::madc_lo_cc(a[i], bi, odd[i + 2]);
      odd[i + 1] = ptx::madc_hi_cc(a[i], bi, odd[i + 3]);
    }
    odd[n - 2] = ptx::madc_lo_cc(a[n - 2], bi, 0);
    odd[n - 1] = ptx::madc_hi(a[n - 2], bi, 0);
  }

  static DEVICE_FORCEINLINE void mad_n_redc(uint32_t *even, uint32_t *odd, const uint32_t *a, uint32_t bi, bool first = false) {
    constexpr uint32_t n = TLC;
    constexpr auto modulus = CONFIG::modulus;
    const uint32_t *const MOD = modulus.limbs;
    if (first) {
      mul_n(odd, a + 1, bi);
      mul_n(even, a, bi);
    } else {
      even[0] = ptx::add_cc(even[0], odd[1]);
      madc_n_rshift(odd, a + 1, bi);
      cmad_n(even, a, bi);
      odd[n - 1] = ptx::addc(odd[n - 1], 0);
    }
    uint32_t mi = even[0] * INV;
    cmad_n(odd, MOD + 1, mi);
    cmad_n(even, MOD, mi);
    odd[n - 1] = ptx::addc(odd[n - 1], 0);
  }

  static DEVICE_FORCEINLINE void mad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    cmad_n(odd, a + 1, bi, n - 2);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, 0);
    odd[n - 1] = ptx::madc_hi(a[n - 1], bi, 0);
    cmad_n(even, a, bi, n);
    odd[n - 1] = ptx::addc(odd[n - 1], 0);
  }

  static DEVICE_FORCEINLINE void qad_row(uint32_t *odd, uint32_t *even, const uint32_t *a, uint32_t bi, size_t n = TLC) {
    cmad_n(odd, a, bi, n - 2);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 2], bi, 0);
    odd[n - 1] = ptx::madc_hi(a[n - 2], bi, 0);
    cmad_n(even, a + 1, bi, n - 2);
    odd[n - 1] = ptx::addc(odd[n - 1], 0);
  }

  static DEVICE_FORCEINLINE void multiply_raw(const storage &as, const storage &bs, storage_wide &rs) {
    const uint32_t *a = as.limbs;
    const uint32_t *b = bs.limbs;
    uint32_t *even = rs.limbs;
    __align__(8) uint32_t odd[2 * TLC - 2];
    mul_n(even, a, b[0]);
    mul_n(odd, a + 1, b[0]);
    mad_row(&even[2], &odd[0], a, b[1]);
    size_t i;
#pragma unroll
    for (i = 2; i < TLC - 1; i += 2) {
      mad_row(&odd[i], &even[i], a, b[i]);
      mad_row(&even[i + 2], &odd[i], a, b[i + 1]);
    }
    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < 2 * TLC - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
  }

  static DEVICE_FORCEINLINE void sqr_raw(const storage &as, storage_wide &rs) {
    const uint32_t *a = as.limbs;
    uint32_t *even = rs.limbs;
    size_t i = 0, j;
    __align__(8) uint32_t odd[2 * TLC - 2];

    // perform |a[i]|*|a[j]| for all j>i
    mul_n(even + 2, a + 2, a[0], TLC - 2);
    mul_n(odd, a + 1, a[0], TLC);

#pragma unroll
    while (i < TLC - 4) {
      ++i;
      mad_row(&even[2 * i + 2], &odd[2 * i], &a[i + 1], a[i], TLC - i - 1);
      ++i;
      qad_row(&odd[2 * i], &even[2 * i + 2], &a[i + 1], a[i], TLC - i);
    }

    even[2 * TLC - 4] = ptx::mul_lo(a[TLC - 1], a[TLC - 3]);
    even[2 * TLC - 3] = ptx::mul_hi(a[TLC - 1], a[TLC - 3]);
    odd[2 * TLC - 6] = ptx::mad_lo_cc(a[TLC - 2], a[TLC - 3], odd[2 * TLC - 6]);
    odd[2 * TLC - 5] = ptx::madc_hi_cc(a[TLC - 2], a[TLC - 3], odd[2 * TLC - 5]);
    even[2 * TLC - 3] = ptx::addc(even[2 * TLC - 3], 0);

    odd[2 * TLC - 4] = ptx::mul_lo(a[TLC - 1], a[TLC - 2]);
    odd[2 * TLC - 3] = ptx::mul_hi(a[TLC - 1], a[TLC - 2]);

    // merge |even[2:]| and |odd[1:]|
    even[2] = ptx::add_cc(even[2], odd[1]);
    for (j = 2; j < 2 * TLC - 3; j++)
      even[j + 1] = ptx::addc_cc(even[j + 1], odd[j]);
    even[j + 1] = ptx::addc(odd[j], 0);

    // double |even|
    even[0] = 0;
    even[1] = ptx::add_cc(odd[0], odd[0]);
    for (j = 2; j < 2 * TLC - 1; j++)
      even[j] = ptx::addc_cc(even[j], even[j]);
    even[j] = ptx::addc(0, 0);

    // accumulate "diagonal" |a[i]|*|a[i]| product
    i = 0;
    even[2 * i] = ptx::mad_lo_cc(a[i], a[i], even[2 * i]);
    even[2 * i + 1] = ptx::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
    for (++i; i < TLC; i++) {
      even[2 * i] = ptx::madc_lo_cc(a[i], a[i], even[2 * i]);
      even[2 * i + 1] = ptx::madc_hi_cc(a[i], a[i], even[2 * i + 1]);
    }
  }

  static DEVICE_FORCEINLINE void mul_by_1_row(uint32_t *even, uint32_t *odd, bool first = false) {
    uint32_t mi;
    constexpr auto modulus = CONFIG::modulus;
    const uint32_t *const MOD = modulus.limbs;
    if (first) {
      mi = even[0] * INV;
      mul_n(odd, MOD + 1, mi);
      cmad_n(even, MOD, mi);
      odd[TLC - 1] = ptx::addc(odd[TLC - 1], 0);
    } else {
      even[0] = ptx::add_cc(even[0], odd[1]);
      // we trust the compiler to *not* touch the carry flag here
      // this code sits in between two "asm volatile" instructions which should guarantee that nothing else interferes with the carry flag
      mi = even[0] * INV;
      madc_n_rshift(odd, MOD + 1, mi);
      cmad_n(even, MOD, mi);
      odd[TLC - 1] = ptx::addc(odd[TLC - 1], 0);
    }
  }

  // Performs Montgomery reduction on a storage_wide input. Input value must be in the range [0, mod*2^(32*TLC)).
  // Does not implement an in-place reduce<REDUCTION_SIZE> epilogue. If you want to further reduce the result,
  // call reduce<whatever>(xs.get_lo()) after the call to redc_wide_inplace.
  static DEVICE_FORCEINLINE void redc_wide_inplace(storage_wide &xs) {
    uint32_t *even = xs.limbs;
    // Yields montmul of lo TLC limbs * 1.
    // Since the hi TLC limbs don't participate in computing the "mi" factor at each mul-and-rightshift stage,
    // it's ok to ignore the hi TLC limbs during this process and just add them in afterward.
    uint32_t odd[TLC];
    size_t i;
#pragma unroll
    for (i = 0; i < TLC; i += 2) {
      mul_by_1_row(&even[0], &odd[0], i == 0);
      mul_by_1_row(&odd[0], &even[0]);
    }
    even[0] = ptx::add_cc(even[0], odd[1]);
#pragma unroll
    for (i = 1; i < TLC - 1; i++)
      even[i] = ptx::addc_cc(even[i], odd[i + 1]);
    even[i] = ptx::addc(even[i], 0);
    // Adds in (hi TLC limbs), implicitly right-shifting them by TLC limbs as if they had participated in the
    // add-and-rightshift stages above.
    xs.limbs[0] = ptx::add_cc(xs.limbs[0], xs.limbs[TLC]);
#pragma unroll
    for (i = 1; i < TLC - 1; i++)
      xs.limbs[i] = ptx::addc_cc(xs.limbs[i], xs.limbs[i + TLC]);
    xs.limbs[TLC - 1] = ptx::addc(xs.limbs[TLC - 1], xs.limbs[2 * TLC - 1]);
  }

  static DEVICE_FORCEINLINE void montmul_raw(const storage &a_in, const storage &b_in, storage &r_in) {
    constexpr uint32_t n = TLC;
    const uint32_t *a = a_in.limbs;
    const uint32_t *b = b_in.limbs;
    uint32_t *even = r_in.limbs;
    __align__(8) uint32_t odd[n + 1];
    size_t i;
#pragma unroll
    for (i = 0; i < n; i += 2) {
      mad_n_redc(&even[0], &odd[0], a, b[i], i == 0);
      mad_n_redc(&odd[0], &even[0], a, b[i + 1]);
    }
    // merge |even| and |odd|
    even[0] = ptx::add_cc(even[0], odd[1]);
#pragma unroll
    for (i = 1; i < n - 1; i++)
      even[i] = ptx::addc_cc(even[i], odd[i + 1]);
    even[i] = ptx::addc(even[i], 0);
    // final reduction from [0, 2*mod) to [0, mod) not done here, instead performed optionally in mul_device wrapper
  }

  // Returns xs * ys without Montgomery reduction.
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage_wide mul_wide(const storage &xs, const storage &ys) {
    // Forces us to think more carefully about the last carry bit if we use a modulus with fewer than 2 leading zeroes of slack
    static_assert(!(CONFIG::modulus.limbs[TLC - 1] >> 30));
    storage_wide rs = {0};
    multiply_raw(xs, ys, rs);
    return reduce_wide<REDUCTION_SIZE>(rs);
  }

  // Performs Montgomery reduction on a storage_wide input. Input value must be in the range [0, mod*2^(32*TLC)).
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage redc_wide(const storage_wide &xs) {
    storage_wide tmp{xs};
    redc_wide_inplace(tmp); // after reduce_twopass, tmp's low TLC limbs should represent a value in [0, 2*mod)
    return reduce<REDUCTION_SIZE>(tmp.get_lo());
  }

  // return xs * ys with field operands
  // Device path adapts http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf to use IMAD.WIDE.
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage mul(const storage &xs, const storage &ys) {
    // Forces us to think more carefully about the last carry bit if we use a modulus with fewer than 2 leading zeroes of slack
    static_assert(!(CONFIG::modulus.limbs[TLC - 1] >> 30));
    storage rs = {0};
    montmul_raw(xs, ys, rs);
    return reduce<REDUCTION_SIZE>(rs);
  }

  // return xs^2 with field operands
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage sqr(const storage &xs) {
    // Forces us to think more carefully about the last carry bit if we use a modulus with fewer than 2 leading zeroes of slack
    static_assert(!(CONFIG::modulus.limbs[TLC - 1] >> 30));
    storage_wide rs = {0};
    sqr_raw(xs, rs);
    redc_wide_inplace(rs); // after reduce_twopass, tmp's low TLC limbs should represent a value in [0, 2*mod)
    return reduce<REDUCTION_SIZE>(rs.get_lo());
  }

  // convert field to montgomery form
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage to_montgomery(const storage &xs) {
    constexpr storage r2 = CONFIG::r2;
    return mul<REDUCTION_SIZE>(xs, r2);
  }

  // convert field from montgomery form
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage from_montgomery(const storage &xs) { return mul<REDUCTION_SIZE>(xs, {1}); }

  // return 2*x with field operands
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage dbl(const storage &xs) {
    const uint32_t *x = xs.limbs;
    storage rs = {};
    uint32_t *r = rs.limbs;
    r[0] = x[0] << 1;
#pragma unroll
    for (unsigned i = 1; i < TLC; i++)
      r[i] = __funnelshift_r(x[i - 1], x[i], 31);
    return reduce<REDUCTION_SIZE>(rs);
  }

  // return x/2 with field operands
  template <unsigned REDUCTION_SIZE = 1> static DEVICE_FORCEINLINE storage div2(const storage &xs) {
    const uint32_t *x = xs.limbs;
    storage rs = {};
    uint32_t *r = rs.limbs;
#pragma unroll
    for (unsigned i = 0; i < TLC - 1; i++)
      r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
    r[TLC - 1] = x[TLC - 1] >> 1;
    return reduce<REDUCTION_SIZE>(rs);
  }

  // return -xs with field operand
  template <unsigned MODULUS_SIZE = 1> static DEVICE_FORCEINLINE storage neg(const storage &xs) {
    const storage modulus = get_modulus<MODULUS_SIZE>();
    storage rs = {};
    sub_limbs<false>(modulus, xs, rs);
    return rs;
  }

  // extract a given count of bits at a given offset from the field
  static DEVICE_FORCEINLINE uint32_t extract_bits(const storage &xs, const unsigned offset, const unsigned count) {
    const unsigned limb_index = offset / warpSize;
    const uint32_t *x = xs.limbs;
    const uint32_t low_limb = x[limb_index];
    const uint32_t high_limb = limb_index < (TLC - 1) ? x[limb_index + 1] : 0;
    uint32_t result = __funnelshift_r(low_limb, high_limb, offset);
    result &= (1 << count) - 1;
    return result;
  }

  template <unsigned REDUCTION_SIZE = 1, unsigned LAST_REDUCTION_SIZE = REDUCTION_SIZE>
  static DEVICE_FORCEINLINE storage mul(const unsigned scalar, const storage &xs) {
    storage rs = {};
    storage temp = xs;
    unsigned l = scalar;
    bool is_zero = true;
#pragma unroll
    for (unsigned i = 0; i < 32; i++) {
      if (l & 1) {
        rs = is_zero ? temp : (l >> 1) ? add<REDUCTION_SIZE>(rs, temp) : add<LAST_REDUCTION_SIZE>(rs, temp);
        is_zero = false;
      }
      l >>= 1;
      if (l == 0)
        break;
      temp = dbl<REDUCTION_SIZE>(temp);
    }
    return rs;
  }

  static DEVICE_FORCEINLINE bool is_odd(const storage &xs) { return xs.limbs[0] & 1; }

  static DEVICE_FORCEINLINE bool is_even(const storage &xs) { return ~xs.limbs[0] & 1; }

  static DEVICE_FORCEINLINE bool lt(const storage &xs, const storage &ys) {
    storage dummy = {};
    uint32_t carry = sub_limbs<true>(xs, ys, dummy);
    return carry;
  }

  static DEVICE_FORCEINLINE storage inverse(const storage &xs) {
    if (is_zero(xs))
      return xs;
    constexpr storage one = {1};
    constexpr storage modulus = CONFIG::modulus;
    storage u = xs;
    storage v = modulus;
    storage b = CONFIG::r2;
    storage c = {};
    while (!eq(u, one) && !eq(v, one)) {
      while (is_even(u)) {
        u = div2(u);
        if (is_odd(b))
          add_limbs<false>(b, modulus, b);
        b = div2(b);
      }
      while (is_even(v)) {
        v = div2(v);
        if (is_odd(c))
          add_limbs<false>(c, modulus, c);
        c = div2(c);
      }
      if (lt(v, u)) {
        sub_limbs<false>(u, v, u);
        b = sub(b, c);
      } else {
        sub_limbs<false>(v, u, v);
        c = sub(c, b);
      }
    }
    return eq(u, one) ? b : c;
  }
};

typedef ff<config_fr, inv_fr> fr;

} // namespace bn254