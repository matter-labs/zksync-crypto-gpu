#include <cuda/std/cstdint>

namespace bn254 {

// definition of variables declared "extern __device__ __constant__" elsewhere
__device__ __constant__ uint32_t inv_fr = 0xefffffff;

} // namespace bn254
