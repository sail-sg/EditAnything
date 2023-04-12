#include <cuda.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <typename T>
__host__ __device__ __forceinline__ T THCRoundUp(T a, T b) {
  return THCCeilDiv(a, b) * b;
}
