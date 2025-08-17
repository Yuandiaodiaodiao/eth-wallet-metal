#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include "constants.cuh"
#include "math_utils.cuh"
#include "ec_ops.cuh"
#include "g16_ops.cuh"

// High-level interface functions for public key operations
__device__ void point_mul_xy(uint32_t* x1, uint32_t* y1, const uint32_t* k, const secp256k1_t* tmps);

#endif // SECP256K1_CUH