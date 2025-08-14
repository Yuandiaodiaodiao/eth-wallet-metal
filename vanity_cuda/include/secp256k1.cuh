#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include "constants.cuh"
#include "math_utils.cuh"
#include "ec_ops.cuh"
#include "g16_ops.cuh"
#include <cuda_runtime.h>

// High-level interface functions for public key operations
__device__ uint32_t transform_public(secp256k1_t* r, const uint32_t* x, const uint32_t first_byte);
__device__ uint32_t parse_public(secp256k1_t* r, const uint32_t* k);
__device__ void point_mul_xy(uint32_t* x1, uint32_t* y1, const uint32_t* k, const secp256k1_t* tmps);
__device__ void point_mul(uint32_t* r, const uint32_t* k, const secp256k1_t* tmps);

#endif // SECP256K1_CUH