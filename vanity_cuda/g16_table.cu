// CUDA G16 Table Builder
// Separated from main kernels for modular loading

#include "include/secp256k1.cuh"

// G16 table builder kernel
extern "C" __global__ void build_g16_table_kernel(
    uint8_t* __restrict__ g16_table,
    uint32_t window,
    uint32_t start_idx,
    uint32_t count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t idx = start_idx + tid;
    if (idx >= 65536) return;
    
    uint64_t offset = ((uint64_t)window * 65536 + idx) * 64;
    uint8_t* out = g16_table + offset;
    
    if (idx == 0) {
        // Zero point
        for (int i = 0; i < 64; i++) {
            out[i] = 0;
        }
        return;
    }
    
    // Build scalar k = idx << (16 * window)
    uint32_t k[8] = {0};
    uint32_t shift = window * 16;
    uint32_t limb = shift >> 5;
    uint32_t rem = shift & 31;
    
    uint64_t wide = ((uint64_t)idx) << rem;
    k[limb] = (uint32_t)(wide & 0xFFFFFFFF);
    if (limb + 1 < 8) {
        k[limb + 1] = (uint32_t)(wide >> 32);
    }
    
    // Compute point multiplication using double-and-add
    uint32_t x[8] = {0}, y[8] = {0}, z[8] = {0};
    bool first = true;
    
    for (int bit = 255; bit >= 0; bit--) {
        int limb_idx = bit >> 5;
        int bit_idx = bit & 31;
        
        if (!first) {
            point_double(x, y, z);
        }
        
        if ((k[limb_idx] >> bit_idx) & 1) {
            if (first) {
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    x[i] = SECP256K1_G[i];
                    y[i] = SECP256K1_G[8 + i];
                    z[i] = (i == 0) ? 1 : 0;
                }
                first = false;
            } else {
                point_add(x, y, z, SECP256K1_G, SECP256K1_G + 8);
            }
        }
    }
    
    // Convert to affine
    uint32_t x_affine[8], y_affine[8];
    if (first) {
        // Point at infinity
        for (int i = 0; i < 8; i++) {
            x_affine[i] = 0;
            y_affine[i] = 0;
        }
    } else {
        jacobian_to_affine(x_affine, y_affine, x, y, z);
    }
    
    // Write to table (little-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4] = x_affine[i] & 0xFF;
        out[i*4 + 1] = (x_affine[i] >> 8) & 0xFF;
        out[i*4 + 2] = (x_affine[i] >> 16) & 0xFF;
        out[i*4 + 3] = (x_affine[i] >> 24) & 0xFF;
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[32 + i*4] = y_affine[i] & 0xFF;
        out[32 + i*4 + 1] = (y_affine[i] >> 8) & 0xFF;
        out[32 + i*4 + 2] = (y_affine[i] >> 16) & 0xFF;
        out[32 + i*4 + 3] = (y_affine[i] >> 24) & 0xFF;
    }
}