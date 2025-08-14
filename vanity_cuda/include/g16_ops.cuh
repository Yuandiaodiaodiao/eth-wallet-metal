#pragma once

#include "constants.cuh"
#include "ec_ops.cuh"
#include <cuda_runtime.h>

// G16 precomputed table structure (in constant memory)
struct G16Table {
    // 16 windows * 65536 entries * 64 bytes per entry
    // Stored in little-endian format
    uint8_t data[16 * 65536 * 64];
};

// Get 16-bit window value from scalar
__device__ __forceinline__ uint32_t get_window16(const uint32_t* k, uint32_t window) {
    uint32_t bit = window * 16;
    uint32_t limb = bit >> 5;
    uint32_t shift = bit & 31;
    
    if (shift <= 16) {
        return (k[limb] >> shift) & 0xFFFF;
    } else {
        uint32_t low = k[limb] >> shift;
        uint32_t high = k[limb + 1] << (32 - shift);
        return (low | high) & 0xFFFF;
    }
}

// Load point from G16 table
__device__ __forceinline__ void load_point_g16(uint32_t* x, uint32_t* y,
                                               const uint8_t* g16_table,
                                               uint32_t window, uint32_t idx) {
    if (idx == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            x[i] = 0;
            y[i] = 0;
        }
        return;
    }
    
    size_t offset = ((size_t)window * 65536 + idx) * 64;
    const uint8_t* ptr = g16_table + offset;
    
    // Load x coordinate (little-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x[i] = ((uint32_t)ptr[i*4 + 0]) |
               ((uint32_t)ptr[i*4 + 1] << 8) |
               ((uint32_t)ptr[i*4 + 2] << 16) |
               ((uint32_t)ptr[i*4 + 3] << 24);
    }
    
    // Load y coordinate (little-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        y[i] = ((uint32_t)ptr[32 + i*4 + 0]) |
               ((uint32_t)ptr[32 + i*4 + 1] << 8) |
               ((uint32_t)ptr[32 + i*4 + 2] << 16) |
               ((uint32_t)ptr[32 + i*4 + 3] << 24);
    }
}

// Scalar multiplication using 16-bit window method with G16 table
__device__ void point_mul_g16(uint32_t* x_out, uint32_t* y_out,
                              const uint32_t* k, const uint8_t* g16_table) {
    uint32_t x[8], y[8], z[8];
    bool first = true;
    
    // Process 16 windows of 16 bits each
    #pragma unroll 1
    for (int window = 0; window < 16; window++) {
        uint32_t idx = get_window16(k, window);
        
        if (idx == 0) continue;
        
        uint32_t x2[8], y2[8];
        load_point_g16(x2, y2, g16_table, window, idx);
        
        if (first) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                x[i] = x2[i];
                y[i] = y2[i];
                z[i] = (i == 0) ? 1 : 0;
            }
            first = false;
        } else {
            point_add(x, y, z, x2, y2);
        }
    }
    
    // Convert to affine coordinates (manual implementation like Metal version)
    uint32_t z_inv[8];
    for (int i = 0; i < 8; i++) {
        z_inv[i] = z[i];
    }
    inv_mod(z_inv);                    // z_inv = 1/z
    uint32_t z2[8]; 
    mul_mod(z2, z_inv, z_inv);            // z2 = z_inv^2 = 1/z^2
    mul_mod(x, x, z2);                    // x = x * 1/z^2
    mul_mod(z2, z2, z_inv);               // z2 = z2 * z_inv = 1/z^3
    mul_mod(y, y, z2);                    // y = y * 1/z^3
    
    // Copy results
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x_out[i] = x[i];
        y_out[i] = y[i];
    }
}

// Batch inversion using Montgomery's trick
__device__ void batch_inverse(uint32_t xs[][8], uint32_t ys[][8], uint32_t zs[][8],
                              uint32_t temp_pref[][8], int batch_size) {
    // Calculate prefix products
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        temp_pref[0][i] = zs[0][i];
    }
    
    for (int i = 1; i < batch_size; i++) {
        mul_mod(temp_pref[i], temp_pref[i-1], zs[i]);
    }
    
    // Invert the total product
    uint32_t inv_total[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        inv_total[j] = temp_pref[batch_size - 1][j];
    }
    inv_mod(inv_total);
    // Backward pass to get individual inverses and convert to affine
    for (int i = batch_size - 1; i >= 0; i--) {
        uint32_t inv_z[8];
        
        if (i == 0) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                inv_z[j] = inv_total[j];
            }
        } else {
            mul_mod(inv_z, inv_total, temp_pref[i - 1]);
        }
        
        // Update inv_total for next iteration
        if (i > 0) {
            mul_mod(inv_total, inv_total, zs[i]);
        }
        
        // Convert to affine: x = x * z^-2, y = y * z^-3
        uint32_t z_inv2[8], z_inv3[8];
        mul_mod(z_inv2, inv_z, inv_z);
        mul_mod(z_inv3, z_inv2, inv_z);
        
        uint32_t x_affine[8], y_affine[8];
        mul_mod(x_affine, xs[i], z_inv2);
        mul_mod(y_affine, ys[i], z_inv3);
        
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            xs[i][j] = x_affine[j];
            ys[i][j] = y_affine[j];
        }
    }
}