#pragma once

#include "constants.cuh"
#include "math_utils.cuh"
#include <cuda_runtime.h>

// Point addition in Jacobian coordinates (mixed: z1 jacobian, z2=1 affine)
// Based on reference implementation from inc_ecc_secp256k1.cl
__device__ void point_add(uint32_t* x1, uint32_t* y1, uint32_t* z1,
                         const uint32_t* x2, const uint32_t* y2) {
    // Copy input values to temporary variables (t1=x1, t2=y1, t3=z1, t4=x2, t5=y2)
    uint32_t t1[8], t2[8], t3[8], t4[8], t5[8];
    uint32_t t6[8], t7[8], t8[8], t9[8];
    
    // Copy input points
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t1[i] = x1[i];  // t1 = x1
        t2[i] = y1[i];  // t2 = y1  
        t3[i] = z1[i];  // t3 = z1
        t4[i] = x2[i];  // t4 = x2
        t5[i] = y2[i];  // t5 = y2
    }
    
    mul_mod(t6, t3, t3);    // t6 = t3^2 = z1^2
    mul_mod(t7, t6, t3);    // t7 = t6*t3 = z1^3
    mul_mod(t6, t6, t4);    // t6 = t6*t4 = z1^2 * x2
    mul_mod(t7, t7, t5);    // t7 = t7*t5 = z1^3 * y2
    
    sub_mod(t6, t6, t1);    // t6 = t6-t1 = z1^2*x2 - x1
    sub_mod(t7, t7, t2);    // t7 = t7-t2 = z1^3*y2 - y1
    
    mul_mod(t8, t3, t6);    // t8 = t3*t6 = z1*(z1^2*x2 - x1)
    mul_mod(t4, t6, t6);    // t4 = t6^2 = (z1^2*x2 - x1)^2
    mul_mod(t9, t4, t6);    // t9 = t4*t6 = (z1^2*x2 - x1)^3
    mul_mod(t4, t4, t1);    // t4 = t4*t1 = (z1^2*x2 - x1)^2 * x1
    
    // Left shift (t4 * 2) with overflow handling
    t6[7] = (t4[7] << 1) | (t4[6] >> 31);
    t6[6] = (t4[6] << 1) | (t4[5] >> 31);
    t6[5] = (t4[5] << 1) | (t4[4] >> 31);
    t6[4] = (t4[4] << 1) | (t4[3] >> 31);
    t6[3] = (t4[3] << 1) | (t4[2] >> 31);
    t6[2] = (t4[2] << 1) | (t4[1] >> 31);
    t6[1] = (t4[1] << 1) | (t4[0] >> 31);
    t6[0] = t4[0] << 1;
    
    // Handle most significant bit overflow
    if (t4[7] & 0x80000000) {
        uint32_t a[8] = {0x000003d1, 1, 0, 0, 0, 0, 0, 0}; // omega value for mod P
        add(t6, t6, a);
    }
    
    mul_mod(t5, t7, t7);    // t5 = t7*t7 = (z1^3*y2 - y1)^2
    sub_mod(t5, t5, t6);    // t5 = t5-t6
    sub_mod(t5, t5, t9);    // t5 = t5-t9
    sub_mod(t4, t4, t5);    // t4 = t4-t5
    
    mul_mod(t4, t4, t7);    // t4 = t4*t7
    mul_mod(t9, t9, t2);    // t9 = t9*t2
    sub_mod(t9, t4, t9);    // t9 = t4-t9
    
    // Store results back to x1, y1, z1
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x1[i] = t5[i];  // x1 = t5
        y1[i] = t9[i];  // y1 = t9
        z1[i] = t8[i];  // z1 = t8
    }
}

// Point doubling in Jacobian coordinates
__device__ void point_double(uint32_t* x3, uint32_t* y3, uint32_t* z3,
                            const uint32_t* x1, const uint32_t* y1, const uint32_t* z1) {
    uint32_t s[8], m[8], t[8];
    
    // s = 4 * x1 * y1^2
    uint32_t y1y1[8];
    mul_mod(y1y1, y1, y1);
    mul_mod(s, x1, y1y1);
    add_mod(s, s, s);
    add_mod(s, s, s);
    
    // m = 3 * x1^2 (for a=0 curve)
    uint32_t x1x1[8];
    mul_mod(x1x1, x1, x1);
    add_mod(m, x1x1, x1x1);
    add_mod(m, m, x1x1);
    
    // t = m^2 - 2*s
    mul_mod(t, m, m);
    sub_mod(t, t, s);
    sub_mod(x3, t, s);
    
    // y3 = m * (s - x3) - 8 * y1^4
    sub_mod(t, s, x3);
    mul_mod(t, m, t);
    mul_mod(y1y1, y1y1, y1y1);
    add_mod(y1y1, y1y1, y1y1);
    add_mod(y1y1, y1y1, y1y1);
    add_mod(y1y1, y1y1, y1y1);
    sub_mod(y3, t, y1y1);
    
    // z3 = 2 * y1 * z1
    mul_mod(z3, y1, z1);
    add_mod(z3, z3, z3);
}

// Convert Jacobian to affine coordinates
__device__ void jacobian_to_affine(uint32_t* x_affine, uint32_t* y_affine,
                                   const uint32_t* x, const uint32_t* y, const uint32_t* z) {
    uint32_t z_inv[8], z_inv2[8], z_inv3[8];
    
    // z_inv = 1/z
    inv_mod(z_inv, z);
    
    // z_inv2 = z_inv^2
    mul_mod(z_inv2, z_inv, z_inv);
    
    // x_affine = x * z_inv2
    mul_mod(x_affine, x, z_inv2);
    
    // z_inv3 = z_inv2 * z_inv
    mul_mod(z_inv3, z_inv2, z_inv);
    
    // y_affine = y * z_inv3
    mul_mod(y_affine, y, z_inv3);
}

// Scalar multiplication using double-and-add method
__device__ void point_mul(uint32_t* x_out, uint32_t* y_out, const uint32_t* k) {
    uint32_t x[8] = {0}, y[8] = {0}, z[8] = {0};
    bool first = true;
    
    // Double-and-add algorithm - process from MSB to LSB
    for (int bit = 255; bit >= 0; bit--) {
        // For little-endian storage: k[0] = bits 0-31, k[1] = bits 32-63, etc.
        int limb_idx = 7 - (bit >> 5);  // Reverse limb order for little-endian
        int bit_idx = bit & 31;
        
        if (!first) {
            point_double(x, y, z, x, y, z);
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
    if (first) {
        // Point at infinity (scalar was zero)
        for (int i = 0; i < 8; i++) {
            x_out[i] = 0;
            y_out[i] = 0;
        }
    } else {
        jacobian_to_affine(x_out, y_out, x, y, z);
    }
}

// Convert scalar to w-NAF (window size is 4)
__device__ int convert_to_window_naf(uint32_t* naf, const uint32_t* k) {
    int loop_start = 0;
    
    uint32_t n[9];
    
    n[0] = 0; // extra slot for subtraction
    n[1] = k[7];
    n[2] = k[6];
    n[3] = k[5];
    n[4] = k[4];
    n[5] = k[3];
    n[6] = k[2];
    n[7] = k[1];
    n[8] = k[0];
    
    for (int i = 0; i <= 256; i++) {
        if (n[8] & 1) {
            int diff = n[8] & 0x0f; // n % 2^w == n & (2^w - 1)
            
            int val = diff;
            
            if (diff >= 0x08) {
                diff -= 0x10;
                val = 0x11 - val;
            }
            
            naf[i >> 3] |= val << ((i & 7) << 2);
            
            uint32_t t = n[8]; // old/unmodified value
            
            n[8] -= diff;
            
            // Handle carry/borrow
            uint32_t k = 8;
            
            if (diff > 0) {
                while (n[k] > t) { // overflow propagation
                    if (k == 0) break;
                    k--;
                    t = n[k];
                    n[k]--;
                }
            } else { // if (diff < 0)
                while (t > n[k]) { // overflow propagation
                    if (k == 0) break;
                    k--;
                    t = n[k];
                    n[k]++;
                }
            }
            
            loop_start = i;
        }
        
        // n = n / 2
        n[8] = n[8] >> 1 | n[7] << 31;
        n[7] = n[7] >> 1 | n[6] << 31;
        n[6] = n[6] >> 1 | n[5] << 31;
        n[5] = n[5] >> 1 | n[4] << 31;
        n[4] = n[4] >> 1 | n[3] << 31;
        n[3] = n[3] >> 1 | n[2] << 31;
        n[2] = n[2] >> 1 | n[1] << 31;
        n[1] = n[1] >> 1 | n[0] << 31;
        n[0] = n[0] >> 1;
    }
    
    return loop_start;
}

// Point multiplication using w-NAF method with precomputed basepoint
__device__ void point_mul_xy(uint32_t* x1, uint32_t* y1, const uint32_t* k, const secp256k1_t* tmps) {
    uint32_t naf[SECP256K1_NAF_SIZE] = {0};
    
    int loop_start = convert_to_window_naf(naf, k);
    
    // First set
    const uint32_t multiplier = (naf[loop_start >> 3] >> ((loop_start & 7) << 2)) & 0x0f;
    
    const uint32_t odd = multiplier & 1;
    
    const uint32_t x_pos = ((multiplier - 1 + odd) >> 1) * 24;
    const uint32_t y_pos = odd ? (x_pos + 8) : (x_pos + 16);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x1[i] = tmps->xy[x_pos + i];
        y1[i] = tmps->xy[y_pos + i];
    }
    
    uint32_t z1[8] = {0};
    z1[0] = 1;
    
    // Main loop (left-to-right binary algorithm)
    for (int pos = loop_start - 1; pos >= 0; pos--) {
        // Always double
        point_double(x1, y1, z1, x1, y1, z1);
        
        // Add only if needed
        const uint32_t multiplier = (naf[pos >> 3] >> ((pos & 7) << 2)) & 0x0f;
        
        if (multiplier) {
            const uint32_t odd = multiplier & 1;
            
            const uint32_t x_pos = ((multiplier - 1 + odd) >> 1) * 24;
            const uint32_t y_pos = odd ? (x_pos + 8) : (x_pos + 16);
            
            uint32_t x2[8], y2[8];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                x2[i] = tmps->xy[x_pos + i];
                y2[i] = tmps->xy[y_pos + i];
            }
            
            // (x1, y1, z1) + (x2, y2, 1)
            point_add(x1, y1, z1, x2, y2);
        }
    }
    
    // Convert to affine coordinates
    inv_mod(z1, z1);
    
    uint32_t z2[8];
    mul_mod(z2, z1, z1); // z1^2
    mul_mod(x1, x1, z2); // x1_affine
    
    mul_mod(z1, z2, z1); // z1^3
    mul_mod(y1, y1, z1); // y1_affine
}