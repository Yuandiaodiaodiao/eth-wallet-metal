#pragma once

#include "constants.cuh"
#include "math_utils.cuh"

// Point addition in Jacobian coordinates (mixed: z1 jacobian, z2=1 affine)
__device__ void point_add(uint32_t* x1, uint32_t* y1, uint32_t* z1, uint32_t* x2, uint32_t* y2) {
    // x1/y1/z1:
    
    uint32_t t1[8];
    
    t1[0] = x1[0];
    t1[1] = x1[1];
    t1[2] = x1[2];
    t1[3] = x1[3];
    t1[4] = x1[4];
    t1[5] = x1[5];
    t1[6] = x1[6];
    t1[7] = x1[7];
    
    uint32_t t2[8];
    
    t2[0] = y1[0];
    t2[1] = y1[1];
    t2[2] = y1[2];
    t2[3] = y1[3];
    t2[4] = y1[4];
    t2[5] = y1[5];
    t2[6] = y1[6];
    t2[7] = y1[7];
    
    uint32_t t3[8];
    
    t3[0] = z1[0];
    t3[1] = z1[1];
    t3[2] = z1[2];
    t3[3] = z1[3];
    t3[4] = z1[4];
    t3[5] = z1[5];
    t3[6] = z1[6];
    t3[7] = z1[7];
    
    // x2/y2:
    
    uint32_t t4[8];
    
    t4[0] = x2[0];
    t4[1] = x2[1];
    t4[2] = x2[2];
    t4[3] = x2[3];
    t4[4] = x2[4];
    t4[5] = x2[5];
    t4[6] = x2[6];
    t4[7] = x2[7];
    
    uint32_t t5[8];
    
    t5[0] = y2[0];
    t5[1] = y2[1];
    t5[2] = y2[2];
    t5[3] = y2[3];
    t5[4] = y2[4];
    t5[5] = y2[5];
    t5[6] = y2[6];
    t5[7] = y2[7];
    
    uint32_t t6[8];
    uint32_t t7[8];
    uint32_t t8[8];
    uint32_t t9[8];
    
    mul_mod(t6, t3, t3); // t6 = t3^2
    
    mul_mod(t7, t6, t3); // t7 = t6*t3
    mul_mod(t6, t6, t4); // t6 = t6*t4
    mul_mod(t7, t7, t5); // t7 = t7*t5
    
    sub_mod(t6, t6, t1); // t6 = t6-t1
    sub_mod(t7, t7, t2); // t7 = t7-t2
    
    mul_mod(t8, t3, t6); // t8 = t3*t6
    mul_mod(t4, t6, t6); // t4 = t6^2
    mul_mod(t9, t4, t6); // t9 = t4*t6
    mul_mod(t4, t4, t1); // t4 = t4*t1
    
    // left shift (t4 * 2):
    
    t6[7] = t4[7] << 1 | t4[6] >> 31;
    t6[6] = t4[6] << 1 | t4[5] >> 31;
    t6[5] = t4[5] << 1 | t4[4] >> 31;
    t6[4] = t4[4] << 1 | t4[3] >> 31;
    t6[3] = t4[3] << 1 | t4[2] >> 31;
    t6[2] = t4[2] << 1 | t4[1] >> 31;
    t6[1] = t4[1] << 1 | t4[0] >> 31;
    t6[0] = t4[0] << 1;
    
    // don't discard the most significant bit, it's important too!
    
    if (t4[7] & 0x80000000)
    {
        // use most significant bit and perform mod P, since we have: t4 * 2 % P
        
        uint32_t a[8] = { 0 };
        
        a[1] = 1;
        a[0] = 0x000003d1; // omega (see: mul_mod ())
        
        add(t6, t6, a);
    }
    
    mul_mod(t5, t7, t7); // t5 = t7*t7
    
    sub_mod(t5, t5, t6); // t5 = t5-t6
    sub_mod(t5, t5, t9); // t5 = t5-t9
    sub_mod(t4, t4, t5); // t4 = t4-t5
    
    mul_mod(t4, t4, t7); // t4 = t4*t7
    mul_mod(t9, t9, t2); // t9 = t9*t2
    
    sub_mod(t9, t4, t9); // t9 = t4-t9
    
    x1[0] = t5[0];
    x1[1] = t5[1];
    x1[2] = t5[2];
    x1[3] = t5[3];
    x1[4] = t5[4];
    x1[5] = t5[5];
    x1[6] = t5[6];
    x1[7] = t5[7];
    
    y1[0] = t9[0];
    y1[1] = t9[1];
    y1[2] = t9[2];
    y1[3] = t9[3];
    y1[4] = t9[4];
    y1[5] = t9[5];
    y1[6] = t9[6];
    y1[7] = t9[7];
    
    z1[0] = t8[0];
    z1[1] = t8[1];
    z1[2] = t8[2];
    z1[3] = t8[3];
    z1[4] = t8[4];
    z1[5] = t8[5];
    z1[6] = t8[6];
    z1[7] = t8[7];
}

// Point doubling in Jacobian coordinates
__device__ void point_double(uint32_t* x, uint32_t* y, uint32_t* z) {
    // Copy input values to temporary variables
    uint32_t t1[8];
    
    t1[0] = x[0];
    t1[1] = x[1];
    t1[2] = x[2];
    t1[3] = x[3];
    t1[4] = x[4];
    t1[5] = x[5];
    t1[6] = x[6];
    t1[7] = x[7];
    
    uint32_t t2[8];
    
    t2[0] = y[0];
    t2[1] = y[1];
    t2[2] = y[2];
    t2[3] = y[3];
    t2[4] = y[4];
    t2[5] = y[5];
    t2[6] = y[6];
    t2[7] = y[7];
    
    uint32_t t3[8];
    
    t3[0] = z[0];
    t3[1] = z[1];
    t3[2] = z[2];
    t3[3] = z[3];
    t3[4] = z[4];
    t3[5] = z[5];
    t3[6] = z[6];
    t3[7] = z[7];
    
    uint32_t t4[8];
    uint32_t t5[8];
    uint32_t t6[8];
    
    mul_mod(t4, t1, t1); // t4 = x^2
    
    mul_mod(t5, t2, t2); // t5 = y^2
    
    mul_mod(t1, t1, t5); // t1 = x*y^2
    
    mul_mod(t5, t5, t5); // t5 = t5^2 = y^4
    
    // here the z^2 and z^4 is not needed for a = 0
    
    mul_mod(t3, t2, t3); // t3 = y * z
    
    add_mod(t2, t4, t4); // t2 = 2 * t4 = 2 * x^2
    add_mod(t4, t4, t2); // t4 = 3 * t4 = 3 * x^2
    
    // a * z^4 = 0 * 1^4 = 0
    
    // don't discard the least significant bit it's important too!
    
    uint32_t c = 0;
    
    if (t4[0] & 1)
    {
        uint32_t t[8];
        
        t[0] = SECP256K1_P0;
        t[1] = SECP256K1_P1;
        t[2] = SECP256K1_P2;
        t[3] = SECP256K1_P3;
        t[4] = SECP256K1_P4;
        t[5] = SECP256K1_P5;
        t[6] = SECP256K1_P6;
        t[7] = SECP256K1_P7;
        
        c = add(t4, t4, t); // t4 + SECP256K1_P
    }
    
    // right shift (t4 / 2):
    
    t4[0] = t4[0] >> 1 | t4[1] << 31;
    t4[1] = t4[1] >> 1 | t4[2] << 31;
    t4[2] = t4[2] >> 1 | t4[3] << 31;
    t4[3] = t4[3] >> 1 | t4[4] << 31;
    t4[4] = t4[4] >> 1 | t4[5] << 31;
    t4[5] = t4[5] >> 1 | t4[6] << 31;
    t4[6] = t4[6] >> 1 | t4[7] << 31;
    t4[7] = t4[7] >> 1 | c << 31;
    
    mul_mod(t6, t4, t4); // t6 = t4^2 = (3/2 * x^2)^2
    
    add_mod(t2, t1, t1); // t2 = 2 * t1
    
    sub_mod(t6, t6, t2); // t6 = t6 - t2
    sub_mod(t1, t1, t6); // t1 = t1 - t6
    
    mul_mod(t4, t4, t1); // t4 = t4 * t1
    
    sub_mod(t1, t4, t5); // t1 = t4 - t5
    
    // => x = t6, y = t1, z = t3:
    
    x[0] = t6[0];
    x[1] = t6[1];
    x[2] = t6[2];
    x[3] = t6[3];
    x[4] = t6[4];
    x[5] = t6[5];
    x[6] = t6[6];
    x[7] = t6[7];
    
    y[0] = t1[0];
    y[1] = t1[1];
    y[2] = t1[2];
    y[3] = t1[3];
    y[4] = t1[4];
    y[5] = t1[5];
    y[6] = t1[6];
    y[7] = t1[7];
    
    z[0] = t3[0];
    z[1] = t3[1];
    z[2] = t3[2];
    z[3] = t3[3];
    z[4] = t3[4];
    z[5] = t3[5];
    z[6] = t3[6];
    z[7] = t3[7];
}

// Convert Jacobian to affine coordinates
__device__ void jacobian_to_affine(uint32_t* x_affine, uint32_t* y_affine,
                                   const uint32_t* x, const uint32_t* y, const uint32_t* z) {
    uint32_t z_inv[8], z_inv2[8], z_inv3[8];
    
    // z_inv = 1/z
    for (int i = 0; i < 8; i++) {
        z_inv[i] = z[i];
    }
    inv_mod(z_inv);
    
    // z_inv2 = z_inv^2
    mul_mod(z_inv2, z_inv, z_inv);
    
    // x_affine = x * z_inv2
    mul_mod(x_affine, x, z_inv2);
    
    // z_inv3 = z_inv2 * z_inv
    mul_mod(z_inv3, z_inv2, z_inv);
    
    // y_affine = y * z_inv3
    mul_mod(y_affine, y, z_inv3);
}

// Simple scalar multiplication using double-and-add method (for reference/testing)
__device__ void point_mul_simple(uint32_t* x_out, uint32_t* y_out, const uint32_t* k) {
    uint32_t x[8] = {0}, y[8] = {0}, z[8] = {0};
    bool first = true;
    
    // Double-and-add algorithm - process from MSB to LSB
    for (int bit = 255; bit >= 0; bit--) {
        // For little-endian storage: k[0] = bits 0-31, k[1] = bits 32-63, etc.
        int limb_idx = 7 - (bit >> 5);  // Reverse limb order for little-endian
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
    
    // first set:
    const uint32_t multiplier = (naf[loop_start >> 3] >> ((loop_start & 7) << 2)) & 0x0f;
    
    const uint32_t odd = multiplier & 1;
    
    const uint32_t x_pos = ((multiplier - 1 + odd) >> 1) * 24;
    const uint32_t y_pos = odd ? (x_pos + 8) : (x_pos + 16);
    
    x1[0] = tmps->xy[x_pos + 0];
    x1[1] = tmps->xy[x_pos + 1];
    x1[2] = tmps->xy[x_pos + 2];
    x1[3] = tmps->xy[x_pos + 3];
    x1[4] = tmps->xy[x_pos + 4];
    x1[5] = tmps->xy[x_pos + 5];
    x1[6] = tmps->xy[x_pos + 6];
    x1[7] = tmps->xy[x_pos + 7];

    y1[0] = tmps->xy[y_pos + 0];
    y1[1] = tmps->xy[y_pos + 1];
    y1[2] = tmps->xy[y_pos + 2];
    y1[3] = tmps->xy[y_pos + 3];
    y1[4] = tmps->xy[y_pos + 4];
    y1[5] = tmps->xy[y_pos + 5];
    y1[6] = tmps->xy[y_pos + 6];
    y1[7] = tmps->xy[y_pos + 7];

    uint32_t z1[8] = {0};
    
    z1[0] = 1;
    
    // main loop (left-to-right binary algorithm):
    for (int pos = loop_start - 1; pos >= 0; pos--) // -1 because we've set/add the point already
    {
        // always double:
        point_double(x1, y1, z1);
        
        // add only if needed:
        const uint32_t multiplier = (naf[pos >> 3] >> ((pos & 7) << 2)) & 0x0f;
        
        if (multiplier)
        {
            const uint32_t odd = multiplier & 1;
            
            const uint32_t x_pos = ((multiplier - 1 + odd) >> 1) * 24;
            const uint32_t y_pos = odd ? (x_pos + 8) : (x_pos + 16);
            
            uint32_t x2[8];
            
            x2[0] = tmps->xy[x_pos + 0];
            x2[1] = tmps->xy[x_pos + 1];
            x2[2] = tmps->xy[x_pos + 2];
            x2[3] = tmps->xy[x_pos + 3];
            x2[4] = tmps->xy[x_pos + 4];
            x2[5] = tmps->xy[x_pos + 5];
            x2[6] = tmps->xy[x_pos + 6];
            x2[7] = tmps->xy[x_pos + 7];

            uint32_t y2[8];

            y2[0] = tmps->xy[y_pos + 0];
            y2[1] = tmps->xy[y_pos + 1];
            y2[2] = tmps->xy[y_pos + 2];
            y2[3] = tmps->xy[y_pos + 3];
            y2[4] = tmps->xy[y_pos + 4];
            y2[5] = tmps->xy[y_pos + 5];
            y2[6] = tmps->xy[y_pos + 6];
            y2[7] = tmps->xy[y_pos + 7];
            
            // (x1, y1, z1) + multiplier * (x, y, z) = (x1, y1, z1) + (x2, y2, z2)
            point_add(x1, y1, z1, x2, y2);
        }
    }
    
    /*
     * Get the corresponding affine coordinates x/y:
     */
    
    inv_mod(z1);
    
    uint32_t z2[8];
    
    mul_mod(z2, z1, z1); // z1^2
    mul_mod(x1, x1, z2); // x1_affine
    
    mul_mod(z1, z2, z1); // z1^3
    mul_mod(y1, y1, z1); // y1_affine
    
    // return values are already in x1 and y1
}