#include "secp256k1.cuh"

// Modular square root using Fermat's little theorem
__device__ void sqrt_mod(uint32_t* r) {
    uint32_t s[8];
    s[0] = SECP256K1_P0 + 1;
    s[1] = SECP256K1_P1;
    s[2] = SECP256K1_P2;
    s[3] = SECP256K1_P3;
    s[4] = SECP256K1_P4;
    s[5] = SECP256K1_P5;
    s[6] = SECP256K1_P6;
    s[7] = SECP256K1_P7;
    
    uint32_t t[8] = {0};
    t[0] = 1;
    
    for (uint32_t i = 255; i > 1; i--) {
        mul_mod(t, t, t);
        
        uint32_t idx = i >> 5;
        uint32_t mask = 1 << (i & 0x1f);
        
        if (s[idx] & mask) {
            mul_mod(t, t, r);
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = t[i];
    }
}

// Get precomputed coordinates for w-NAF
__device__ void point_get_coords(secp256k1_t* r, const uint32_t* x, const uint32_t* y) {
    uint32_t p[8];
    p[0] = SECP256K1_P0;
    p[1] = SECP256K1_P1;
    p[2] = SECP256K1_P2;
    p[3] = SECP256K1_P3;
    p[4] = SECP256K1_P4;
    p[5] = SECP256K1_P5;
    p[6] = SECP256K1_P6;
    p[7] = SECP256K1_P7;
    
    // Store x1, y1
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[i] = x[i];
        r->xy[8 + i] = y[i];
    }
    
    // Store -y1
    uint32_t neg[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        neg[i] = y[i];
    }
    sub_mod(neg, p, neg);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[16 + i] = neg[i];
    }
    
    // Compute 3*P, 5*P, 7*P
    uint32_t tx[8], ty[8], rx[8], ry[8], rz[8];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        tx[i] = x[i];
        ty[i] = y[i];
        rx[i] = x[i];
        ry[i] = y[i];
    }
    
    rz[0] = 1;
    for (int i = 1; i < 8; i++) rz[i] = 0;
    
    // 3*P
    point_double(rx, ry, rz);
    point_add(rx, ry, rz, tx, ty);
    
    // Convert to affine
    jacobian_to_affine(rx, ry, rx, ry, rz);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[24 + i] = rx[i];
        r->xy[32 + i] = ry[i];
    }
    
    // Store -y3
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        neg[i] = ry[i];
    }
    sub_mod(neg, p, neg);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[40 + i] = neg[i];
    }
    
    // 5*P = 3*P + 2*P
    rz[0] = 1;
    for (int i = 1; i < 8; i++) rz[i] = 0;
    
    point_add(rx, ry, rz, tx, ty);
    point_add(rx, ry, rz, tx, ty);
    
    jacobian_to_affine(rx, ry, rx, ry, rz);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[48 + i] = rx[i];
        r->xy[56 + i] = ry[i];
    }
    
    // Store -y5
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        neg[i] = ry[i];
    }
    sub_mod(neg, p, neg);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[64 + i] = neg[i];
    }
    
    // 7*P = 5*P + 2*P
    rz[0] = 1;
    for (int i = 1; i < 8; i++) rz[i] = 0;
    
    point_add(rx, ry, rz, tx, ty);
    point_add(rx, ry, rz, tx, ty);
    
    jacobian_to_affine(rx, ry, rx, ry, rz);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[72 + i] = rx[i];
        r->xy[80 + i] = ry[i];
    }
    
    // Store -y7
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        neg[i] = ry[i];
    }
    sub_mod(neg, p, neg);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[88 + i] = neg[i];
    }
}

// Transform a public key from compressed format
__device__ uint32_t transform_public(secp256k1_t* r, const uint32_t* x, const uint32_t first_byte) {
    uint32_t p[8];
    p[0] = SECP256K1_P0;
    p[1] = SECP256K1_P1;
    p[2] = SECP256K1_P2;
    p[3] = SECP256K1_P3;
    p[4] = SECP256K1_P4;
    p[5] = SECP256K1_P5;
    p[6] = SECP256K1_P6;
    p[7] = SECP256K1_P7;
    
    // Check if x is valid (x < p)
    for (int i = 7; i >= 0; i--) {
        if (x[i] < p[i]) break;
        if (x[i] > p[i]) return 1;
    }
    
    // Compute y^2 = x^3 + 7
    uint32_t b[8] = {0};
    b[0] = SECP256K1_B;
    
    uint32_t y[8];
    mul_mod(y, x, x);
    mul_mod(y, y, x);
    add_mod(y, y, b);
    
    // Compute square root
    sqrt_mod(y);
    
    // Choose correct sign based on parity
    if ((first_byte & 1) != (y[0] & 1)) {
        sub_mod(y, p, y);
    }
    
    // Generate precomputed points
    point_get_coords(r, x, y);
    
    return 0;
}

// Parse a public key from compressed format  
__device__ uint32_t parse_public(secp256k1_t* r, const uint32_t* k) {
    const uint32_t first_byte = k[0] & 0xff;
    
    if ((first_byte != 0x02) && (first_byte != 0x03)) {
        return 1;
    }
    
    uint32_t x[8];
    x[0] = (k[7] & 0xff00) << 16 | (k[7] & 0xff0000) | (k[7] & 0xff000000) >> 16 | (k[8] & 0xff);
    x[1] = (k[6] & 0xff00) << 16 | (k[6] & 0xff0000) | (k[6] & 0xff000000) >> 16 | (k[7] & 0xff);
    x[2] = (k[5] & 0xff00) << 16 | (k[5] & 0xff0000) | (k[5] & 0xff000000) >> 16 | (k[6] & 0xff);
    x[3] = (k[4] & 0xff00) << 16 | (k[4] & 0xff0000) | (k[4] & 0xff000000) >> 16 | (k[5] & 0xff);
    x[4] = (k[3] & 0xff00) << 16 | (k[3] & 0xff0000) | (k[3] & 0xff000000) >> 16 | (k[4] & 0xff);
    x[5] = (k[2] & 0xff00) << 16 | (k[2] & 0xff0000) | (k[2] & 0xff000000) >> 16 | (k[3] & 0xff);
    x[6] = (k[1] & 0xff00) << 16 | (k[1] & 0xff0000) | (k[1] & 0xff000000) >> 16 | (k[2] & 0xff);
    x[7] = (k[0] & 0xff00) << 16 | (k[0] & 0xff0000) | (k[0] & 0xff000000) >> 16 | (k[1] & 0xff);
    
    return transform_public(r, x, first_byte);
}

// Point multiplication with compressed output
__device__ void point_mul(uint32_t* r, const uint32_t* k, const secp256k1_t* tmps) {
    uint32_t x[8];
    uint32_t y[8];
    
    point_mul_xy(x, y, k, tmps);
    
    // Pack into compressed format
    r[8] = (x[0] << 24);
    r[7] = (x[0] >> 8) | (x[1] << 24);
    r[6] = (x[1] >> 8) | (x[2] << 24);
    r[5] = (x[2] >> 8) | (x[3] << 24);
    r[4] = (x[3] >> 8) | (x[4] << 24);
    r[3] = (x[4] >> 8) | (x[5] << 24);
    r[2] = (x[5] >> 8) | (x[6] << 24);
    r[1] = (x[6] >> 8) | (x[7] << 24);
    r[0] = (x[7] >> 8);
    
    const uint32_t type = 0x02 | (y[0] & 1);
    r[0] = r[0] | type << 24;
}