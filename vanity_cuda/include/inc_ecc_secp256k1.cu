#include "inc_ecc_secp256k1.cuh"

// 256-bit subtraction with borrow
__device__ uint32_t sub(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    uint32_t borrow = 0;
    
    #if __CUDA_ARCH__ >= 300
    asm volatile(
        "sub.cc.u32  %0, %9, %17;\n\t"
        "subc.cc.u32 %1, %10, %18;\n\t"
        "subc.cc.u32 %2, %11, %19;\n\t"
        "subc.cc.u32 %3, %12, %20;\n\t"
        "subc.cc.u32 %4, %13, %21;\n\t"
        "subc.cc.u32 %5, %14, %22;\n\t"
        "subc.cc.u32 %6, %15, %23;\n\t"
        "subc.cc.u32 %7, %16, %24;\n\t"
        "subc.u32    %8, 0, 0;\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
          "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
          "=r"(borrow)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    #else
    for (int i = 0; i < 8; i++) {
        const uint32_t diff = a[i] - b[i] - borrow;
        if (diff != a[i]) borrow = (diff > a[i]);
        r[i] = diff;
    }
    #endif
    
    return borrow;
}

// 256-bit addition with carry
__device__ uint32_t add(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    uint32_t carry = 0;
    
    #if __CUDA_ARCH__ >= 300
    asm volatile(
        "add.cc.u32  %0, %9, %17;\n\t"
        "addc.cc.u32 %1, %10, %18;\n\t"
        "addc.cc.u32 %2, %11, %19;\n\t"
        "addc.cc.u32 %3, %12, %20;\n\t"
        "addc.cc.u32 %4, %13, %21;\n\t"
        "addc.cc.u32 %5, %14, %22;\n\t"
        "addc.cc.u32 %6, %15, %23;\n\t"
        "addc.cc.u32 %7, %16, %24;\n\t"
        "addc.u32    %8, 0, 0;\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
          "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
          "=r"(carry)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    #else
    for (int i = 0; i < 8; i++) {
        const uint32_t t = a[i] + b[i] + carry;
        if (t != a[i]) carry = (t < a[i]);
        r[i] = t;
    }
    #endif
    
    return carry;
}

// Modular subtraction modulo P
__device__ void sub_mod(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    const uint32_t c = sub(r, a, b);
    
    if (c) {
        uint32_t t[8];
        t[0] = SECP256K1_P0;
        t[1] = SECP256K1_P1;
        t[2] = SECP256K1_P2;
        t[3] = SECP256K1_P3;
        t[4] = SECP256K1_P4;
        t[5] = SECP256K1_P5;
        t[6] = SECP256K1_P6;
        t[7] = SECP256K1_P7;
        add(r, r, t);
    }
}

// Modular addition modulo P
__device__ void add_mod(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    const uint32_t c = add(r, a, b);
    
    uint32_t t[8];
    t[0] = SECP256K1_P0;
    t[1] = SECP256K1_P1;
    t[2] = SECP256K1_P2;
    t[3] = SECP256K1_P3;
    t[4] = SECP256K1_P4;
    t[5] = SECP256K1_P5;
    t[6] = SECP256K1_P6;
    t[7] = SECP256K1_P7;
    
    uint32_t mod = 1;
    
    if (c == 0) {
        for (int i = 7; i >= 0; i--) {
            if (r[i] < t[i]) {
                mod = 0;
                break;
            }
            if (r[i] > t[i]) break;
        }
    }
    
    if (mod == 1) {
        sub(r, r, t);
    }
}

// Modular multiplication modulo P
__device__ void mul_mod(uint32_t *r, const uint32_t *a, const uint32_t *b) {
    uint32_t t[16] = {0};
    
    uint32_t t0 = 0;
    uint32_t t1 = 0;
    uint32_t c = 0;
    
    for (uint32_t i = 0; i < 8; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            uint64_t p = ((uint64_t)a[j]) * b[i - j];
            uint64_t d = ((uint64_t)t1) << 32 | t0;
            d += p;
            t0 = (uint32_t)d;
            t1 = d >> 32;
            c += d < p;
        }
        t[i] = t0;
        t0 = t1;
        t1 = c;
        c = 0;
    }
    
    for (uint32_t i = 8; i < 15; i++) {
        for (uint32_t j = i - 7; j < 8; j++) {
            uint64_t p = ((uint64_t)a[j]) * b[i - j];
            uint64_t d = ((uint64_t)t1) << 32 | t0;
            d += p;
            t0 = (uint32_t)d;
            t1 = d >> 32;
            c += d < p;
        }
        t[i] = t0;
        t0 = t1;
        t1 = c;
        c = 0;
    }
    
    t[15] = t0;
    
    // Modulo reduction using SECP256K1_P = 2^256 - 2^32 - 977 (0x03d1 = 977)
    uint32_t tmp[16] = {0};
    
    for (uint32_t i = 0, j = 8; i < 8; i++, j++) {
        uint64_t p = ((uint64_t)0x03d1) * t[j] + c;
        tmp[i] = (uint32_t)p;
        c = p >> 32;
    }
    
    tmp[8] = c;
    c = add(tmp + 1, tmp + 1, t + 8);
    tmp[9] = c;
    
    c = add(r, t, tmp);
    
    uint32_t c2 = 0;
    for (uint32_t i = 0, j = 8; i < 8; i++, j++) {
        uint64_t p = ((uint64_t)0x3d1) * tmp[j] + c2;
        t[i] = (uint32_t)p;
        c2 = p >> 32;
    }
    
    t[8] = c2;
    c2 = add(t + 1, t + 1, tmp + 8);
    t[9] = c2;
    
    c2 = add(r, r, t);
    c += c2;
    
    t[0] = SECP256K1_P0;
    t[1] = SECP256K1_P1;
    t[2] = SECP256K1_P2;
    t[3] = SECP256K1_P3;
    t[4] = SECP256K1_P4;
    t[5] = SECP256K1_P5;
    t[6] = SECP256K1_P6;
    t[7] = SECP256K1_P7;
    
    for (uint32_t i = c; i > 0; i--) {
        sub(r, r, t);
    }
    
    for (int i = 7; i >= 0; i--) {
        if (r[i] < t[i]) break;
        if (r[i] > t[i]) {
            sub(r, r, t);
            break;
        }
    }
}

// Square root modulo P using Fermat's Little Theorem
__device__ void sqrt_mod(uint32_t *r) {
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

// Modular inverse using extended Euclidean algorithm
__device__ void inv_mod(uint32_t *a) {
    uint32_t t0[8], t1[8], t2[8], t3[8], p[8];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t0[i] = a[i];
    }
    
    p[0] = SECP256K1_P0;
    p[1] = SECP256K1_P1;
    p[2] = SECP256K1_P2;
    p[3] = SECP256K1_P3;
    p[4] = SECP256K1_P4;
    p[5] = SECP256K1_P5;
    p[6] = SECP256K1_P6;
    p[7] = SECP256K1_P7;
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t1[i] = p[i];
    }
    
    t2[0] = 1;
    for (int i = 1; i < 8; i++) t2[i] = 0;
    
    for (int i = 0; i < 8; i++) t3[i] = 0;
    
    uint32_t b = (t0[0] != t1[0]) | (t0[1] != t1[1]) | (t0[2] != t1[2]) | (t0[3] != t1[3]) |
                 (t0[4] != t1[4]) | (t0[5] != t1[5]) | (t0[6] != t1[6]) | (t0[7] != t1[7]);
    
    while (b) {
        if ((t0[0] & 1) == 0) {
            for (int i = 0; i < 7; i++) {
                t0[i] = t0[i] >> 1 | t0[i+1] << 31;
            }
            t0[7] = t0[7] >> 1;
            
            uint32_t c = 0;
            if (t2[0] & 1) c = add(t2, t2, p);
            
            for (int i = 0; i < 7; i++) {
                t2[i] = t2[i] >> 1 | t2[i+1] << 31;
            }
            t2[7] = t2[7] >> 1 | c << 31;
        }
        else if ((t1[0] & 1) == 0) {
            for (int i = 0; i < 7; i++) {
                t1[i] = t1[i] >> 1 | t1[i+1] << 31;
            }
            t1[7] = t1[7] >> 1;
            
            uint32_t c = 0;
            if (t3[0] & 1) c = add(t3, t3, p);
            
            for (int i = 0; i < 7; i++) {
                t3[i] = t3[i] >> 1 | t3[i+1] << 31;
            }
            t3[7] = t3[7] >> 1 | c << 31;
        }
        else {
            uint32_t gt = 0;
            for (int i = 7; i >= 0; i--) {
                if (t0[i] > t1[i]) {
                    gt = 1;
                    break;
                }
                if (t0[i] < t1[i]) break;
            }
            
            if (gt) {
                sub(t0, t0, t1);
                
                for (int i = 0; i < 7; i++) {
                    t0[i] = t0[i] >> 1 | t0[i+1] << 31;
                }
                t0[7] = t0[7] >> 1;
                
                uint32_t lt = 0;
                for (int i = 7; i >= 0; i--) {
                    if (t2[i] < t3[i]) {
                        lt = 1;
                        break;
                    }
                    if (t2[i] > t3[i]) break;
                }
                
                if (lt) add(t2, t2, p);
                sub(t2, t2, t3);
                
                uint32_t c = 0;
                if (t2[0] & 1) c = add(t2, t2, p);
                
                for (int i = 0; i < 7; i++) {
                    t2[i] = t2[i] >> 1 | t2[i+1] << 31;
                }
                t2[7] = t2[7] >> 1 | c << 31;
            }
            else {
                sub(t1, t1, t0);
                
                for (int i = 0; i < 7; i++) {
                    t1[i] = t1[i] >> 1 | t1[i+1] << 31;
                }
                t1[7] = t1[7] >> 1;
                
                uint32_t lt = 0;
                for (int i = 7; i >= 0; i--) {
                    if (t3[i] < t2[i]) {
                        lt = 1;
                        break;
                    }
                    if (t3[i] > t2[i]) break;
                }
                
                if (lt) add(t3, t3, p);
                sub(t3, t3, t2);
                
                uint32_t c = 0;
                if (t3[0] & 1) c = add(t3, t3, p);
                
                for (int i = 0; i < 7; i++) {
                    t3[i] = t3[i] >> 1 | t3[i+1] << 31;
                }
                t3[7] = t3[7] >> 1 | c << 31;
            }
        }
        
        b = (t0[0] != t1[0]) | (t0[1] != t1[1]) | (t0[2] != t1[2]) | (t0[3] != t1[3]) |
            (t0[4] != t1[4]) | (t0[5] != t1[5]) | (t0[6] != t1[6]) | (t0[7] != t1[7]);
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        a[i] = t2[i];
    }
}

// Point doubling in Jacobian coordinates
__device__ void point_double(uint32_t *x, uint32_t *y, uint32_t *z) {
    uint32_t t1[8], t2[8], t3[8], t4[8], t5[8], t6[8];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t1[i] = x[i];
        t2[i] = y[i];
        t3[i] = z[i];
    }
    
    mul_mod(t4, t1, t1);  // t4 = x^2
    mul_mod(t5, t2, t2);  // t5 = y^2
    mul_mod(t1, t1, t5);  // t1 = x*y^2
    mul_mod(t5, t5, t5);  // t5 = y^4
    mul_mod(t3, t2, t3);  // t3 = y*z
    
    add_mod(t2, t4, t4);  // t2 = 2*x^2
    add_mod(t4, t4, t2);  // t4 = 3*x^2
    
    uint32_t c = 0;
    if (t4[0] & 1) {
        uint32_t t[8];
        t[0] = SECP256K1_P0;
        t[1] = SECP256K1_P1;
        t[2] = SECP256K1_P2;
        t[3] = SECP256K1_P3;
        t[4] = SECP256K1_P4;
        t[5] = SECP256K1_P5;
        t[6] = SECP256K1_P6;
        t[7] = SECP256K1_P7;
        c = add(t4, t4, t);
    }
    
    t4[0] = t4[0] >> 1 | t4[1] << 31;
    t4[1] = t4[1] >> 1 | t4[2] << 31;
    t4[2] = t4[2] >> 1 | t4[3] << 31;
    t4[3] = t4[3] >> 1 | t4[4] << 31;
    t4[4] = t4[4] >> 1 | t4[5] << 31;
    t4[5] = t4[5] >> 1 | t4[6] << 31;
    t4[6] = t4[6] >> 1 | t4[7] << 31;
    t4[7] = t4[7] >> 1 | c << 31;
    
    mul_mod(t6, t4, t4);
    add_mod(t2, t1, t1);
    sub_mod(t6, t6, t2);
    sub_mod(t1, t1, t6);
    mul_mod(t4, t4, t1);
    sub_mod(t1, t4, t5);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x[i] = t6[i];
        y[i] = t1[i];
        z[i] = t3[i];
    }
}

// Point addition in Jacobian coordinates (mixed: z2 = 1)
__device__ void point_add(uint32_t *x1, uint32_t *y1, uint32_t *z1, uint32_t *x2, uint32_t *y2) {
    uint32_t t1[8], t2[8], t3[8], t4[8], t5[8], t6[8], t7[8], t8[8], t9[8];
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t1[i] = x1[i];
        t2[i] = y1[i];
        t3[i] = z1[i];
        t4[i] = x2[i];
        t5[i] = y2[i];
    }
    
    mul_mod(t6, t3, t3);
    mul_mod(t7, t6, t3);
    mul_mod(t6, t6, t4);
    mul_mod(t7, t7, t5);
    
    sub_mod(t6, t6, t1);
    sub_mod(t7, t7, t2);
    
    mul_mod(t8, t3, t6);
    mul_mod(t4, t6, t6);
    mul_mod(t9, t4, t6);
    mul_mod(t4, t4, t1);
    
    t6[7] = t4[7] << 1 | t4[6] >> 31;
    t6[6] = t4[6] << 1 | t4[5] >> 31;
    t6[5] = t4[5] << 1 | t4[4] >> 31;
    t6[4] = t4[4] << 1 | t4[3] >> 31;
    t6[3] = t4[3] << 1 | t4[2] >> 31;
    t6[2] = t4[2] << 1 | t4[1] >> 31;
    t6[1] = t4[1] << 1 | t4[0] >> 31;
    t6[0] = t4[0] << 1;
    
    if (t4[7] & 0x80000000) {
        uint32_t a[8] = {0};
        a[1] = 1;
        a[0] = 0x000003d1;
        add(t6, t6, a);
    }
    
    mul_mod(t5, t7, t7);
    sub_mod(t5, t5, t6);
    sub_mod(t5, t5, t9);
    sub_mod(t4, t4, t5);
    
    mul_mod(t4, t4, t7);
    mul_mod(t9, t9, t2);
    sub_mod(t9, t4, t9);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x1[i] = t5[i];
        y1[i] = t9[i];
        z1[i] = t8[i];
    }
}

// Get precomputed coordinates for wNAF
__device__ void point_get_coords(secp256k1_t *r, const uint32_t *x, const uint32_t *y) {
    // Store x1, y1
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[i] = x[i];
        r->xy[8 + i] = y[i];
    }
    
    // Store -y1
    uint32_t p[8];
    p[0] = SECP256K1_P0;
    p[1] = SECP256K1_P1;
    p[2] = SECP256K1_P2;
    p[3] = SECP256K1_P3;
    p[4] = SECP256K1_P4;
    p[5] = SECP256K1_P5;
    p[6] = SECP256K1_P6;
    p[7] = SECP256K1_P7;
    
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
    
    inv_mod(rz);
    mul_mod(neg, rz, rz);
    mul_mod(rx, rx, neg);
    mul_mod(rz, neg, rz);
    mul_mod(ry, ry, rz);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[24 + i] = rx[i];
        r->xy[32 + i] = ry[i];
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        neg[i] = ry[i];
    }
    sub_mod(neg, p, neg);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[40 + i] = neg[i];
    }
    
    // 5*P
    rz[0] = 1;
    for (int i = 1; i < 8; i++) rz[i] = 0;
    
    point_add(rx, ry, rz, tx, ty);
    point_add(rx, ry, rz, tx, ty);
    
    inv_mod(rz);
    mul_mod(neg, rz, rz);
    mul_mod(rx, rx, neg);
    mul_mod(rz, neg, rz);
    mul_mod(ry, ry, rz);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[48 + i] = rx[i];
        r->xy[56 + i] = ry[i];
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        neg[i] = ry[i];
    }
    sub_mod(neg, p, neg);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[64 + i] = neg[i];
    }
    
    // 7*P
    rz[0] = 1;
    for (int i = 1; i < 8; i++) rz[i] = 0;
    
    point_add(rx, ry, rz, tx, ty);
    point_add(rx, ry, rz, tx, ty);
    
    inv_mod(rz);
    mul_mod(neg, rz, rz);
    mul_mod(rx, rx, neg);
    mul_mod(rz, neg, rz);
    mul_mod(ry, ry, rz);
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r->xy[72 + i] = rx[i];
        r->xy[80 + i] = ry[i];
    }
    
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

// Convert scalar to w-NAF
__device__ int convert_to_window_naf(uint32_t *naf, const uint32_t *k) {
    int loop_start = 0;
    
    uint32_t n[9];
    n[0] = 0;
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
            int diff = n[8] & 0x0f;
            int val = diff;
            
            if (diff >= 0x08) {
                diff -= 0x10;
                val = 0x11 - val;
            }
            
            naf[i >> 3] |= val << ((i & 7) << 2);
            
            uint32_t t = n[8];
            n[8] -= diff;
            
            uint32_t k = 8;
            
            if (diff > 0) {
                while (n[k] > t) {
                    if (k == 0) break;
                    k--;
                    t = n[k];
                    n[k]--;
                }
            } else {
                while (t > n[k]) {
                    if (k == 0) break;
                    k--;
                    t = n[k];
                    n[k]++;
                }
            }
            
            loop_start = i;
        }
        
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

// Point multiplication with x,y output
__device__ void point_mul_xy(uint32_t *x1, uint32_t *y1, const uint32_t *k, const secp256k1_t *tmps) {
    uint32_t naf[SECP256K1_NAF_SIZE] = {0};
    
    int loop_start = convert_to_window_naf(naf, k);
    
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
    
    for (int pos = loop_start - 1; pos >= 0; pos--) {
        point_double(x1, y1, z1);
        
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
            
            point_add(x1, y1, z1, x2, y2);
        }
    }
    
    inv_mod(z1);
    
    uint32_t z2[8];
    mul_mod(z2, z1, z1);
    mul_mod(x1, x1, z2);
    
    mul_mod(z1, z2, z1);
    mul_mod(y1, y1, z1);
}

// Point multiplication with compressed output
__device__ void point_mul(uint32_t *r, const uint32_t *k, const secp256k1_t *tmps) {
    uint32_t x[8];
    uint32_t y[8];
    
    point_mul_xy(x, y, k, tmps);
    
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

// Transform a public key
__device__ uint32_t transform_public(secp256k1_t *r, const uint32_t *x, const uint32_t first_byte) {
    uint32_t p[8];
    p[0] = SECP256K1_P0;
    p[1] = SECP256K1_P1;
    p[2] = SECP256K1_P2;
    p[3] = SECP256K1_P3;
    p[4] = SECP256K1_P4;
    p[5] = SECP256K1_P5;
    p[6] = SECP256K1_P6;
    p[7] = SECP256K1_P7;
    
    for (int i = 7; i >= 0; i--) {
        if (x[i] < p[i]) break;
        if (x[i] > p[i]) return 1;
    }
    
    uint32_t b[8] = {0};
    b[0] = SECP256K1_B;
    
    uint32_t y[8];
    mul_mod(y, x, x);
    mul_mod(y, y, x);
    add_mod(y, y, b);
    
    sqrt_mod(y);
    
    if ((first_byte & 1) != (y[0] & 1)) {
        sub_mod(y, p, y);
    }
    
    point_get_coords(r, x, y);
    
    return 0;
}

// Parse a public key
__device__ uint32_t parse_public(secp256k1_t *r, const uint32_t *k) {
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

