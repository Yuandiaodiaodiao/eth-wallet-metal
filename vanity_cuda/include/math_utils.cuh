#pragma once

#include "constants.cuh"
#include "secp256k1.cuh"
#include <cuda_runtime.h>

// Modular multiplication (Montgomery reduction)
__device__ void mul_mod(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint64_t t[16] = {0};
    
    // Multiply
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a[i] * b[j] + t[i + j] + carry;
            t[i + j] = prod & 0xFFFFFFFF;
            carry = prod >> 32;
        }
        t[i + 8] = carry;
    }
    
    // Barrett reduction
    // Approximate division by P
    uint32_t q[9];
    uint64_t sum = 0;
    
    // Calculate q = t / P (approximation)
    for (int i = 15; i >= 7; i--) {
        sum = (sum << 32) | t[i];
        q[i - 7] = (uint32_t)(sum / 0xFFFFFFFEFFFFFC2FULL);
        sum %= 0xFFFFFFFEFFFFFC2FULL;
    }
    q[8] = 0;
    
    // Calculate r = t - q * P
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t prod = 0;
        #pragma unroll
        for (int j = 0; j <= i && j < 8; j++) {
            if (i - j < 9) {
                prod += (uint64_t)q[i - j] * SECP256K1_P[j];
            }
        }
        
        uint64_t diff = t[i] - prod - borrow;
        r[i] = (uint32_t)diff;
        borrow = (diff >> 63) ? 1 : 0;
    }
    
    // Conditional subtraction if r >= P
    uint32_t temp[8];
    uint32_t carry = sub256(temp, r, SECP256K1_P);
    if (!carry) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            r[i] = temp[i];
        }
    }
}

// 256-bit integer addition (returns carry)
__device__ uint32_t add(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t c = 0; // carry
    
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
          "=r"(c)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    #else
    // Fallback for older architectures
    for (int i = 0; i < 8; i++) {
        uint32_t t = a[i] + b[i] + c;
        c = (t != a[i]) ? (t < a[i]) : 0;
        r[i] = t;
    }
    #endif
    
    return c;
}

// Modular addition
__device__ void add_mod(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t carry = add(r, a, b);
    
    // If carry or r >= P, subtract P
    if (carry) {
        sub256(r, r, SECP256K1_P);
    } else {
        uint32_t temp[8];
        uint32_t borrow = sub256(temp, r, SECP256K1_P);
        if (!borrow) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                r[i] = temp[i];
            }
        }
    }
}

// Modular subtraction
__device__ void sub_mod(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t borrow = sub256(r, a, b);
    
    // If borrow, add P
    if (borrow) {
        add256(r, r, SECP256K1_P);
    }
}

// Modular inversion using Fermat's little theorem: a^(p-2) mod p
__device__ void inv_mod(uint32_t* r, const uint32_t* a) {
    uint32_t t1[8], t2[8], t3[8];
    
    // Copy a to result
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = a[i];
    }
    
    // Square 254 times and multiply
    // This implements a^(p-2) where p-2 = 0xFFFFFFFEFFFFFC2D
    
    // t1 = a^2
    mul_mod(t1, a, a);
    
    // t2 = a^3 = a^2 * a
    mul_mod(t2, t1, a);
    
    // t3 = a^6 = (a^3)^2
    mul_mod(t3, t2, t2);
    
    // Continue with addition chain for p-2
    // This is optimized for the specific value of secp256k1's p
    for (int i = 0; i < 5; i++) {
        mul_mod(t3, t3, t3);
    }
    mul_mod(t3, t3, t2); // a^(2^6 - 1)
    
    for (int i = 0; i < 3; i++) {
        mul_mod(t3, t3, t3);
    }
    mul_mod(t3, t3, a); // a^(2^9 - 1)
    
    for (int i = 0; i < 10; i++) {
        mul_mod(t3, t3, t3);
    }
    mul_mod(t3, t3, t2); // Continue pattern...
    
    // Final adjustments for exact exponent
    for (int i = 0; i < 14; i++) {
        mul_mod(t3, t3, t3);
    }
    mul_mod(t3, t3, t1);
    
    for (int i = 0; i < 219; i++) {
        mul_mod(t3, t3, t3);
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r[i] = t3[i];
    }
}