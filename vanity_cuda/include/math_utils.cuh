#pragma once

#include "constants.cuh"


// Optimized 256-bit copy using PTX assembly with 8x mov.u32
__forceinline__ __device__ void move_u256(uint32_t* __restrict__ dst, const uint32_t* __restrict__ src) {
    asm volatile (
        "mov.u32 %0, %8;\n\t"
        "mov.u32 %1, %9;\n\t"
        "mov.u32 %2, %10;\n\t"
        "mov.u32 %3, %11;\n\t"
        "mov.u32 %4, %12;\n\t"
        "mov.u32 %5, %13;\n\t"
        "mov.u32 %6, %14;\n\t"
        "mov.u32 %7, %15;\n\t"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
          "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
        : "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
          "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7])
    );
}

// Optimized SECP256K1_P constant loading using PTX assembly
__device__ __forceinline__ void load_secp256k1_p(uint32_t* t) {
    // SECP256K1_P = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F
    // Using PTX assembly for direct register operations and optimal performance
    #if __CUDA_ARCH__ >= 300
    asm volatile(
        "mov.u64 %0, 0xfffffffefffffc2f;\n\t"    // t[0:1] = FFFFFFFE FFFFFC2F
        "mov.u64 %1, 0xffffffffffffffff;\n\t"    // t[2:3] = FFFFFFFF FFFFFFFF
        "mov.u64 %2, 0xffffffffffffffff;\n\t"    // t[4:5] = FFFFFFFF FFFFFFFF
        "mov.u64 %3, 0xffffffffffffffff;\n\t"    // t[6:7] = FFFFFFFF FFFFFFFF
        : "=l"(*(uint64_t*)&t[0]), "=l"(*(uint64_t*)&t[2]),
          "=l"(*(uint64_t*)&t[4]), "=l"(*(uint64_t*)&t[6])
    );
    #else
    // Fallback for older architectures
    *(uint64_t*)&t[0] = 0xfffffffefffffc2fULL;  // P0|P1: FFFFFFFE FFFFFC2F
    *(uint64_t*)&t[2] = 0xffffffffffffffffULL;  // P2|P3: FFFFFFFF FFFFFFFF
    *(uint64_t*)&t[4] = 0xffffffffffffffffULL;  // P4|P5: FFFFFFFF FFFFFFFF
    *(uint64_t*)&t[6] = 0xffffffffffffffffULL;  // P6|P7: FFFFFFFF FFFFFFFF
    #endif
}

// 256-bit integer addition (returns carry) - internal helper
__device__ __forceinline__ uint32_t add(uint32_t* r, const uint32_t* a, const uint32_t* b) {
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
    // Fallback for older architectures
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        r[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    #endif
    
    return carry;
}

// 256-bit integer subtraction (returns borrow) - internal helper
__device__ __forceinline__ uint32_t sub(uint32_t* r, const uint32_t* a, const uint32_t* b) {
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
    // Fallback for older architectures
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t diff = a[i] - b[i] - borrow;
        borrow = (diff > a[i]) ? 1 : borrow;
        r[i] = diff;
    }
    #endif
    
    return borrow;
}

// 256-bit integer addition (no return value) - optimized for cases where carry is not needed
__device__ __forceinline__ void add_no_return(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "add.cc.u32  %0, %8, %16;\n\t"
        "addc.cc.u32 %1, %9, %17;\n\t"
        "addc.cc.u32 %2, %10, %18;\n\t"
        "addc.cc.u32 %3, %11, %19;\n\t"
        "addc.cc.u32 %4, %12, %20;\n\t"
        "addc.cc.u32 %5, %13, %21;\n\t"
        "addc.cc.u32 %6, %14, %22;\n\t"
        "addc.cc.u32 %7, %15, %23;\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
          "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
}

// 256-bit integer subtraction (no return value) - optimized for cases where borrow is not needed
__device__ __forceinline__ void sub_no_return(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    #if __CUDA_ARCH__ >= 300
    asm volatile(
        "sub.cc.u32  %0, %8, %16;\n\t"
        "subc.cc.u32 %1, %9, %17;\n\t"
        "subc.cc.u32 %2, %10, %18;\n\t"
        "subc.cc.u32 %3, %11, %19;\n\t"
        "subc.cc.u32 %4, %12, %20;\n\t"
        "subc.cc.u32 %5, %13, %21;\n\t"
        "subc.cc.u32 %6, %14, %22;\n\t"
        "subc.cc.u32 %7, %15, %23;\n\t"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
          "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
    #else
    // Fallback for older architectures
    uint32_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t diff = a[i] - b[i] - borrow;
        borrow = (diff > a[i]) ? 1 : borrow;
        r[i] = diff;
    }
    #endif
}

// 32-bit multiplication using inline assembly to avoid 64-bit registers
__device__ __forceinline__ void mul32(uint32_t& lo, uint32_t& hi, uint32_t a, uint32_t b) {
    #if __CUDA_ARCH__ >= 300
    asm volatile(
        "mul.lo.u32  %0, %2, %3;\n\t"
        "mul.hi.u32  %1, %2, %3;\n\t"
        : "=r"(lo), "=r"(hi)
        : "r"(a), "r"(b)
    );
    #else
    // Fallback for older architectures
    uint64_t result = (uint64_t)a * b;
    lo = (uint32_t)result;
    hi = (uint32_t)(result >> 32);
    #endif
}

// Optimized 32-bit multiplication using mul.wide for better performance
__device__ __forceinline__ void mul32_wide(uint32_t& lo, uint32_t& hi, uint32_t a, uint32_t b) {
    #if __CUDA_ARCH__ >= 300
    uint64_t result;
    asm volatile(
        "mul.wide.u32  %0, %1, %2;\n\t"
        : "=l"(result)
        : "r"(a), "r"(b)
    );
    lo = (uint32_t)result;
    hi = (uint32_t)(result >> 32);
    #else
    // Fallback for older architectures
    uint64_t result = (uint64_t)a * b;
    lo = (uint32_t)result;
    hi = (uint32_t)(result >> 32);
    #endif
}

// Direct 64-bit multiplication using mul.wide
__device__ __forceinline__ uint64_t mul64(uint32_t a, uint32_t b) {
    uint64_t result;
    asm volatile(
        "mul.wide.u32  %0, %1, %2;\n\t"
        : "=l"(result)
        : "r"(a), "r"(b)
    );
    return result;
}

// Optimized multiplication and accumulation for mul_mod inner loops
__device__ __forceinline__ void mul_add_carry_opt(uint32_t& t0, uint32_t& t1, uint32_t& c, 
                                                   uint32_t a, uint32_t b) {
    // Use simpler approach: direct 64-bit operations with better register management
    uint64_t p = mul64(a, b);                    // p = a * b using mul.wide
    uint64_t d = ((uint64_t)t1 << 32) | t0;     // combine t0,t1
    asm volatile(
        "add.cc.u64    %0, %0, %2;\n\t"         // d += p with carry
        "addc.u32      %1, %1, 0;\n\t"          // handle overflow carry
        : "+l"(d), "+r"(c)
        : "l"(p)
    );
    t0 = (uint32_t)d;
    t1 = (uint32_t)(d >> 32);
 
}

// 32-bit multiplication with addition using inline assembly
__device__ __forceinline__ void mad32(uint32_t& lo, uint32_t& hi, uint32_t a, uint32_t b, uint32_t c) {
    #if __CUDA_ARCH__ >= 300
    asm volatile(
        "mad.lo.cc.u32  %0, %2, %3, %4;\n\t"
        "madc.hi.u32    %1, %2, %3, 0;\n\t"
        : "=r"(lo), "=r"(hi)
        : "r"(a), "r"(b), "r"(c)
    );
    #else
    // Fallback for older architectures
    uint64_t result = (uint64_t)a * b + c;
    lo = (uint32_t)result;
    hi = (uint32_t)(result >> 32);
    #endif
}

// 32-bit multiplication with double addition using inline assembly
__device__ __forceinline__ void mad32_add(uint32_t& lo, uint32_t& hi, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    #if __CUDA_ARCH__ >= 300
    asm volatile(
        "mad.lo.cc.u32  %0, %2, %3, %4;\n\t"
        "madc.hi.cc.u32 %1, %2, %3, 0;\n\t"
        "add.cc.u32     %0, %0, %5;\n\t"
        "addc.u32       %1, %1, 0;\n\t"
        : "=r"(lo), "=r"(hi)
        : "r"(a), "r"(b), "r"(c), "r"(d)
    );
    #else
    // Fallback for older architectures
    uint64_t result = (uint64_t)a * b + c + d;
    lo = (uint32_t)result;
    hi = (uint32_t)(result >> 32);
    #endif
}

// Optimized modular multiplication using SECP256K1 specific reduction with PTX assembly
__device__ void mul_mod(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t t[16] = { 0 }; // we need up to double the space (2 * 8)
    
    /*
     * First start with the basic a * b multiplication using optimized PTX:
     */
    
    uint32_t t0 = 0;
    uint32_t t1 = 0;
    uint32_t c = 0;

    #pragma unroll
    for (uint32_t i = 0; i < 8; i++)
    {
        #pragma unroll
        for (uint32_t j = 0; j <= i; j++)
        {
            // Use optimized multiplication and accumulation
            mul_add_carry_opt(t0, t1, c, a[j], b[i - j]);
        }
        
        t[i] = t0;
        t0 = t1;
        t1 = c;
        c = 0;
    }
    
    #pragma unroll
    for (uint32_t i = 8; i < 15; i++)
    {
        #pragma unroll
        for (uint32_t j = i - 7; j < 8; j++)
        {
            // Use optimized multiplication and accumulation
            mul_add_carry_opt(t0, t1, c, a[j], b[i - j]);
        }
        
        t[i] = t0;
        t0 = t1;
        t1 = c;
        c = 0;
    }
    
    t[15] = t0;
    
    /*
     * Now do the modulo operation:
     * (r = t % p)
     */
    
    uint32_t tmp[16] = { 0 };
    
    // Note: SECP256K1_P = 2^256 - 2^32 - 977 (0x03d1 = 977)
    // multiply t[8]...t[15] by omega:
    
    #pragma unroll
    for (uint32_t i = 0, j = 8; i < 8; i++, j++)
    {
        uint32_t p_lo, p_hi;
        mad32(p_lo, p_hi, 0x03d1, t[j], c);
        
        tmp[i] = p_lo;
        c = p_hi;
    }
    
    tmp[8] = c;
    
    c = add(tmp + 1, tmp + 1, t + 8); // modifies tmp[1]...tmp[8]
    
    tmp[9] = c;
    
    // r = t + tmp
    
    c = add(r, t, tmp);
    
    // multiply t[0]...t[7] by omega:
    
    uint32_t c2 = 0;
    
    #pragma unroll
    for (uint32_t i = 0, j = 8; i < 8; i++, j++)
    {
        uint32_t p_lo, p_hi;
        mad32(p_lo, p_hi, 0x3d1, tmp[j], c2);
        
        t[i] = p_lo;
        c2 = p_hi;
    }
    
    t[8] = c2;
    
    c2 = add(t + 1, t + 1, tmp + 8); // modifies t[1]...t[8]
    
    t[9] = c2;
    
    // r = r + t
    
    c2 = add(r, r, t);
    
    c += c2;
    
    load_secp256k1_p(t);
    
    #pragma unroll
    for (uint32_t i = c; i > 0; i--)
    {
        sub_no_return(r, r, t);
    }
    
    #pragma unroll
    for (int i = 7; i >= 0; i--)
    {
        if (r[i] < t[i]) break;
        
        if (r[i] > t[i])
        {
            sub_no_return(r, r, t);
            
            break;
        }
    }
}

// Original modular multiplication using SECP256K1 specific reduction

// Modular addition
__device__ void add_mod(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t carry = add(r, a, b);
    
    // If carry or r >= P, subtract P
    if (carry) {
        sub(r, r, SECP256K1_P);
    } else {
        uint32_t temp[8];
        uint32_t borrow = sub(temp, r, SECP256K1_P);
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
    uint32_t borrow = sub(r, a, b);
    
    // If borrow, add P
    if (borrow) {
        add(r, r, SECP256K1_P);
    }
}

// Modular inversion using binary extended Euclidean algorithm
__device__ void inv_mod(uint32_t* a) {
    uint32_t t0[8];
    
    move_u256(t0, a);


    
    uint32_t p[8];
    
    load_secp256k1_p(p);
    
    uint32_t t1[8];
    move_u256(t1, p);
    
    
    uint32_t t2[8] = { 0 };
    
    t2[0] = 0x00000001;
    
    uint32_t t3[8] = { 0 };
    
    uint32_t b = (t0[0] != t1[0])
                | (t0[1] != t1[1])
                | (t0[2] != t1[2])
                | (t0[3] != t1[3])
                | (t0[4] != t1[4])
                | (t0[5] != t1[5])
                | (t0[6] != t1[6])
                | (t0[7] != t1[7]);
    
    while (b)
    {
        if ((t0[0] & 1) == 0) // even
        {
            t0[0] = t0[0] >> 1 | t0[1] << 31;
            t0[1] = t0[1] >> 1 | t0[2] << 31;
            t0[2] = t0[2] >> 1 | t0[3] << 31;
            t0[3] = t0[3] >> 1 | t0[4] << 31;
            t0[4] = t0[4] >> 1 | t0[5] << 31;
            t0[5] = t0[5] >> 1 | t0[6] << 31;
            t0[6] = t0[6] >> 1 | t0[7] << 31;
            t0[7] = t0[7] >> 1;
            
            uint32_t c = 0;
            
            if (t2[0] & 1) c = add(t2, t2, p);
            
            t2[0] = t2[0] >> 1 | t2[1] << 31;
            t2[1] = t2[1] >> 1 | t2[2] << 31;
            t2[2] = t2[2] >> 1 | t2[3] << 31;
            t2[3] = t2[3] >> 1 | t2[4] << 31;
            t2[4] = t2[4] >> 1 | t2[5] << 31;
            t2[5] = t2[5] >> 1 | t2[6] << 31;
            t2[6] = t2[6] >> 1 | t2[7] << 31;
            t2[7] = t2[7] >> 1 | c << 31;
        }
        else if ((t1[0] & 1) == 0)
        {
            t1[0] = t1[0] >> 1 | t1[1] << 31;
            t1[1] = t1[1] >> 1 | t1[2] << 31;
            t1[2] = t1[2] >> 1 | t1[3] << 31;
            t1[3] = t1[3] >> 1 | t1[4] << 31;
            t1[4] = t1[4] >> 1 | t1[5] << 31;
            t1[5] = t1[5] >> 1 | t1[6] << 31;
            t1[6] = t1[6] >> 1 | t1[7] << 31;
            t1[7] = t1[7] >> 1;
            
            uint32_t c = 0;
            
            if (t3[0] & 1) c = add(t3, t3, p);
            
            t3[0] = t3[0] >> 1 | t3[1] << 31;
            t3[1] = t3[1] >> 1 | t3[2] << 31;
            t3[2] = t3[2] >> 1 | t3[3] << 31;
            t3[3] = t3[3] >> 1 | t3[4] << 31;
            t3[4] = t3[4] >> 1 | t3[5] << 31;
            t3[5] = t3[5] >> 1 | t3[6] << 31;
            t3[6] = t3[6] >> 1 | t3[7] << 31;
            t3[7] = t3[7] >> 1 | c << 31;
        }
        else
        {
            uint32_t gt = 0;
            
            for (int i = 7; i >= 0; i--)
            {
                if (t0[i] > t1[i])
                {
                    gt = 1;
                    
                    break;
                }
                
                if (t0[i] < t1[i]) break;
            }
            
            if (gt)
            {
                sub_no_return(t0, t0, t1);
                
                t0[0] = t0[0] >> 1 | t0[1] << 31;
                t0[1] = t0[1] >> 1 | t0[2] << 31;
                t0[2] = t0[2] >> 1 | t0[3] << 31;
                t0[3] = t0[3] >> 1 | t0[4] << 31;
                t0[4] = t0[4] >> 1 | t0[5] << 31;
                t0[5] = t0[5] >> 1 | t0[6] << 31;
                t0[6] = t0[6] >> 1 | t0[7] << 31;
                t0[7] = t0[7] >> 1;
                
                uint32_t lt = 0;
                
                #pragma unroll
                for (int i = 7; i >= 0; i--)
                {
                    if (t2[i] < t3[i])
                    {
                        lt = 1;
                        
                        break;
                    }
                    
                    if (t2[i] > t3[i]) break;
                }
                
                if (lt) add_no_return(t2, t2, p);
                
                sub_no_return(t2, t2, t3);
                
                uint32_t c = 0;
                
                if (t2[0] & 1) c = add(t2, t2, p);
                
                t2[0] = t2[0] >> 1 | t2[1] << 31;
                t2[1] = t2[1] >> 1 | t2[2] << 31;
                t2[2] = t2[2] >> 1 | t2[3] << 31;
                t2[3] = t2[3] >> 1 | t2[4] << 31;
                t2[4] = t2[4] >> 1 | t2[5] << 31;
                t2[5] = t2[5] >> 1 | t2[6] << 31;
                t2[6] = t2[6] >> 1 | t2[7] << 31;
                t2[7] = t2[7] >> 1 | c << 31;
            }
            else
            {
                sub_no_return(t1, t1, t0);
                
                t1[0] = t1[0] >> 1 | t1[1] << 31;
                t1[1] = t1[1] >> 1 | t1[2] << 31;
                t1[2] = t1[2] >> 1 | t1[3] << 31;
                t1[3] = t1[3] >> 1 | t1[4] << 31;
                t1[4] = t1[4] >> 1 | t1[5] << 31;
                t1[5] = t1[5] >> 1 | t1[6] << 31;
                t1[6] = t1[6] >> 1 | t1[7] << 31;
                t1[7] = t1[7] >> 1;
                
                uint32_t lt = 0;
                
                #pragma unroll
                for (int i = 7; i >= 0; i--)
                {
                    if (t3[i] < t2[i])
                    {
                        lt = 1;
                        
                        break;
                    }
                    
                    if (t3[i] > t2[i]) break;
                }
                
                if (lt) add_no_return(t3, t3, p);
                
                sub_no_return(t3, t3, t2);
                
                uint32_t c = 0;
                
                if (t3[0] & 1) c = add(t3, t3, p);
                
                t3[0] = t3[0] >> 1 | t3[1] << 31;
                t3[1] = t3[1] >> 1 | t3[2] << 31;
                t3[2] = t3[2] >> 1 | t3[3] << 31;
                t3[3] = t3[3] >> 1 | t3[4] << 31;
                t3[4] = t3[4] >> 1 | t3[5] << 31;
                t3[5] = t3[5] >> 1 | t3[6] << 31;
                t3[6] = t3[6] >> 1 | t3[7] << 31;
                t3[7] = t3[7] >> 1 | c << 31;
            }
        }
        
        // update b:
        
        b = (t0[0] != t1[0])
          | (t0[1] != t1[1])
          | (t0[2] != t1[2])
          | (t0[3] != t1[3])
          | (t0[4] != t1[4])
          | (t0[5] != t1[5])
          | (t0[6] != t1[6])
          | (t0[7] != t1[7]);
    }
    
    // set result:
    move_u256(a, t2);

}