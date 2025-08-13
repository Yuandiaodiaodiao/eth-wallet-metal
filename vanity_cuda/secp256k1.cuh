#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include <cuda_runtime.h>

// Basic type definitions
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned long long size_t;

// secp256k1 curve parameters
#define SECP256K1_P0 0xFFFFFC2F
#define SECP256K1_P1 0xFFFFFFFE
#define SECP256K1_P2 0xFFFFFFFF
#define SECP256K1_P3 0xFFFFFFFF
#define SECP256K1_P4 0xFFFFFFFF
#define SECP256K1_P5 0xFFFFFFFF
#define SECP256K1_P6 0xFFFFFFFF
#define SECP256K1_P7 0xFFFFFFFF

// Order of the curve
#define SECP256K1_N0 0xD0364141
#define SECP256K1_N1 0xBFD25E8C
#define SECP256K1_N2 0xAF48A03B
#define SECP256K1_N3 0xBAAEDCE6
#define SECP256K1_N4 0xFFFFFFFE
#define SECP256K1_N5 0xFFFFFFFF
#define SECP256K1_N6 0xFFFFFFFF
#define SECP256K1_N7 0xFFFFFFFF

// Generator point G
#define SECP256K1_GX0 0x16F81798
#define SECP256K1_GX1 0x59F2815B
#define SECP256K1_GX2 0x2DCE28D9
#define SECP256K1_GX3 0x029BFCDB
#define SECP256K1_GX4 0xCE870B07
#define SECP256K1_GX5 0x55A06295
#define SECP256K1_GX6 0xF9DCBBAC
#define SECP256K1_GX7 0x79BE667E

#define SECP256K1_GY0 0xFB10D4B8
#define SECP256K1_GY1 0x9C47D08F
#define SECP256K1_GY2 0xA6855419
#define SECP256K1_GY3 0xFD17B448
#define SECP256K1_GY4 0x0E1108A8
#define SECP256K1_GY5 0x5DA4FBFC
#define SECP256K1_GY6 0x26A3C465
#define SECP256K1_GY7 0x483ADA77

// 256-bit integer operations using CUDA PTX inline assembly
__device__ __forceinline__ uint32_t sub256(uint32_t* r, const uint32_t* a, const uint32_t* b) {
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
    for (int i = 0; i < 8; i++) {
        uint32_t diff = a[i] - b[i] - borrow;
        borrow = (diff > a[i]) ? 1 : borrow;
        r[i] = diff;
    }
    #endif
    
    return borrow;
}

__device__ __forceinline__ uint32_t add256(uint32_t* r, const uint32_t* a, const uint32_t* b) {
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
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        r[i] = (uint32_t)sum;
        carry = (uint32_t)(sum >> 32);
    }
    #endif
    
    return carry;
}

// Modular multiplication modulo P
__device__ void mul_mod(uint32_t* r, const uint32_t* a, const uint32_t* b);

// Modular addition modulo P
__device__ void add_mod(uint32_t* r, const uint32_t* a, const uint32_t* b);

// Modular subtraction modulo P
__device__ void sub_mod(uint32_t* r, const uint32_t* a, const uint32_t* b);

// Modular inversion using Fermat's little theorem
__device__ void inv_mod(uint32_t* r, const uint32_t* a);

// Point addition in Jacobian coordinates
__device__ void point_add(uint32_t* x3, uint32_t* y3, uint32_t* z3,
                         const uint32_t* x1, const uint32_t* y1, const uint32_t* z1,
                         const uint32_t* x2, const uint32_t* y2);

// Point doubling in Jacobian coordinates
__device__ void point_double(uint32_t* x3, uint32_t* y3, uint32_t* z3,
                            const uint32_t* x1, const uint32_t* y1, const uint32_t* z1);

// Scalar multiplication using 16-bit window method
__device__ void point_mul_g16(uint32_t* x_out, uint32_t* y_out,
                              const uint32_t* k, const uint8_t* g16_table);

// Convert Jacobian to affine coordinates
__device__ void jacobian_to_affine(uint32_t* x_affine, uint32_t* y_affine,
                                   const uint32_t* x, const uint32_t* y, const uint32_t* z);

// Batch inversion using Montgomery's trick
__device__ void batch_inverse(uint32_t xs[][8], uint32_t ys[][8], uint32_t zs[][8],
                              uint32_t temp_pref[][8], int batch_size);

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

#endif // SECP256K1_CUH