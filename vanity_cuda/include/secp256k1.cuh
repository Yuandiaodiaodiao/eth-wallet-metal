#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include "constants.cuh"
#include <cuda_runtime.h>

// 256-bit integer addition (returns carry)
__device__ __forceinline__ uint32_t add(uint32_t* r, const uint32_t* a, const uint32_t* b);

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

// Point addition in Jacobian coordinates (mixed: z1 jacobian, z2=1 affine)
__device__ void point_add(uint32_t* x1, uint32_t* y1, uint32_t* z1,
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

// Forward declarations for G16 operations (implementations in g16_ops.cuh)
struct G16Table;
__device__ __forceinline__ uint32_t get_window16(const uint32_t* k, uint32_t window);
__device__ __forceinline__ void load_point_g16(uint32_t* x, uint32_t* y,
                                               const uint8_t* g16_table,
                                               uint32_t window, uint32_t idx);

#endif // SECP256K1_CUH