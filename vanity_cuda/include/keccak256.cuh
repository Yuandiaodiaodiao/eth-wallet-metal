#ifndef KECCAK256_CUH
#define KECCAK256_CUH

#include "math_utils.cuh"


// Rho offsets for Keccak (inlined as constants)
// Original values: {0,1,62,28,27,36,44,6,55,20,3,10,43,25,39,41,45,15,21,8,18,2,61,56,14}

// Rotate left 64-bit using PTX assembly
__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
    // jump these check  n must be 1-63
    // n &= 63; // Ensure n is in valid range
    // if (n == 0) return x;
    
    uint64_t result;
    
    asm volatile (
        "shl.b64  %0, %1, %2;\n\t"
        "shr.b64  %1, %1, %3;\n\t"
        "or.b64   %0, %0, %1;"
        : "=l"(result), "+l"(x)
        : "r"(n), "r"(64 - n)
    );
    
    return result;
}

// Optimized rotl64(x, 1) ^ y for fixed shift amount 1
__device__ __forceinline__ uint64_t rotl64_1_xor(uint64_t x, uint64_t y) {
    uint64_t result;
    asm volatile (
        "shl.b64  %0, %1, 1;\n\t"      // result = x << 1
        "shr.b64  %1, %1, 63;\n\t"     // x = x >> 63 (reuse register)
        "or.b64   %0, %0, %1;\n\t"     // result |= x (complete rotation)
        "xor.b64  %0, %0, %2;"         // result ^= y
        : "=l"(result)
        : "l"(x), "l"(y)
    );
    return result;
}

// Ultra-optimized D calculation with out-of-order execution
__device__ __forceinline__ void compute_D_optimized(uint64_t C[5], uint64_t D[5]) {
    uint64_t temp0, temp1, temp2, temp3, temp4;
    asm volatile (
        // Phase 1: Parallel shl operations (no dependencies)
        "shl.b64  %0, %10, 1;\n\t"    // D[0] shl: C[1] << 1
        "shl.b64  %1, %11, 1;\n\t"    // D[1] shl: C[2] << 1  
        "shl.b64  %2, %12, 1;\n\t"    // D[2] shl: C[3] << 1
        "shl.b64  %3, %13, 1;\n\t"    // D[3] shl: C[4] << 1
        "shl.b64  %4, %14, 1;\n\t"    // D[4] shl: C[0] << 1
        
        // Phase 2: Parallel shr operations using independent temp registers
        "shr.b64  %5, %10, 63;\n\t"   // temp0: C[1] >> 63
        "shr.b64  %6, %11, 63;\n\t"   // temp1: C[2] >> 63
        "shr.b64  %7, %12, 63;\n\t"   // temp2: C[3] >> 63
        "shr.b64  %8, %13, 63;\n\t"   // temp3: C[4] >> 63
        "shr.b64  %9, %14, 63;\n\t"   // temp4: C[0] >> 63
        
        // Phase 3: Interleaved or and xor operations to maximize ILP
        "or.b64   %0, %0, %5;\n\t"    // D[0] |= temp0 (complete rotation)
        "or.b64   %1, %1, %6;\n\t"    // D[1] |= temp1
        "or.b64   %2, %2, %7;\n\t"    // D[2] |= temp2
        "or.b64   %3, %3, %8;\n\t"    // D[3] |= temp3
        "or.b64   %4, %4, %9;\n\t"    // D[4] |= temp4

        "xor.b64  %0, %0, %19;\n\t"   // D[0] ^= C[4] (final D[0])
        "xor.b64  %1, %1, %15;\n\t"   // D[1] ^= C[0] (final D[1])
        "xor.b64  %2, %2, %16;\n\t"   // D[2] ^= C[1] (final D[2])
        "xor.b64  %3, %3, %17;\n\t"   // D[3] ^= C[2] (final D[3])
        "xor.b64  %4, %4, %18;"       // D[4] ^= C[3] (final D[4])
        
        : "=l"(D[0]), "=l"(D[1]), "=l"(D[2]), "=l"(D[3]), "=l"(D[4]),
          "=l"(temp0), "=l"(temp1), "=l"(temp2), "=l"(temp3), "=l"(temp4)
        : "l"(C[1]), "l"(C[2]), "l"(C[3]), "l"(C[4]), "l"(C[0]),    // %10-14
          "l"(C[0]), "l"(C[1]), "l"(C[2]), "l"(C[3]), "l"(C[4])     // %15-19
    );
}



// Keccak-f[1600] permutation (optimized for registers)
__device__ __forceinline__ void keccak_f1600(uint64_t state[25]) {
    uint64_t C[5], D[5], B[25];

    for (int round = 0; round < 24; round++) {
        // Theta step - Split into 4 groups to reduce register pressure
        // C[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20]
        
        // Group 1: Process first row (state[0-4])
        asm volatile (
            "xor.b64    %0, %5, %6;\n\t"     // C[0] = state[0] ^ state[5]
            "xor.b64    %1, %7, %8;\n\t"     // C[1] = state[1] ^ state[6]
            "xor.b64    %2, %9, %10;\n\t"    // C[2] = state[2] ^ state[7]
            "xor.b64    %3, %11, %12;\n\t"   // C[3] = state[3] ^ state[8]  
            "xor.b64    %4, %13, %14;"       // C[4] = state[4] ^ state[9]
            : "=l"(C[0]), "=l"(C[1]), "=l"(C[2]), "=l"(C[3]), "=l"(C[4])
            : "l"(state[0]), "l"(state[5]),
              "l"(state[1]), "l"(state[6]),
              "l"(state[2]), "l"(state[7]),
              "l"(state[3]), "l"(state[8]),
              "l"(state[4]), "l"(state[9])
        );
        
        // Group 2: XOR with second row (state[10-14])
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // C[0] ^= state[10]
            "xor.b64    %1, %1, %6;\n\t"     // C[1] ^= state[11]
            "xor.b64    %2, %2, %7;\n\t"     // C[2] ^= state[12]
            "xor.b64    %3, %3, %8;\n\t"     // C[3] ^= state[13]
            "xor.b64    %4, %4, %9;"         // C[4] ^= state[14]
            : "+l"(C[0]), "+l"(C[1]), "+l"(C[2]), "+l"(C[3]), "+l"(C[4])
            : "l"(state[10]), "l"(state[11]), "l"(state[12]), "l"(state[13]), "l"(state[14])
        );
        
        // Group 3: XOR with third row (state[15-19])
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // C[0] ^= state[15]
            "xor.b64    %1, %1, %6;\n\t"     // C[1] ^= state[16]
            "xor.b64    %2, %2, %7;\n\t"     // C[2] ^= state[17]
            "xor.b64    %3, %3, %8;\n\t"     // C[3] ^= state[18]
            "xor.b64    %4, %4, %9;"         // C[4] ^= state[19]
            : "+l"(C[0]), "+l"(C[1]), "+l"(C[2]), "+l"(C[3]), "+l"(C[4])
            : "l"(state[15]), "l"(state[16]), "l"(state[17]), "l"(state[18]), "l"(state[19])
        );
        
        // Group 4: XOR with fourth row (state[20-24])
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // C[0] ^= state[20]
            "xor.b64    %1, %1, %6;\n\t"     // C[1] ^= state[21]
            "xor.b64    %2, %2, %7;\n\t"     // C[2] ^= state[22]
            "xor.b64    %3, %3, %8;\n\t"     // C[3] ^= state[23]
            "xor.b64    %4, %4, %9;"         // C[4] ^= state[24]
            : "+l"(C[0]), "+l"(C[1]), "+l"(C[2]), "+l"(C[3]), "+l"(C[4])
            : "l"(state[20]), "l"(state[21]), "l"(state[22]), "l"(state[23]), "l"(state[24])
        );
        
        // D calculation - ultra-optimized inline assembly
        compute_D_optimized(C, D);
        
        // State update - optimized with 5-group assembly
        // Group 1: state[0-4] ^= D[0-4]
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // state[0] ^= D[0]
            "xor.b64    %1, %1, %6;\n\t"     // state[1] ^= D[1]
            "xor.b64    %2, %2, %7;\n\t"     // state[2] ^= D[2]
            "xor.b64    %3, %3, %8;\n\t"     // state[3] ^= D[3]
            "xor.b64    %4, %4, %9;"         // state[4] ^= D[4]
            : "+l"(state[0]), "+l"(state[1]), "+l"(state[2]), "+l"(state[3]), "+l"(state[4])
            : "l"(D[0]), "l"(D[1]), "l"(D[2]), "l"(D[3]), "l"(D[4])
        );
        
        // Group 2: state[5-9] ^= D[0-4]
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // state[5] ^= D[0]
            "xor.b64    %1, %1, %6;\n\t"     // state[6] ^= D[1]
            "xor.b64    %2, %2, %7;\n\t"     // state[7] ^= D[2]
            "xor.b64    %3, %3, %8;\n\t"     // state[8] ^= D[3]
            "xor.b64    %4, %4, %9;"         // state[9] ^= D[4]
            : "+l"(state[5]), "+l"(state[6]), "+l"(state[7]), "+l"(state[8]), "+l"(state[9])
            : "l"(D[0]), "l"(D[1]), "l"(D[2]), "l"(D[3]), "l"(D[4])
        );
        
        // Group 3: state[10-14] ^= D[0-4]
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // state[10] ^= D[0]
            "xor.b64    %1, %1, %6;\n\t"     // state[11] ^= D[1]
            "xor.b64    %2, %2, %7;\n\t"     // state[12] ^= D[2]
            "xor.b64    %3, %3, %8;\n\t"     // state[13] ^= D[3]
            "xor.b64    %4, %4, %9;"         // state[14] ^= D[4]
            : "+l"(state[10]), "+l"(state[11]), "+l"(state[12]), "+l"(state[13]), "+l"(state[14])
            : "l"(D[0]), "l"(D[1]), "l"(D[2]), "l"(D[3]), "l"(D[4])
        );
        
        // Group 4: state[15-19] ^= D[0-4]
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // state[15] ^= D[0]
            "xor.b64    %1, %1, %6;\n\t"     // state[16] ^= D[1]
            "xor.b64    %2, %2, %7;\n\t"     // state[17] ^= D[2]
            "xor.b64    %3, %3, %8;\n\t"     // state[18] ^= D[3]
            "xor.b64    %4, %4, %9;"         // state[19] ^= D[4]
            : "+l"(state[15]), "+l"(state[16]), "+l"(state[17]), "+l"(state[18]), "+l"(state[19])
            : "l"(D[0]), "l"(D[1]), "l"(D[2]), "l"(D[3]), "l"(D[4])
        );
        
        // Group 5: state[20-24] ^= D[0-4]
        asm volatile (
            "xor.b64    %0, %0, %5;\n\t"     // state[20] ^= D[0]
            "xor.b64    %1, %1, %6;\n\t"     // state[21] ^= D[1]
            "xor.b64    %2, %2, %7;\n\t"     // state[22] ^= D[2]
            "xor.b64    %3, %3, %8;\n\t"     // state[23] ^= D[3]
            "xor.b64    %4, %4, %9;"         // state[24] ^= D[4]
            : "+l"(state[20]), "+l"(state[21]), "+l"(state[22]), "+l"(state[23]), "+l"(state[24])
            : "l"(D[0]), "l"(D[1]), "l"(D[2]), "l"(D[3]), "l"(D[4])
        );
        
        // Rho and Pi steps - fully unrolled with inlined constants
        // y=0
        B[0] = state[0];   // x=0, y=0: yp=(3*0)%5=0
        
        // Optimized rotl64 operations with instruction reordering for ILP
        asm volatile (
            // Phase 1: All shl operations in parallel (no dependencies)
            "shl.b64  %0, %4, 1;\n\t"      // B[10] = state[1] << 1
            "shl.b64  %1, %5, 62;\n\t"     // B[20] = state[2] << 62
            "shl.b64  %2, %6, 28;\n\t"     // B[5] = state[3] << 28
            "shl.b64  %3, %7, 27;\n\t"     // B[15] = state[4] << 27
            
            // Phase 2: All shr operations in parallel (input reuse optimization)
            "shr.b64  %4, %4, 63;\n\t"     // temp1 = state[1] >> 63
            "shr.b64  %5, %5, 2;\n\t"      // temp2 = state[2] >> 2
            "shr.b64  %6, %6, 36;\n\t"     // temp3 = state[3] >> 36
            "shr.b64  %7, %7, 37;\n\t"     // temp4 = state[4] >> 37
            
            // Phase 3: All or operations in parallel
            "or.b64   %0, %0, %4;\n\t"     // B[10] |= temp1
            "or.b64   %1, %1, %5;\n\t"     // B[20] |= temp2
            "or.b64   %2, %2, %6;\n\t"     // B[5] |= temp3
            "or.b64   %3, %3, %7;"         // B[15] |= temp4
            
            : "=l"(B[10]), "=l"(B[20]), "=l"(B[5]), "=l"(B[15])
            : "l"(state[1]), "l"(state[2]), "l"(state[3]), "l"(state[4])
        );
        
        // y=1
        // Optimized rotl64 operations with instruction reordering for ILP
        asm volatile (
            // Phase 1: All shl operations in parallel (no dependencies)
            "shl.b64  %0, %5, 36;\n\t"     // B[16] = state[5] << 36
            "shl.b64  %1, %6, 44;\n\t"     // B[1] = state[6] << 44
            "shl.b64  %2, %7, 6;\n\t"      // B[11] = state[7] << 6
            "shl.b64  %3, %8, 55;\n\t"     // B[21] = state[8] << 55
            "shl.b64  %4, %9, 20;\n\t"     // B[6] = state[9] << 20
            
            // Phase 2: All shr operations in parallel (input reuse optimization)
            "shr.b64  %5, %5, 28;\n\t"     // temp1 = state[5] >> 28
            "shr.b64  %6, %6, 20;\n\t"     // temp2 = state[6] >> 20
            "shr.b64  %7, %7, 58;\n\t"     // temp3 = state[7] >> 58
            "shr.b64  %8, %8, 9;\n\t"      // temp4 = state[8] >> 9
            "shr.b64  %9, %9, 44;\n\t"     // temp5 = state[9] >> 44
            
            // Phase 3: All or operations in parallel
            "or.b64   %0, %0, %5;\n\t"     // B[16] |= temp1
            "or.b64   %1, %1, %6;\n\t"     // B[1] |= temp2
            "or.b64   %2, %2, %7;\n\t"     // B[11] |= temp3
            "or.b64   %3, %3, %8;\n\t"     // B[21] |= temp4
            "or.b64   %4, %4, %9;"         // B[6] |= temp5
            
            : "=l"(B[16]), "=l"(B[1]), "=l"(B[11]), "=l"(B[21]), "=l"(B[6])
            : "l"(state[5]), "l"(state[6]), "l"(state[7]), "l"(state[8]), "l"(state[9])
        );
        
        // y=2
        // Optimized rotl64 operations with instruction reordering for ILP
        asm volatile (
            // Phase 1: All shl operations in parallel (no dependencies)
            "shl.b64  %0, %5, 3;\n\t"      // B[7] = state[10] << 3
            "shl.b64  %1, %6, 10;\n\t"     // B[17] = state[11] << 10
            "shl.b64  %2, %7, 43;\n\t"     // B[2] = state[12] << 43
            "shl.b64  %3, %8, 25;\n\t"     // B[12] = state[13] << 25
            "shl.b64  %4, %9, 39;\n\t"     // B[22] = state[14] << 39
            
            // Phase 2: All shr operations in parallel (input reuse optimization)
            "shr.b64  %5, %5, 61;\n\t"     // temp1 = state[10] >> 61
            "shr.b64  %6, %6, 54;\n\t"     // temp2 = state[11] >> 54
            "shr.b64  %7, %7, 21;\n\t"     // temp3 = state[12] >> 21
            "shr.b64  %8, %8, 39;\n\t"     // temp4 = state[13] >> 39
            "shr.b64  %9, %9, 25;\n\t"     // temp5 = state[14] >> 25
            
            // Phase 3: All or operations in parallel
            "or.b64   %0, %0, %5;\n\t"     // B[7] |= temp1
            "or.b64   %1, %1, %6;\n\t"     // B[17] |= temp2
            "or.b64   %2, %2, %7;\n\t"     // B[2] |= temp3
            "or.b64   %3, %3, %8;\n\t"     // B[12] |= temp4
            "or.b64   %4, %4, %9;"         // B[22] |= temp5
            
            : "=l"(B[7]), "=l"(B[17]), "=l"(B[2]), "=l"(B[12]), "=l"(B[22])
            : "l"(state[10]), "l"(state[11]), "l"(state[12]), "l"(state[13]), "l"(state[14])
        );
        
        // y=3
        // Optimized rotl64 operations with instruction reordering for ILP
        asm volatile (
            // Phase 1: All shl operations in parallel (no dependencies)
            "shl.b64  %0, %5, 41;\n\t"     // B[23] = state[15] << 41
            "shl.b64  %1, %6, 45;\n\t"     // B[8] = state[16] << 45
            "shl.b64  %2, %7, 15;\n\t"     // B[18] = state[17] << 15
            "shl.b64  %3, %8, 21;\n\t"     // B[3] = state[18] << 21
            "shl.b64  %4, %9, 8;\n\t"      // B[13] = state[19] << 8
            
            // Phase 2: All shr operations in parallel (input reuse optimization)
            "shr.b64  %5, %5, 23;\n\t"     // temp1 = state[15] >> 23
            "shr.b64  %6, %6, 19;\n\t"     // temp2 = state[16] >> 19
            "shr.b64  %7, %7, 49;\n\t"     // temp3 = state[17] >> 49
            "shr.b64  %8, %8, 43;\n\t"     // temp4 = state[18] >> 43
            "shr.b64  %9, %9, 56;\n\t"     // temp5 = state[19] >> 56
            
            // Phase 3: All or operations in parallel
            "or.b64   %0, %0, %5;\n\t"     // B[23] |= temp1
            "or.b64   %1, %1, %6;\n\t"     // B[8] |= temp2
            "or.b64   %2, %2, %7;\n\t"     // B[18] |= temp3
            "or.b64   %3, %3, %8;\n\t"     // B[3] |= temp4
            "or.b64   %4, %4, %9;"         // B[13] |= temp5
            
            : "=l"(B[23]), "=l"(B[8]), "=l"(B[18]), "=l"(B[3]), "=l"(B[13])
            : "l"(state[15]), "l"(state[16]), "l"(state[17]), "l"(state[18]), "l"(state[19])
        );
        
        // y=4
        // Optimized rotl64 operations with instruction reordering for ILP
        asm volatile (
            // Phase 1: All shl operations in parallel (no dependencies)
            "shl.b64  %0, %5, 18;\n\t"     // B[14] = state[20] << 18
            "shl.b64  %1, %6, 2;\n\t"      // B[24] = state[21] << 2
            "shl.b64  %2, %7, 61;\n\t"     // B[9] = state[22] << 61
            "shl.b64  %3, %8, 56;\n\t"     // B[19] = state[23] << 56
            "shl.b64  %4, %9, 14;\n\t"     // B[4] = state[24] << 14
            
            // Phase 2: All shr operations in parallel (input reuse optimization)
            "shr.b64  %5, %5, 46;\n\t"     // temp1 = state[20] >> 46
            "shr.b64  %6, %6, 62;\n\t"     // temp2 = state[21] >> 62
            "shr.b64  %7, %7, 3;\n\t"      // temp3 = state[22] >> 3
            "shr.b64  %8, %8, 8;\n\t"      // temp4 = state[23] >> 8
            "shr.b64  %9, %9, 50;\n\t"     // temp5 = state[24] >> 50
            
            // Phase 3: All or operations in parallel
            "or.b64   %0, %0, %5;\n\t"     // B[14] |= temp1
            "or.b64   %1, %1, %6;\n\t"     // B[24] |= temp2
            "or.b64   %2, %2, %7;\n\t"     // B[9] |= temp3
            "or.b64   %3, %3, %8;\n\t"     // B[19] |= temp4
            "or.b64   %4, %4, %9;"         // B[4] |= temp5
            
            : "=l"(B[14]), "=l"(B[24]), "=l"(B[9]), "=l"(B[19]), "=l"(B[4])
            : "l"(state[20]), "l"(state[21]), "l"(state[22]), "l"(state[23]), "l"(state[24])
        );
        
        // Chi step - fully unrolled with no modulo operations
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            int yy = y * 5;
            // Direct computation without temporary array
            state[yy + 0] = B[yy + 0] ^ ((~B[yy + 1]) & B[yy + 2]);
            state[yy + 1] = B[yy + 1] ^ ((~B[yy + 2]) & B[yy + 3]);
            state[yy + 2] = B[yy + 2] ^ ((~B[yy + 3]) & B[yy + 4]);
            state[yy + 3] = B[yy + 3] ^ ((~B[yy + 4]) & B[yy + 0]);
            state[yy + 4] = B[yy + 4] ^ ((~B[yy + 0]) & B[yy + 1]);
     
        }
        
        
        // Iota step
        state[0] ^= KECCAK_RC[round];
    }
}


// Ultra-fast zero 17 uint64_t elements using PTX assembly
__device__ __forceinline__ void zero_17(uint64_t* ptr) {
    asm volatile (
        "mov.b64 %0, 0;\n\t"
        "mov.b64 %1, 0;\n\t"
        "mov.b64 %2, 0;\n\t"
        "mov.b64 %3, 0;\n\t"
        "mov.b64 %4, 0;\n\t"
        "mov.b64 %5, 0;\n\t"
        "mov.b64 %6, 0;\n\t"
        "mov.b64 %7, 0;\n\t"
        "mov.b64 %8, 0;\n\t"
        "mov.b64 %9, 0;\n\t"
        "mov.b64 %10, 0;\n\t"
        "mov.b64 %11, 0;\n\t"
        "mov.b64 %12, 0;\n\t"
        "mov.b64 %13, 0;\n\t"
        "mov.b64 %14, 0;\n\t"
        "mov.b64 %15, 0;\n\t"
        "mov.b64 %16, 0;"
        : "=l"(ptr[0]), "=l"(ptr[1]), "=l"(ptr[2]), "=l"(ptr[3]),
          "=l"(ptr[4]), "=l"(ptr[5]), "=l"(ptr[6]), "=l"(ptr[7]),
          "=l"(ptr[8]), "=l"(ptr[9]), "=l"(ptr[10]), "=l"(ptr[11]),
          "=l"(ptr[12]), "=l"(ptr[13]), "=l"(ptr[14]), "=l"(ptr[15]),
          "=l"(ptr[16])
        :
        : "memory"
    );
}

// Compute Ethereum address from affine coordinates (last 20 bytes of Keccak-256)
__device__ __forceinline__ void eth_address(const uint32_t* xa, const uint32_t* ya, uint8_t* addr20) {
    // Inline Keccak-256 for 64-byte input
    uint64_t state[25];
    
    // Initialize state to zero using optimized PTX assembly
    zero_17(&state[8]);
    
    // Directly pack coordinates into state array 
    // Need to convert xa,ya (little-endian uint32 arrays) to big-endian byte sequence
    // then load as little-endian uint64 for keccak
    
    // x coordinates (32 bytes): xa[7] xa[6] xa[5] xa[4] xa[3] xa[2] xa[1] xa[0] in big-endian
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t w_hi = xa[7 - i * 2];
        uint32_t w_lo = xa[6 - i * 2];
        
        uint32_t bytes_hi, bytes_lo;
        
        asm volatile (
            "prmt.b32 %0, %2, 0, 0x0123;\n\t"
            "prmt.b32 %1, %3, 0, 0x0123;"
            : "=r"(bytes_hi), "=r"(bytes_lo)
            : "r"(w_hi), "r"(w_lo)
        );
        
        state[i] = ((uint64_t)bytes_hi) | (((uint64_t)bytes_lo) << 32);
    }
    
    // y coordinates (32 bytes): ya[7] ya[6] ya[5] ya[4] ya[3] ya[2] ya[1] ya[0] in big-endian
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t w_hi = ya[7 - i * 2];
        uint32_t w_lo = ya[6 - i * 2];
        
        uint32_t bytes_hi, bytes_lo;
        
        asm volatile (
            "prmt.b32 %0, %2, 0, 0x0123;\n\t"
            "prmt.b32 %1, %3, 0, 0x0123;"
            : "=r"(bytes_hi), "=r"(bytes_lo)
            : "r"(w_hi), "r"(w_lo)
        );
        
        state[4 + i] = ((uint64_t)bytes_hi) | (((uint64_t)bytes_lo) << 32);
    }
    
    // Padding for 64-byte message (rate = 136 bytes for Keccak-256)
    state[8] ^= 0x01ULL;
    state[16] ^= 0x8000000000000000ULL;
    
    // Apply Keccak-f[1600]
    keccak_f1600(state);
    
    // Extract last 20 bytes directly from state using optimized operations
    // Original: hash[0-7]=state[0], hash[8-15]=state[1], hash[16-23]=state[2], hash[24-31]=state[3]
    // Need: addr20[0-19] = hash[12-31]
    // addr20[0-3] = hash[12-15] (state[1] upper 4 bytes)
    // addr20[4-11] = hash[16-23] (state[2] all 8 bytes)  
    // addr20[12-19] = hash[24-31] (state[3] all 8 bytes)
    
    uint64_t temp1, temp2, temp3;
    
    asm volatile (
        // Extract upper 4 bytes from state[1] (hash[12-15])
        "mov.b64    %0, %3;\n\t"
        "shr.u64    %0, %0, 32;\n\t"
        
        // Load state[2] and state[3] directly
        "mov.b64    %1, %4;\n\t"
        "mov.b64    %2, %5;"
        
        : "=l"(temp1), "=l"(temp2), "=l"(temp3)
        : "l"(state[1]), "l"(state[2]), "l"(state[3])
    );
    
    // Store using correct address offsets
    *(uint32_t*)addr20 = (uint32_t)temp1;           // addr20[0-3]
    *(uint64_t*)(addr20+4) = temp2;                 // addr20[4-11]
    *(uint64_t*)(addr20+12) = temp3;                // addr20[12-19]
}


// Check if address matches head and/or tail patterns (original version)
__device__ __forceinline__ bool check_vanity_pattern(
    const uint8_t* addr20,
    const uint8_t* head_pattern, uint32_t head_nibbles,
    const uint8_t* tail_pattern, uint32_t tail_nibbles) {
    return true;
}

// Optimized pattern check using compile-time macros
#ifdef USE_PATTERN_OPTIMIZATION

// Default macro definitions (overridden by compiler defines)
#ifndef CHECK_HEAD_PATTERN
#define CHECK_HEAD_PATTERN(addr) (true)
#endif

#ifndef CHECK_TAIL_PATTERN
#define CHECK_TAIL_PATTERN(addr) (true)
#endif

// Ultra-fast pattern check with compile-time patterns
__device__ __forceinline__ bool check_vanity_pattern_optimized(const uint8_t* addr20) {
    return CHECK_HEAD_PATTERN(addr20) && CHECK_TAIL_PATTERN(addr20);
}

#endif // USE_PATTERN_OPTIMIZATION


#endif // KECCAK256_CUH