#ifndef KECCAK256_CUH
#define KECCAK256_CUH

#include "math_utils.cuh"

// Keccak-256 round constants
__constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rho offsets for Keccak
__constant__ uint8_t KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

// Rotate left 64-bit
__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint8_t n) {
    return (x << n) | (x >> (64 - n));
}

// Load 64-bit little-endian
__device__ __forceinline__ uint64_t load64_le(const uint8_t* src) {
    uint64_t v = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        v |= ((uint64_t)src[i]) << (8 * i);
    }
    return v;
}

// Store 64-bit little-endian
__device__ __forceinline__ void store64_le(uint8_t* dst, uint64_t v) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dst[i] = (uint8_t)((v >> (8 * i)) & 0xFF);
    }
}

// Keccak-f[1600] permutation (optimized for registers)
__device__ __forceinline__ void keccak_f1600(uint64_t state[25]) {
    uint64_t C[5], D[5], B[25];
    
    #pragma unroll 1
    for (int round = 0; round < 24; round++) {
        // Theta step
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = rotl64(C[(x + 1) % 5], 1) ^ C[(x + 4) % 5];
        }
        
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            #pragma unroll
            for (int x = 0; x < 5; x++) {
                state[y * 5 + x] ^= D[x];
            }
        }
        
        // Rho and Pi steps
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            #pragma unroll
            for (int x = 0; x < 5; x++) {
                int idx = x + 5 * y;
                int xp = y;
                int yp = (2 * x + 3 * y) % 5;
                B[xp + 5 * yp] = rotl64(state[idx], KECCAK_RHO[idx]);
            }
        }
        
        // Chi step
        #pragma unroll
        for (int y = 0; y < 5; y++) {
            int yy = y * 5;
            uint64_t t[5];
            #pragma unroll
            for (int x = 0; x < 5; x++) {
                t[x] = B[yy + x];
            }
            #pragma unroll
            for (int x = 0; x < 5; x++) {
                state[yy + x] = t[x] ^ ((~t[(x + 1) % 5]) & t[(x + 2) % 5]);
            }
        }
        
        // Iota step
        state[0] ^= KECCAK_RC[round];
    }
}

// Keccak-256 for 64-byte input (optimized for Ethereum public keys)
__device__ __forceinline__ void keccak256_64(const uint8_t* input, uint8_t* output) {
    uint64_t state[25];
    
    // Initialize state to zero
    #pragma unroll
    for (int i = 0; i < 25; i++) {
        state[i] = 0;
    }
    
    // Absorb 64 bytes (8 * 8-byte words)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = load64_le(input + i * 8);
    }
    
    // Padding for 64-byte message (rate = 136 bytes for Keccak-256)
    // Append domain separator (0x01) at byte 64
    state[8] ^= 0x01ULL;
    // Append final bit at byte 135 (bit 1087)
    state[16] ^= 0x8000000000000000ULL;
    
    // Apply Keccak-f[1600]
    keccak_f1600(state);
    
    // Squeeze first 32 bytes (256 bits)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        store64_le(output + i * 8, state[i]);
    }
}

// Generic Keccak-256 (for arbitrary length input)
__device__ void keccak256(const uint8_t* input, uint64_t len, uint8_t* output) {
    uint64_t state[25];
    const uint64_t rate = 136; // Rate for Keccak-256 (1088 bits / 8)
    
    // Initialize state
    #pragma unroll
    for (int i = 0; i < 25; i++) {
        state[i] = 0;
    }
    
    // Absorb input
    uint64_t absorbed = 0;
    while (absorbed < len) {
        uint64_t to_absorb = (rate < len - absorbed) ? rate : len - absorbed;
        
        // Absorb block
        for (uint64_t i = 0; i < to_absorb; i += 8) {
            uint64_t bytes = (to_absorb - i < 8) ? to_absorb - i : 8;
            uint64_t word = 0;
            for (uint64_t j = 0; j < bytes; j++) {
                word |= ((uint64_t)input[absorbed + i + j]) << (8 * j);
            }
            state[i / 8] ^= word;
        }
        
        absorbed += to_absorb;
        
        if (to_absorb == rate) {
            keccak_f1600(state);
        }
    }
    
    // Padding
    uint64_t pad_pos = absorbed % rate;
    state[pad_pos / 8] ^= 0x01ULL << ((pad_pos % 8) * 8);
    state[(rate - 1) / 8] ^= 0x8000000000000000ULL;
    
    // Final permutation
    keccak_f1600(state);
    
    // Squeeze output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        store64_le(output + i * 8, state[i]);
    }
}

// Compute Ethereum address from public key (last 20 bytes of Keccak-256)
__device__ __forceinline__ void eth_address(const uint8_t* pubkey64, uint8_t* addr20) {
    uint8_t hash[32];
    keccak256_64(pubkey64, hash);
    
    // Copy last 20 bytes
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        addr20[i] = hash[12 + i];
    }
}

// Check if address matches vanity pattern
__device__ __forceinline__ bool check_vanity(const uint8_t* addr20, uint8_t nibble, uint32_t nibble_count) {
    uint8_t want_byte = (nibble << 4) | nibble;
    uint32_t full_bytes = nibble_count >> 1;
    uint32_t has_half = nibble_count & 1;
    
    // Check full bytes
    for (uint32_t i = 0; i < full_bytes; i++) {
        if (addr20[i] != want_byte) {
            return false;
        }
    }
    
    // Check half byte if needed
    if (has_half) {
        if ((addr20[full_bytes] >> 4) != nibble) {
            return false;
        }
    }
    
    return true;
}

// Check if address matches head and/or tail patterns (original version)
__device__ __forceinline__ bool check_vanity_pattern(
    const uint8_t* addr20,
    const uint8_t* head_pattern, uint32_t head_nibbles,
    const uint8_t* tail_pattern, uint32_t tail_nibbles) {
    
    // Check head pattern if provided
    if (head_nibbles > 0 && head_pattern) {
        uint32_t full_bytes = head_nibbles >> 1;
        uint32_t has_half = head_nibbles & 1;
        
        // Check full bytes
        for (uint32_t i = 0; i < full_bytes; i++) {
            if (addr20[i] != head_pattern[i]) {
                return false;
            }
        }
        
        // Check half byte if needed
        if (has_half) {
            uint8_t addr_nibble = (addr20[full_bytes] >> 4) & 0xF;
            uint8_t pattern_nibble = (head_pattern[full_bytes] >> 4) & 0xF;
            if (addr_nibble != pattern_nibble) {
                return false;
            }
        }
    }
    
    // Check tail pattern if provided
    if (tail_nibbles > 0 && tail_pattern) {
        uint32_t full_bytes = tail_nibbles >> 1;
        uint32_t has_half = tail_nibbles & 1;
        
        // Calculate starting position from the end
        uint32_t start_pos;
        if (has_half) {
            start_pos = 20 - full_bytes - 1;
        } else {
            start_pos = 20 - full_bytes;
        }
        
        // Check half byte first if needed (from the end)
        if (has_half) {
            uint8_t addr_nibble = addr20[19] & 0xF;
            uint8_t pattern_nibble = tail_pattern[full_bytes] & 0xF;
            if (addr_nibble != pattern_nibble) {
                return false;
            }
        }
        
        // Check full bytes from the end
        for (uint32_t i = 0; i < full_bytes; i++) {
            uint32_t addr_pos = start_pos + i + (has_half ? 1 : 0);
            if (addr20[addr_pos] != tail_pattern[i]) {
                return false;
            }
        }
    }
    
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