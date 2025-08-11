#include <metal_stdlib>
using namespace metal;

// Keccak-f[1600] permutation and a fixed-size Keccak-256 (Ethereum Keccak) hash
// for exactly 64-byte inputs, producing a 32-byte output.

inline ulong rotl64(ulong x, ushort n) {
    return (x << n) | (x >> (64 - n));
}

constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

// Rho rotation offsets r[x + 5*y] with x,y in 0..4
constant ushort RHO[25] = {
    // y = 0
     0,  1, 62, 28, 27,
    // y = 1
    36, 44,  6, 55, 20,
    // y = 2
     3, 10, 43, 25, 39,
    // y = 3
    41, 45, 15, 21,  8,
    // y = 4
    18,  2, 61, 56, 14
};

inline void keccak_f1600(thread ulong a[25]) {
    ulong b[25];
    ulong c[5];
    ulong d[5];

    #pragma unroll
    for (ushort round = 0; round < 24; ++round) {
        // Theta
        #pragma unroll
        for (ushort x = 0; x < 5; ++x) {
            c[x] = a[x + 0] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        #pragma unroll
        for (ushort x = 0; x < 5; ++x) {
            d[x] = rotl64(c[(x + 1) % 5], 1) ^ c[(x + 4) % 5];
        }
        #pragma unroll
        for (ushort y = 0; y < 5; ++y) {
            ushort yy = y * 5;
            a[yy + 0] ^= d[0];
            a[yy + 1] ^= d[1];
            a[yy + 2] ^= d[2];
            a[yy + 3] ^= d[3];
            a[yy + 4] ^= d[4];
        }

        // Rho + Pi: b[x',y'] = ROT(a[x,y], RHO[x,y]) where x' = y, y' = (2x + 3y) mod 5
        #pragma unroll
        for (ushort y = 0; y < 5; ++y) {
            #pragma unroll
            for (ushort x = 0; x < 5; ++x) {
                ushort idx = x + 5 * y;
                ushort xp = y;
                ushort yp = (ushort)((2 * x + 3 * y) % 5);
                b[xp + 5 * yp] = rotl64(a[idx], RHO[idx]);
            }
        }

        // Chi
        #pragma unroll
        for (ushort y = 0; y < 5; ++y) {
            ushort yy = y * 5;
            ulong b0 = b[yy + 0];
            ulong b1 = b[yy + 1];
            ulong b2 = b[yy + 2];
            ulong b3 = b[yy + 3];
            ulong b4 = b[yy + 4];
            a[yy + 0] = b0 ^ ((~b1) & b2);
            a[yy + 1] = b1 ^ ((~b2) & b3);
            a[yy + 2] = b2 ^ ((~b3) & b4);
            a[yy + 3] = b3 ^ ((~b4) & b0);
            a[yy + 4] = b4 ^ ((~b0) & b1);
        }

        // Iota
        a[0] ^= RC[round];
    }
}

// Helper: load 64-bit little-endian from 8 bytes
inline ulong load64_device(const device uchar *src) {
    ulong v = 0;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) {
        v |= ((ulong)src[i]) << (8 * i);
    }
    return v;
}

// Helper: store 64-bit little-endian to 8 bytes
inline void store64(ulong v, device uchar *dst) {
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) {
        dst[i] = (uchar)((v >> (8 * i)) & 0xFF);
    }
}

// Helper: load 64-bit from thread address space
inline ulong load64_thread(const thread uchar *src) {
    ulong v = 0;
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) {
        v |= ((ulong)src[i]) << (8 * i);
    }
    return v;
}

kernel void keccak256_kernel(
    device const uchar *inBytes           [[ buffer(0) ]],  // 64-byte input per thread
    device uchar *outBytes                [[ buffer(1) ]],  // 32-byte output per thread
    uint gid                               [[ thread_position_in_grid ]])
{
    // We process exactly one 64-byte message per thread
    const device uchar *msg = inBytes + (gid * 64);
    device uchar *out = outBytes + (gid * 32);

    // Initialize state (25 lanes = 200 bytes)
    ulong A[25];
    #pragma unroll
    for (ushort i = 0; i < 25; ++i) A[i] = 0UL;

    // Directly absorb 64-byte message (little-endian) into lanes A[0..7]
    #pragma unroll
    for (ushort i = 0; i < 8; ++i) {
        A[i] ^= load64_device(msg + (i * 8));
    }
    // Keccak padding for 64-byte message within rate (136B)
    // XOR domain separation bit at byte 64 -> lane 8, bit 0
    A[8] ^= 0x01UL;
    // Final bit of pad10*1 at byte 135 bit 7 -> lane 16, bit 63
    A[16] ^= 0x8000000000000000UL;

    // Permutation
    keccak_f1600(A);

    // Squeeze first 32 bytes of the state (little-endian of lanes 0..3)
    #pragma unroll
    for (ushort i = 0; i < 4; ++i) {
        store64(A[i], out + (i * 8));
    }
}


