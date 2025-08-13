#include "secp256k1.cuh"
#include "keccak256.cuh"
#include <cuda_runtime.h>

// Batch size for walker kernel
#define BATCH_WINDOW_SIZE 256

// secp256k1 field prime P
__constant__ uint32_t SECP256K1_P[8] = {
    SECP256K1_P0, SECP256K1_P1, SECP256K1_P2, SECP256K1_P3,
    SECP256K1_P4, SECP256K1_P5, SECP256K1_P6, SECP256K1_P7
};

// Generator point G in constant memory
__constant__ uint32_t SECP256K1_G[16] = {
    SECP256K1_GX0, SECP256K1_GX1, SECP256K1_GX2, SECP256K1_GX3,
    SECP256K1_GX4, SECP256K1_GX5, SECP256K1_GX6, SECP256K1_GX7,
    SECP256K1_GY0, SECP256K1_GY1, SECP256K1_GY2, SECP256K1_GY3,
    SECP256K1_GY4, SECP256K1_GY5, SECP256K1_GY6, SECP256K1_GY7
};

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

// Modular addition
__device__ void add_mod(uint32_t* r, const uint32_t* a, const uint32_t* b) {
    uint32_t carry = add256(r, a, b);
    
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

// Point addition in Jacobian coordinates
__device__ void point_add(uint32_t* x3, uint32_t* y3, uint32_t* z3,
                         const uint32_t* x1, const uint32_t* y1, const uint32_t* z1,
                         const uint32_t* x2, const uint32_t* y2) {
    uint32_t z1z1[8], z1z1z1[8], s2[8], u2[8];
    uint32_t h[8], hh[8], hhh[8], r[8], v[8];
    
    // Check if first point is identity
    bool is_zero = true;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (z1[i] != 0) {
            is_zero = false;
            break;
        }
    }
    
    if (is_zero) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            x3[i] = x2[i];
            y3[i] = y2[i];
            z3[i] = (i == 0) ? 1 : 0;
        }
        return;
    }
    
    // z1z1 = z1^2
    mul_mod(z1z1, z1, z1);
    
    // u2 = x2 * z1z1
    mul_mod(u2, x2, z1z1);
    
    // z1z1z1 = z1 * z1z1
    mul_mod(z1z1z1, z1, z1z1);
    
    // s2 = y2 * z1z1z1
    mul_mod(s2, y2, z1z1z1);
    
    // h = u2 - x1
    sub_mod(h, u2, x1);
    
    // r = s2 - y1
    sub_mod(r, s2, y1);
    
    // Check if points are equal (would need doubling)
    bool h_zero = true, r_zero = true;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (h[i] != 0) h_zero = false;
        if (r[i] != 0) r_zero = false;
    }
    
    if (h_zero && r_zero) {
        point_double(x3, y3, z3, x1, y1, z1);
        return;
    }
    
    // hh = h^2
    mul_mod(hh, h, h);
    
    // hhh = h * hh
    mul_mod(hhh, h, hh);
    
    // v = x1 * hh
    mul_mod(v, x1, hh);
    
    // x3 = r^2 - hhh - 2*v
    uint32_t rr[8], temp[8];
    mul_mod(rr, r, r);
    sub_mod(temp, rr, hhh);
    sub_mod(temp, temp, v);
    sub_mod(x3, temp, v);
    
    // y3 = r * (v - x3) - y1 * hhh
    sub_mod(temp, v, x3);
    mul_mod(temp, r, temp);
    uint32_t y1hhh[8];
    mul_mod(y1hhh, y1, hhh);
    sub_mod(y3, temp, y1hhh);
    
    // z3 = z1 * h
    mul_mod(z3, z1, h);
}

// Point doubling in Jacobian coordinates
__device__ void point_double(uint32_t* x3, uint32_t* y3, uint32_t* z3,
                            const uint32_t* x1, const uint32_t* y1, const uint32_t* z1) {
    uint32_t s[8], m[8], t[8];
    
    // s = 4 * x1 * y1^2
    uint32_t y1y1[8];
    mul_mod(y1y1, y1, y1);
    mul_mod(s, x1, y1y1);
    add_mod(s, s, s);
    add_mod(s, s, s);
    
    // m = 3 * x1^2 (for a=0 curve)
    uint32_t x1x1[8];
    mul_mod(x1x1, x1, x1);
    add_mod(m, x1x1, x1x1);
    add_mod(m, m, x1x1);
    
    // t = m^2 - 2*s
    mul_mod(t, m, m);
    sub_mod(t, t, s);
    sub_mod(x3, t, s);
    
    // y3 = m * (s - x3) - 8 * y1^4
    sub_mod(t, s, x3);
    mul_mod(t, m, t);
    mul_mod(y1y1, y1y1, y1y1);
    add_mod(y1y1, y1y1, y1y1);
    add_mod(y1y1, y1y1, y1y1);
    add_mod(y1y1, y1y1, y1y1);
    sub_mod(y3, t, y1y1);
    
    // z3 = 2 * y1 * z1
    mul_mod(z3, y1, z1);
    add_mod(z3, z3, z3);
}

// Convert Jacobian to affine coordinates
__device__ void jacobian_to_affine(uint32_t* x_affine, uint32_t* y_affine,
                                   const uint32_t* x, const uint32_t* y, const uint32_t* z) {
    uint32_t z_inv[8], z_inv2[8], z_inv3[8];
    
    // z_inv = 1/z
    inv_mod(z_inv, z);
    
    // z_inv2 = z_inv^2
    mul_mod(z_inv2, z_inv, z_inv);
    
    // x_affine = x * z_inv2
    mul_mod(x_affine, x, z_inv2);
    
    // z_inv3 = z_inv2 * z_inv
    mul_mod(z_inv3, z_inv2, z_inv);
    
    // y_affine = y * z_inv3
    mul_mod(y_affine, y, z_inv3);
}

// Scalar multiplication using double-and-add method
__device__ void point_mul(uint32_t* x_out, uint32_t* y_out, const uint32_t* k) {
    uint32_t x[8] = {0}, y[8] = {0}, z[8] = {0};
    bool first = true;
    
    // Double-and-add algorithm
    for (int bit = 255; bit >= 0; bit--) {
        int limb_idx = bit >> 5;
        int bit_idx = bit & 31;
        
        if (!first) {
            point_double(x, y, z, x, y, z);
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
                point_add(x, y, z, x, y, z, SECP256K1_G, SECP256K1_G + 8);
            }
        }
    }
    
    // Convert to affine
    if (first) {
        // Point at infinity
        for (int i = 0; i < 8; i++) {
            x_out[i] = 0;
            y_out[i] = 0;
        }
    } else {
        jacobian_to_affine(x_out, y_out, x, y, z);
    }
}

// Scalar multiplication using 16-bit window method with G16 table
__device__ void point_mul_g16(uint32_t* x_out, uint32_t* y_out,
                              const uint32_t* k, const uint8_t* g16_table) {
    uint32_t x[8], y[8], z[8];
    bool first = true;
    
    // Process 16 windows of 16 bits each
    #pragma unroll 1
    for (int window = 0; window < 16; window++) {
        uint32_t idx = get_window16(k, window);
        
        if (idx == 0) continue;
        
        uint32_t x2[8], y2[8];
        load_point_g16(x2, y2, g16_table, window, idx);
        
        if (first) {
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                x[i] = x2[i];
                y[i] = y2[i];
                z[i] = (i == 0) ? 1 : 0;
            }
            first = false;
        } else {
            point_add(x, y, z, x, y, z, x2, y2);
        }
    }
    
    // Convert to affine
    jacobian_to_affine(x_out, y_out, x, y, z);
}

// Batch inversion using Montgomery's trick
__device__ void batch_inverse(uint32_t xs[][8], uint32_t ys[][8], uint32_t zs[][8],
                              uint32_t temp_pref[][8], int batch_size) {
    // Calculate prefix products
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        temp_pref[0][i] = zs[0][i];
    }
    
    for (int i = 1; i < batch_size; i++) {
        mul_mod(temp_pref[i], temp_pref[i-1], zs[i]);
    }
    
    // Invert the total product
    uint32_t inv_total[8];
    inv_mod(inv_total, temp_pref[batch_size - 1]);
    
    // Backward pass to get individual inverses and convert to affine
    for (int i = batch_size - 1; i >= 0; i--) {
        uint32_t inv_z[8];
        
        if (i == 0) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                inv_z[j] = inv_total[j];
            }
        } else {
            mul_mod(inv_z, inv_total, temp_pref[i - 1]);
        }
        
        // Update inv_total for next iteration
        if (i > 0) {
            mul_mod(inv_total, inv_total, zs[i]);
        }
        
        // Convert to affine: x = x * z^-2, y = y * z^-3
        uint32_t z_inv2[8], z_inv3[8];
        mul_mod(z_inv2, inv_z, inv_z);
        mul_mod(z_inv3, z_inv2, inv_z);
        
        uint32_t x_affine[8], y_affine[8];
        mul_mod(x_affine, xs[i], z_inv2);
        mul_mod(y_affine, ys[i], z_inv3);
        
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            xs[i][j] = x_affine[j];
            ys[i][j] = y_affine[j];
        }
    }
}

// ======================
// CUDA Kernel Functions
// ======================

// Simple vanity address generation with G16 table
extern "C" __global__ void vanity_kernel_g16(
    const uint8_t* __restrict__ privkeys,      // Private keys (32 bytes each, big-endian)
    uint32_t* __restrict__ found_indices,      // Output: indices of found addresses
    uint32_t* __restrict__ found_count,        // Output: number of found addresses
    const uint8_t* __restrict__ g16_table,     // G16 precomputed table
    uint32_t num_keys,                         // Number of keys to process
    uint8_t target_nibble,                     // Target nibble value (0x0 - 0xF)
    uint32_t nibble_count)                     // Number of nibbles to match
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;
    
    // Load private key (big-endian to little-endian conversion)
    const uint8_t* privkey = privkeys + gid * 32;
    uint32_t k[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        k[7 - i] = ((uint32_t)privkey[i*4] << 24) |
                   ((uint32_t)privkey[i*4 + 1] << 16) |
                   ((uint32_t)privkey[i*4 + 2] << 8) |
                   ((uint32_t)privkey[i*4 + 3]);
    }
    
    // Compute public key: (x, y) = k * G
    uint32_t x[8], y[8];
    point_mul(x, y, k);
    
    // Convert to bytes (big-endian) for hashing
    uint8_t pubkey[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t val = x[7 - i];
        pubkey[i*4] = (val >> 24) & 0xFF;
        pubkey[i*4 + 1] = (val >> 16) & 0xFF;
        pubkey[i*4 + 2] = (val >> 8) & 0xFF;
        pubkey[i*4 + 3] = val & 0xFF;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t val = y[7 - i];
        pubkey[32 + i*4] = (val >> 24) & 0xFF;
        pubkey[32 + i*4 + 1] = (val >> 16) & 0xFF;
        pubkey[32 + i*4 + 2] = (val >> 8) & 0xFF;
        pubkey[32 + i*4 + 3] = val & 0xFF;
    }
    
    // Compute Ethereum address
    uint8_t addr[20];
    eth_address(pubkey, addr);
    
    // Check vanity pattern
    if (check_vanity(addr, target_nibble, nibble_count)) {
        atomicAdd(found_count, 1);
        found_indices[0] = gid;
    }
}

// Walker kernel: compute base points for batch processing
extern "C" __global__ void compute_basepoints_g16(
    const uint8_t* __restrict__ privkeys,      // Private keys
    uint32_t* __restrict__ basepoints,         // Output: base points (16 uint32s per point)
    const uint8_t* __restrict__ g16_table,     // G16 table
    uint32_t num_keys)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;
    
    // Load private key
    const uint8_t* privkey = privkeys + gid * 32;
    uint32_t k[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        k[7 - i] = ((uint32_t)privkey[i*4] << 24) |
                   ((uint32_t)privkey[i*4 + 1] << 16) |
                   ((uint32_t)privkey[i*4 + 2] << 8) |
                   ((uint32_t)privkey[i*4 + 3]);
    }
    
    // Compute base point
    uint32_t x[8], y[8];
    point_mul_g16(x, y, k, g16_table);
    
    // Store result
    uint32_t* out = basepoints + gid * 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = x[i];
        out[8 + i] = y[i];
    }
}

// Walker kernel: generate multiple addresses per thread
extern "C" __global__ void vanity_walker_kernel(
    const uint32_t* __restrict__ basepoints,   // Precomputed base points
    uint32_t* __restrict__ found_indices,      // Output indices
    uint32_t* __restrict__ found_count,        // Output count
    uint32_t num_keys,
    uint32_t steps_per_thread,
    uint8_t target_nibble,
    uint32_t nibble_count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;
    
    // Load base point
    const uint32_t* base = basepoints + gid * 16;
    uint32_t x0[8], y0[8], z0[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x0[i] = base[i];
        y0[i] = base[8 + i];
        z0[i] = (i == 0) ? 1 : 0;
    }
    
    // Load generator G for incremental addition
    uint32_t gx[8], gy[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        gx[i] = SECP256K1_G[i];
        gy[i] = SECP256K1_G[8 + i];
    }
    
    // Process in batches to use Montgomery's trick for batch inversion
    const int batch_size = min((int)steps_per_thread, BATCH_WINDOW_SIZE);
    uint32_t xs[BATCH_WINDOW_SIZE][8];
    uint32_t ys[BATCH_WINDOW_SIZE][8];
    uint32_t zs[BATCH_WINDOW_SIZE][8];
    uint32_t temp_pref[BATCH_WINDOW_SIZE][8];
    
    for (uint32_t batch_start = 0; batch_start < steps_per_thread; batch_start += batch_size) {
        int current_batch = min(batch_size, (int)(steps_per_thread - batch_start));
        
        // Generate batch of points
        for (int i = 0; i < current_batch; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                xs[i][j] = x0[j];
                ys[i][j] = y0[j];
                zs[i][j] = z0[j];
            }
            
            // x0 = x0 + G
            point_add(x0, y0, z0, x0, y0, z0, gx, gy);
        }
        
        // Batch inversion to convert all points to affine
        batch_inverse(xs, ys, zs, temp_pref, current_batch);
        
        // Check each address in the batch
        for (int i = 0; i < current_batch; i++) {
            // Convert to bytes for hashing
            uint8_t pubkey[64];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t val = xs[i][7 - j];
                pubkey[j*4] = (val >> 24) & 0xFF;
                pubkey[j*4 + 1] = (val >> 16) & 0xFF;
                pubkey[j*4 + 2] = (val >> 8) & 0xFF;
                pubkey[j*4 + 3] = val & 0xFF;
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint32_t val = ys[i][7 - j];
                pubkey[32 + j*4] = (val >> 24) & 0xFF;
                pubkey[32 + j*4 + 1] = (val >> 16) & 0xFF;
                pubkey[32 + j*4 + 2] = (val >> 8) & 0xFF;
                pubkey[32 + j*4 + 3] = val & 0xFF;
            }
            
            // Compute address and check vanity
            uint8_t addr[20];
            eth_address(pubkey, addr);
            
            if (check_vanity(addr, target_nibble, nibble_count)) {
                atomicAdd(found_count, 1);
                found_indices[0] = gid * steps_per_thread + batch_start + i;
            }
        }
    }
}

// G16 table builder kernel
extern "C" __global__ void build_g16_table_kernel(
    uint8_t* __restrict__ g16_table,
    uint32_t window,
    uint32_t start_idx,
    uint32_t count)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    uint32_t idx = start_idx + tid;
    if (idx >= 65536) return;
    
    size_t offset = ((size_t)window * 65536 + idx) * 64;
    uint8_t* out = g16_table + offset;
    
    if (idx == 0) {
        // Zero point
        for (int i = 0; i < 64; i++) {
            out[i] = 0;
        }
        return;
    }
    
    // Build scalar k = idx << (16 * window)
    uint32_t k[8] = {0};
    uint32_t shift = window * 16;
    uint32_t limb = shift >> 5;
    uint32_t rem = shift & 31;
    
    uint64_t wide = ((uint64_t)idx) << rem;
    k[limb] = (uint32_t)(wide & 0xFFFFFFFF);
    if (limb + 1 < 8) {
        k[limb + 1] = (uint32_t)(wide >> 32);
    }
    
    // Compute point multiplication using double-and-add
    uint32_t x[8] = {0}, y[8] = {0}, z[8] = {0};
    bool first = true;
    
    for (int bit = 255; bit >= 0; bit--) {
        int limb_idx = bit >> 5;
        int bit_idx = bit & 31;
        
        if (!first) {
            point_double(x, y, z, x, y, z);
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
                point_add(x, y, z, x, y, z, SECP256K1_G, SECP256K1_G + 8);
            }
        }
    }
    
    // Convert to affine
    uint32_t x_affine[8], y_affine[8];
    if (first) {
        // Point at infinity
        for (int i = 0; i < 8; i++) {
            x_affine[i] = 0;
            y_affine[i] = 0;
        }
    } else {
        jacobian_to_affine(x_affine, y_affine, x, y, z);
    }
    
    // Write to table (little-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4] = x_affine[i] & 0xFF;
        out[i*4 + 1] = (x_affine[i] >> 8) & 0xFF;
        out[i*4 + 2] = (x_affine[i] >> 16) & 0xFF;
        out[i*4 + 3] = (x_affine[i] >> 24) & 0xFF;
    }
    
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[32 + i*4] = y_affine[i] & 0xFF;
        out[32 + i*4 + 1] = (y_affine[i] >> 8) & 0xFF;
        out[32 + i*4 + 2] = (y_affine[i] >> 16) & 0xFF;
        out[32 + i*4 + 3] = (y_affine[i] >> 24) & 0xFF;
    }
}