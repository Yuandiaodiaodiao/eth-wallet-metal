// CUDA runtime

// Secp256k1 elliptic curve implementation
#include "include/secp256k1.cuh"
#include "include/keccak256.cuh"

// Batch size for walker kernel (matches Metal implementation)
#define BATCH_WINDOW_SIZE 512

// ======================
// CUDA Kernel Functions
// ======================

// Simple vanity address generation with G16 table
extern "C" __global__ void vanity_kernel_g16(
    const uint8_t* __restrict__ privkeys,      // Private keys (32 bytes each, big-endian)
    uint32_t* __restrict__ found_indices,      // Output: indices of found addresses
    uint32_t* __restrict__ found_count,        // Output: number of found addresses
    uint8_t* __restrict__ outbuffer,           // Output: addresses for each privkey (20 bytes each)
    const uint8_t* __restrict__ g16_table,     // G16 precomputed table
    uint32_t num_keys,                         // Number of keys to process
    const uint8_t* __restrict__ head_pattern,  // Head pattern (packed nibbles as bytes)
    uint32_t head_nibbles,                     // Number of head nibbles to match
    const uint8_t* __restrict__ tail_pattern,  // Tail pattern (packed nibbles as bytes)
    uint32_t tail_nibbles)                     // Number of tail nibbles to match
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;
    
    // Load private key (big-endian to little-endian conversion)
    const uint8_t* privkey = privkeys + gid * 32;
    
    // load priv big-endian -> limbs
    uint32_t k_be[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int off = i * 4;
        uint32_t w = ((uint32_t)privkey[off] << 24) |
                     ((uint32_t)privkey[off + 1] << 16) |
                     ((uint32_t)privkey[off + 2] << 8) |
                     ((uint32_t)privkey[off + 3]);
        k_be[i] = w;
    }

    // uint8_t* out_addr = outbuffer + gid * 20;
    
    uint32_t k_local[8];
    k_local[7] = k_be[0]; k_local[6] = k_be[1]; k_local[5] = k_be[2]; k_local[4] = k_be[3];
    k_local[3] = k_be[4]; k_local[2] = k_be[5]; k_local[1] = k_be[6]; k_local[0] = k_be[7];
     
   
    // Compute public key: (x, y) = k * G
    uint32_t x[8], y[8];
    point_mul_xy(x, y, k_local, &G_PRECOMP);
   

    // pack pub (x||y) big-endian into thread buffer
    uint8_t pubkey[64];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t w = x[7 - i];
        int off = i * 4;
        pubkey[off] = w >> 24;
        pubkey[off + 1] = w >> 16;
        pubkey[off + 2] = w >> 8;
        pubkey[off + 3] = w;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t w = y[7 - i];
        int off = 32 + i * 4;
        pubkey[off] = w >> 24;
        pubkey[off + 1] = w >> 16;
        pubkey[off + 2] = w >> 8;
        pubkey[off + 3] = w;
    }


   
    // Compute Ethereum address
    uint8_t addr[20];
    eth_address(pubkey, addr);
    
    // for (int i = 0; i < 20; i++) {
    //     out_addr[i] = addr[i];
    // }
    
    // Check vanity pattern
    if (check_vanity_pattern(addr, head_pattern, head_nibbles, tail_pattern, tail_nibbles)) {
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
        int off = i * 4;
        k[i] = ((uint32_t)privkey[off] << 24) |
                   ((uint32_t)privkey[off + 1] << 16) |
                   ((uint32_t)privkey[off + 2] << 8) |
                   ((uint32_t)privkey[off + 3]);

    }
    uint32_t k_local[8];
    k_local[7] = k[0]; k_local[6] = k[1]; k_local[5] = k[2]; k_local[4] = k[3];
    k_local[3] = k[4]; k_local[2] = k[5]; k_local[1] = k[6]; k_local[0] = k[7];
     
    // Compute base point
    uint32_t x[8], y[8];
    // point_mul_g16(x, y, k, g16_table);
    point_mul_xy(x, y, k_local, &G_PRECOMP);
    // Store result
    uint32_t* out = basepoints + gid * 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = x[i];
        out[8 + i] = y[i];
    }
}

// Walker kernel: generate multiple addresses per thread (following Metal implementation)
extern "C" __global__ void vanity_walker_kernel(
    const uint32_t* __restrict__ basepoints,   // Precomputed base points
    uint32_t* __restrict__ found_indices,      // Output indices
    uint32_t* __restrict__ found_count,        // Output count
    uint32_t num_keys,
    uint32_t steps_per_thread,
    const uint8_t* __restrict__ head_pattern,  // Head pattern (packed nibbles as bytes)
    uint32_t head_nibbles,                     // Number of head nibbles to match
    const uint8_t* __restrict__ tail_pattern,  // Tail pattern (packed nibbles as bytes)
    uint32_t tail_nibbles)                     // Number of tail nibbles to match
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;
    
    // Load precomputed base point (x0, y0) and initialize z0
    const uint32_t* base_point = basepoints + gid * 16;
    uint32_t x0[8], y0[8], z0[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        x0[i] = base_point[i];
        y0[i] = base_point[8 + i];
        z0[i] = 0;  // Initialize z0
    }
    z0[0] = 1;
    
    // Prepare delta = G (affine)
    uint32_t dx[8], dy[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        dx[i] = SECP256K1_G[i];
        dy[i] = SECP256K1_G[8 + i];
    }
    
    
    // Stack arrays for batch processing - fixed size to avoid overflow
    uint32_t xs[BATCH_WINDOW_SIZE][8];
    uint32_t ys[BATCH_WINDOW_SIZE][8]; 
    uint32_t zs[BATCH_WINDOW_SIZE][8];
    uint32_t pref[BATCH_WINDOW_SIZE][8];
    
    // Process in batches of BATCH_WINDOW_SIZE to avoid stack overflow
    uint32_t processed = 0;
    while (processed < steps_per_thread) {
        // Generate batch_size points by repeated addition
        for (int i = 0; i < BATCH_WINDOW_SIZE; ++i) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                xs[i][k] = x0[k];
                ys[i][k] = y0[k];
                zs[i][k] = z0[k];
            }
            point_add(x0, y0, z0, dx, dy);
        }
        
        // Batch inversion using Montgomery trick
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            pref[0][k] = zs[0][k];
        }
        for (int i = 1; i < BATCH_WINDOW_SIZE; ++i) {
            mul_mod(pref[i], pref[i-1], zs[i]);
        }
        
        // Inverse total
        uint32_t inv_total[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            inv_total[k] = pref[BATCH_WINDOW_SIZE-1][k];
        }
        inv_mod(inv_total);
        
        // Backward pass and vanity check
        for (int ii = (int)BATCH_WINDOW_SIZE-1; ii >= 0; --ii) {
            uint32_t inv_z[8];
            if (ii == 0) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    inv_z[k] = inv_total[k];
                }
            } else {
                mul_mod(inv_z, inv_total, pref[ii-1]);
            }
            mul_mod(inv_total, inv_total, zs[ii]);
            
            // Convert to affine coordinates
            uint32_t z2[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                z2[k] = inv_z[k];
            }
            mul_mod(z2, z2, z2);
            
            uint32_t xa[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                xa[k] = xs[ii][k];
            }
            mul_mod(xa, xa, z2);
            
            uint32_t z3[8];
            mul_mod(z3, z2, inv_z);
            uint32_t ya[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                ya[k] = ys[ii][k];
            }
            mul_mod(ya, ya, z3);
            
            // Pack public key and compute Keccak
            uint8_t pub[64];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t w = xa[7-k];
                int off = k * 4;
                pub[off] = w >> 24;
                pub[off+1] = w >> 16;
                pub[off+2] = w >> 8;
                pub[off+3] = w;
            }
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                uint32_t w = ya[7-k];
                int off = 32 + k * 4;
                pub[off] = w >> 24;
                pub[off+1] = w >> 16;
                pub[off+2] = w >> 8;
                pub[off+3] = w;
            }
            
            uint8_t addr[20];
            eth_address(pub, addr);
            
        

            if (check_vanity_pattern(addr, head_pattern, head_nibbles, tail_pattern, tail_nibbles)) {
                atomicAdd(found_count, 1);
                found_indices[0] = gid * steps_per_thread + processed + (uint32_t)ii;
            }
        }
        
        processed += BATCH_WINDOW_SIZE;
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
    
    uint64_t offset = ((uint64_t)window * 65536 + idx) * 64;
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
            point_double(x, y, z);
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
                point_add(x, y, z, SECP256K1_G, SECP256K1_G + 8);
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