#include "include/constants.cuh"
#include "include/secp256k1.cuh"
#include "include/inc_ecc_secp256k1.cuh"
#include "include/keccak256.cuh"
#include "include/ec_ops.cuh"
#include "include/g16_ops.cuh"
#include <cuda_runtime.h>

// Batch size for walker kernel
#define BATCH_WINDOW_SIZE 256

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
    uint8_t target_nibble,                     // Target nibble value (0x0 - 0xF)
    uint32_t nibble_count)                     // Number of nibbles to match
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
    uint8_t* out_addr = outbuffer + gid * 20;
    
    uint32_t k_local[8];
    k_local[7] = k_be[0]; k_local[6] = k_be[1]; k_local[5] = k_be[2]; k_local[4] = k_be[3];
    k_local[3] = k_be[4]; k_local[2] = k_be[5]; k_local[1] = k_be[6]; k_local[0] = k_be[7];
    
    // Compute public key: (x, y) = k * G
    uint32_t x[8], y[8];
    point_mul_xy(x, y, k_local, &G_PRECOMP);
    
    // Store address in output buffer
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out_addr[i] = x[i];
    }

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
            point_add(x0, y0, z0, gx, gy);
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