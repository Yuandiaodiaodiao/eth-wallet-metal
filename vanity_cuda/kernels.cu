// CUDA runtime

// Secp256k1 elliptic curve implementation
#include "include/secp256k1.cuh"
#include "include/keccak256.cuh"



// Batch size for walker kernel (matches Metal implementation)
#define BATCH_WINDOW_SIZE 128

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
    uint32_t num_keys)                         // Number of keys to process
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
   

    // Compute Ethereum address directly from affine coordinates
    uint8_t addr[20];
    eth_address(x, y, addr);
    
    // for (int i = 0; i < 20; i++) {
    //     out_addr[i] = addr[i];
    // }
    
    // Check vanity pattern
    if (check_vanity_pattern_optimized(addr)) {
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
    point_mul_g16(x, y, k_local, g16_table);
    // point_mul_xy(x, y, k_local, &G_PRECOMP);
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
    uint32_t steps_per_thread
)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_keys) return;
    
    // Load precomputed base point (x0, y0) and initialize z0
    const uint32_t* base_point = basepoints + gid * 16;
    uint32_t x0[8], y0[8], z0[8];
    move_u256(x0, base_point);
    move_u256(y0, base_point + 8);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        z0[i] = 0;  // Initialize z0
    }
    z0[0] = 1;
    
    // Prepare delta = G (affine)
    uint32_t dx[8], dy[8];
    move_u256(dx, SECP256K1_G);
    move_u256(dy, SECP256K1_G + 8);
    
    
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
            move_u256(xs[i], x0);
            move_u256(ys[i], y0);
            move_u256(zs[i], z0);
            point_add(x0, y0, z0, dx, dy);
        }
        
        // Batch inversion using Montgomery trick
        move_u256(pref[0], zs[0]);
        for (int i = 1; i < BATCH_WINDOW_SIZE; ++i) {
            mul_mod(pref[i], pref[i-1], zs[i]);
        }
        
        // Inverse total
        uint32_t inv_total[8];
        move_u256(inv_total, pref[BATCH_WINDOW_SIZE-1]);
        inv_mod(inv_total);
        
        // Backward pass and vanity check
        for (int ii = (int)BATCH_WINDOW_SIZE-1; ii >= 0; --ii) {
            uint32_t inv_z[8];
            if (ii == 0) {
                move_u256(inv_z, inv_total);
            } else {
                mul_mod(inv_z, inv_total, pref[ii-1]);
            }
            mul_mod(inv_total, inv_total, zs[ii]);
            
            // Convert to affine coordinates
            uint32_t z2[8];
            move_u256(z2, inv_z);
            mul_mod(z2, z2, z2);
            
            uint32_t xa[8];
            move_u256(xa, xs[ii]);
            mul_mod(xa, xa, z2);
            
            uint32_t z3[8];
            mul_mod(z3, z2, inv_z);
            uint32_t ya[8];
            move_u256(ya, ys[ii]);
            mul_mod(ya, ya, z3);
            
            // Compute Ethereum address directly from affine coordinates
            uint8_t addr[20];
            eth_address(xa, ya, addr);
            
        

            // Check vanity pattern
            if (check_vanity_pattern_optimized(addr)) {
                uint32_t idx = atomicAdd(found_count, 1);
                if (idx == 0) {
                    found_indices[0] = gid * steps_per_thread + processed + (uint32_t)ii;
                }
            }
        }
        
        processed += BATCH_WINDOW_SIZE;
    }
}

