import os
import time
import threading
from typing import List, Tuple, Optional

from Metal import MTLCreateSystemDefaultDevice, MTLSizeMake
from g16_table import build_g16_table

# Make build_g16_table available as a method
def _build_g16_table_method(self, repo_root: str) -> None:
    """Wrapper to make build_g16_table available as a method"""
    build_g16_table(self, repo_root)


def _load_text_no_includes(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return "\n".join(l for l in lines if not l.strip().startswith("#include"))


class MetalVanity:
    # Add build_g16_table method
    build_g16_table = _build_g16_table_method
    def __init__(self, repo_root: str) -> None:
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device available")

        secp_dir = os.path.join(repo_root, "gen_eth", "secp256k1")

        src_parts: List[str] = []
        vendor_text = _load_text_no_includes(os.path.join(secp_dir, "inc_vendor.h"))
        src_parts.append(vendor_text)
        src_parts.append("\n#undef DECLSPEC\n#define DECLSPEC inline static\n")
        # Use constant address space for shared precomputed basepoint to avoid per-thread copies
        src_parts.append("#undef SECP256K1_TMPS_TYPE\n#define SECP256K1_TMPS_TYPE CONSTANT_AS\n")
        src_parts.append(_load_text_no_includes(os.path.join(secp_dir, "inc_types.h")))
        src_parts.append("\n")
        src_parts.append(_load_text_no_includes(os.path.join(secp_dir, "inc_ecc_secp256k1.h")))
        src_parts.append("\n")
        src_parts.append(_load_text_no_includes(os.path.join(secp_dir, "inc_ecc_secp256k1.cl")))

        # Append Keccak functions (fixed-size 64-byte input) and constant precomputed secp256k1 basepoint
        src_parts.append(
            r'''
// --- Steps specialization ---
#ifndef T_STEPS
#define T_STEPS 16
#endif

// Fixed window size for sliding batch processing to avoid stack overflow
#define BATCH_WINDOW_SIZE 256

// Precomputed basepoint table in constant memory to be shared across all threads
constant secp256k1_t G_PRECOMP = { {
    SECP256K1_G_PRE_COMPUTED_00, SECP256K1_G_PRE_COMPUTED_01, SECP256K1_G_PRE_COMPUTED_02, SECP256K1_G_PRE_COMPUTED_03,
    SECP256K1_G_PRE_COMPUTED_04, SECP256K1_G_PRE_COMPUTED_05, SECP256K1_G_PRE_COMPUTED_06, SECP256K1_G_PRE_COMPUTED_07,
    SECP256K1_G_PRE_COMPUTED_08, SECP256K1_G_PRE_COMPUTED_09, SECP256K1_G_PRE_COMPUTED_10, SECP256K1_G_PRE_COMPUTED_11,
    SECP256K1_G_PRE_COMPUTED_12, SECP256K1_G_PRE_COMPUTED_13, SECP256K1_G_PRE_COMPUTED_14, SECP256K1_G_PRE_COMPUTED_15,
    SECP256K1_G_PRE_COMPUTED_16, SECP256K1_G_PRE_COMPUTED_17, SECP256K1_G_PRE_COMPUTED_18, SECP256K1_G_PRE_COMPUTED_19,
    SECP256K1_G_PRE_COMPUTED_20, SECP256K1_G_PRE_COMPUTED_21, SECP256K1_G_PRE_COMPUTED_22, SECP256K1_G_PRE_COMPUTED_23,
    SECP256K1_G_PRE_COMPUTED_24, SECP256K1_G_PRE_COMPUTED_25, SECP256K1_G_PRE_COMPUTED_26, SECP256K1_G_PRE_COMPUTED_27,
    SECP256K1_G_PRE_COMPUTED_28, SECP256K1_G_PRE_COMPUTED_29, SECP256K1_G_PRE_COMPUTED_30, SECP256K1_G_PRE_COMPUTED_31,
    SECP256K1_G_PRE_COMPUTED_32, SECP256K1_G_PRE_COMPUTED_33, SECP256K1_G_PRE_COMPUTED_34, SECP256K1_G_PRE_COMPUTED_35,
    SECP256K1_G_PRE_COMPUTED_36, SECP256K1_G_PRE_COMPUTED_37, SECP256K1_G_PRE_COMPUTED_38, SECP256K1_G_PRE_COMPUTED_39,
    SECP256K1_G_PRE_COMPUTED_40, SECP256K1_G_PRE_COMPUTED_41, SECP256K1_G_PRE_COMPUTED_42, SECP256K1_G_PRE_COMPUTED_43,
    SECP256K1_G_PRE_COMPUTED_44, SECP256K1_G_PRE_COMPUTED_45, SECP256K1_G_PRE_COMPUTED_46, SECP256K1_G_PRE_COMPUTED_47,
    SECP256K1_G_PRE_COMPUTED_48, SECP256K1_G_PRE_COMPUTED_49, SECP256K1_G_PRE_COMPUTED_50, SECP256K1_G_PRE_COMPUTED_51,
    SECP256K1_G_PRE_COMPUTED_52, SECP256K1_G_PRE_COMPUTED_53, SECP256K1_G_PRE_COMPUTED_54, SECP256K1_G_PRE_COMPUTED_55,
    SECP256K1_G_PRE_COMPUTED_56, SECP256K1_G_PRE_COMPUTED_57, SECP256K1_G_PRE_COMPUTED_58, SECP256K1_G_PRE_COMPUTED_59,
    SECP256K1_G_PRE_COMPUTED_60, SECP256K1_G_PRE_COMPUTED_61, SECP256K1_G_PRE_COMPUTED_62, SECP256K1_G_PRE_COMPUTED_63,
    SECP256K1_G_PRE_COMPUTED_64, SECP256K1_G_PRE_COMPUTED_65, SECP256K1_G_PRE_COMPUTED_66, SECP256K1_G_PRE_COMPUTED_67,
    SECP256K1_G_PRE_COMPUTED_68, SECP256K1_G_PRE_COMPUTED_69, SECP256K1_G_PRE_COMPUTED_70, SECP256K1_G_PRE_COMPUTED_71,
    SECP256K1_G_PRE_COMPUTED_72, SECP256K1_G_PRE_COMPUTED_73, SECP256K1_G_PRE_COMPUTED_74, SECP256K1_G_PRE_COMPUTED_75,
    SECP256K1_G_PRE_COMPUTED_76, SECP256K1_G_PRE_COMPUTED_77, SECP256K1_G_PRE_COMPUTED_78, SECP256K1_G_PRE_COMPUTED_79,
    SECP256K1_G_PRE_COMPUTED_80, SECP256K1_G_PRE_COMPUTED_81, SECP256K1_G_PRE_COMPUTED_82, SECP256K1_G_PRE_COMPUTED_83,
    SECP256K1_G_PRE_COMPUTED_84, SECP256K1_G_PRE_COMPUTED_85, SECP256K1_G_PRE_COMPUTED_86, SECP256K1_G_PRE_COMPUTED_87,
    SECP256K1_G_PRE_COMPUTED_88, SECP256K1_G_PRE_COMPUTED_89, SECP256K1_G_PRE_COMPUTED_90, SECP256K1_G_PRE_COMPUTED_91,
    SECP256K1_G_PRE_COMPUTED_92, SECP256K1_G_PRE_COMPUTED_93, SECP256K1_G_PRE_COMPUTED_94, SECP256K1_G_PRE_COMPUTED_95
} };

inline ulong rotl64(ulong x, ushort n) { return (x << n) | (x >> (64 - n)); }
constant ulong K_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL, 0x8000000080008000UL,
    0x000000000000808bUL, 0x0000000080000001UL, 0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008aUL, 0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL, 0x000000000000800aUL, 0x800000008000000aUL,
    0x8000000080008081UL, 0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};
constant ushort K_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};
inline void keccak_f1600_inline(thread ulong a[25]) {
    ulong b[25]; ulong c[5]; ulong d[5];
    for (ushort round = 0; round < 24; ++round) {
        for (ushort x = 0; x < 5; ++x) c[x] = a[x+0]^a[x+5]^a[x+10]^a[x+15]^a[x+20];
        for (ushort x = 0; x < 5; ++x) d[x] = rotl64(c[(x+1)%5],1)^c[(x+4)%5];
        for (ushort y = 0; y < 5; ++y) { ushort yy=y*5; a[yy+0]^=d[0]; a[yy+1]^=d[1]; a[yy+2]^=d[2]; a[yy+3]^=d[3]; a[yy+4]^=d[4]; }
        for (ushort y = 0; y < 5; ++y) for (ushort x = 0; x < 5; ++x) {
            ushort idx=x+5*y; ushort xp=y; ushort yp=(ushort)((2*x+3*y)%5);
            b[xp+5*yp] = rotl64(a[idx], K_RHO[idx]);
        }
        for (ushort y = 0; y < 5; ++y) {
            ushort yy=y*5; ulong b0=b[yy+0],b1=b[yy+1],b2=b[yy+2],b3=b[yy+3],b4=b[yy+4];
            a[yy+0]= b0 ^ ((~b1) & b2);
            a[yy+1]= b1 ^ ((~b2) & b3);
            a[yy+2]= b2 ^ ((~b3) & b4);
            a[yy+3]= b3 ^ ((~b4) & b0);
            a[yy+4]= b4 ^ ((~b0) & b1);
        }
        a[0] ^= K_RC[round];
    }
}
inline ulong load64_thread(const thread uchar *src){ ulong v=0; for(ushort i=0;i<8;++i) v|=((ulong)src[i])<<(8*i); return v; }
inline void store64_le(ulong v, thread uchar *dst){ for(ushort i=0;i<8;++i) dst[i]=(uchar)((v>>(8*i))&0xFF); }


inline void keccak256_64(const thread uchar *msg64, thread uchar *out32){
    ulong A[25];
    for (ushort i=0;i<25;++i) A[i]=0UL;
    // absorb 64 bytes directly into lanes 0..7 (little-endian words)
    for (ushort i=0;i<8;++i) { A[i] ^= load64_thread(msg64 + (i*8)); }
    // padding for 64-byte message within 136-byte rate
    A[8] ^= 0x01UL;                 // domain separation bit at byte 64
    A[16] ^= 0x8000000000000000UL;  // final bit at byte 135
    keccak_f1600_inline(A);
    // squeeze first 32 bytes (lanes 0..3) little-endian
    for (ushort i=0;i<4;++i){ store64_le(A[i], out32 + i*8); }
}

// params buffer layout
struct VanityParams { uint count; uint nibble; uint nibbleCount; };


    // --------------------------
    // 16-bit window precomp path
    // --------------------------

    inline void load_point_from_g16(device const uchar *tbl, uint window_idx, uint idx16, thread u32 *x, thread u32 *y) {
        if (idx16 == 0u) { for (ushort i=0;i<8;++i){ x[i]=0; y[i]=0; } return; }
        ulong base = ((ulong)window_idx * 65536ul + (ulong)idx16) * 64ul;
        device const uchar *p = tbl + base;
        for (ushort i=0;i<8;++i){ ushort off=i*4; x[i] = ((u32)p[off]) | ((u32)p[off+1]<<8) | ((u32)p[off+2]<<16) | ((u32)p[off+3]<<24); }
        for (ushort i=0;i<8;++i){ ushort off=32+i*4; y[i] = ((u32)p[off]) | ((u32)p[off+1]<<8) | ((u32)p[off+2]<<16) | ((u32)p[off+3]<<24); }
    }

    inline uint get_window16(const thread u32 *k_local, uint win){
        uint bit = win * 16u;
        uint limb = bit >> 5;           // /32
        uint shift = bit & 31u;         // %32
        uint val;
        if (shift <= 16u) {
            val = (k_local[limb] >> shift) & 0xFFFFu;
        } else {
            uint low = k_local[limb] >> shift;
            uint high = k_local[limb+1u] << (32u - shift);
            val = (low | high) & 0xFFFFu;
        }
        return val;
    }

    inline void point_mul_xy_w16(thread u32 *x_out, thread u32 *y_out, const thread u32 *k_local, device const uchar *g16_tbl){
        bool have = false;
        u32 x1[8]; u32 y1[8]; u32 z1[8];
        for (ushort i=0;i<8;++i) { z1[i]=0; }
        for (uint win = 0; win < 16u; ++win){
            uint idx16 = get_window16(k_local, win);
            if (idx16 == 0u) continue;
            u32 x2[8]; u32 y2[8];
            load_point_from_g16(g16_tbl, win, idx16, x2, y2);
            if (!have){
                for (ushort i=0;i<8;++i){ x1[i]=x2[i]; y1[i]=y2[i]; z1[i]=0; }
                z1[0]=1;
                have = true;
            } else {
                point_add(x1, y1, z1, x2, y2);
            }
        }
        inv_mod(z1);
        u32 z2[8]; mul_mod(z2, z1, z1);
        mul_mod(x1, x1, z2);
        mul_mod(z1, z2, z1);
        mul_mod(y1, y1, z1);
        for (ushort i=0;i<8;++i){ x_out[i]=x1[i]; y_out[i]=y1[i]; }
    }

    

    KERNEL_FQ void vanity_kernel_w16_compact(
        device const uchar *priv_in                  [[ buffer(0) ]],
        device uint *index_compact_out               [[ buffer(1) ]],
        device atomic_uint *out_count                [[ buffer(2) ]],
        device const VanityParams *params            [[ buffer(3) ]],
        device const uchar *g16_table                [[ buffer(4) ]],
        uint gid                                     [[ thread_position_in_grid ]])
    {
        uint count = params->count; if (gid >= count) return;
        const device uchar *p = priv_in + gid * 32;
        u32 k_be[8]; for (ushort i=0;i<8;++i){ ushort off=i*4; u32 w=((u32)p[off]<<24)|((u32)p[off+1]<<16)|((u32)p[off+2]<<8)|((u32)p[off+3]); k_be[i]=w; }
        u32 k_local[8]; k_local[7]=k_be[0]; k_local[6]=k_be[1]; k_local[5]=k_be[2]; k_local[4]=k_be[3]; k_local[3]=k_be[4]; k_local[2]=k_be[5]; k_local[1]=k_be[6]; k_local[0]=k_be[7];
        u32 x[8]; u32 y[8]; point_mul_xy_w16(x, y, k_local, g16_table);
        uchar pub[64];
        for (ushort i=0;i<8;++i){ u32 w=x[7-i]; ushort off=i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
        for (ushort i=0;i<8;++i){ u32 w=y[7-i]; ushort off=32+i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
        uchar digest[32]; keccak256_64(pub, digest);
        uint want = params->nibble & 0xF; uint nibs = params->nibbleCount; bool ok = true;
        uchar want_byte = (uchar)((want << 4) | want); uint full = nibs >> 1; uint rem = nibs & 1u;
        for (uint i = 0; i < full; ++i) { if (digest[12 + i] != want_byte) { ok = false; break; } }
        if (ok && rem) { uchar b = digest[12 + full]; if ((b >> 4) != want) ok = false; }
        if (ok) { uint idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed); index_compact_out[0] = gid; }
    }

    // --- Builder for g16 table ---
    struct G16Params { uint win; uint start; };

    KERNEL_FQ void g16_builder_kernel(
        device uchar *g16_table                  [[ buffer(0) ]],
        device const G16Params *pp               [[ buffer(1) ]],
        uint gid                                 [[ thread_position_in_grid ]])
    {
        uint idx = pp->start + gid;
        if (idx >= 65536u) return;
        uint win = pp->win;
        ulong base = ((ulong)win * 65536ul + (ulong)idx) * 64ul;
        device uchar *outp = g16_table + base;
        if (idx == 0u){
            for (ushort i=0;i<64;++i) outp[i]=0;
            return;
        }
        // Build scalar k_local = idx << (16*win)
        u32 k_local[8]; for (ushort i=0;i<8;++i) k_local[i]=0u;
        uint shift = win * 16u; uint limb = shift >> 5; uint rem = shift & 31u;
        u64 wide = ((u64)idx) << rem;
        k_local[limb] = (u32)(wide & 0xffffffffu);
        if (limb + 1u < 8u) k_local[limb+1u] = (u32)(wide >> 32);
        u32 x[8]; u32 y[8]; point_mul_xy(x, y, k_local, &G_PRECOMP);
        // write x,y as little-endian bytes
        for (ushort i=0;i<8;++i){ u32 w=x[i]; ushort off=i*4; outp[off+0]=(uchar)(w & 0xFF); outp[off+1]=(uchar)((w>>8)&0xFF); outp[off+2]=(uchar)((w>>16)&0xFF); outp[off+3]=(uchar)(w>>24); }
        for (ushort i=0;i<8;++i){ u32 w=y[i]; ushort off=32+i*4; outp[off+0]=(uchar)(w & 0xFF); outp[off+1]=(uchar)((w>>8)&0xFF); outp[off+2]=(uchar)((w>>16)&0xFF); outp[off+3]=(uchar)(w>>24); }
    }

    // --------------------------
    // Walker kernels (batched inversion) - generate multiple pubkeys per thread by repeated addition
    // --------------------------

    struct WalkParams { uint count; uint nibble; uint nibbleCount; uint steps; };

    inline void load_G_affine(thread u32 *gx, thread u32 *gy){
        // G_PRECOMP.xy[0..7] = x(G); [8..15] = y(G)
        for (ushort i=0;i<8;++i){ gx[i] = G_PRECOMP.xy[i]; }
        for (ushort i=0;i<8;++i){ gy[i] = G_PRECOMP.xy[8 + i]; }
    }



    // Optimized version using g16 table for faster point multiplication
    KERNEL_FQ void vanity_kernel_compute_basepoint_w16(
        device const uchar *priv_in                  [[ buffer(0) ]],
        device u32 *base_points_out                  [[ buffer(1) ]],
        device const WalkParams *params              [[ buffer(2) ]],
        device const uchar *g16_table                [[ buffer(3) ]],
        uint gid                                     [[ thread_position_in_grid ]])
    {
        uint count = params->count; if (gid >= count) return;
        const device uchar *p = priv_in + gid * 32;
        u32 k_be[8];
        for (ushort i=0;i<8;++i){ ushort off=i*4; u32 w=((u32)p[off]<<24)|((u32)p[off+1]<<16)|((u32)p[off+2]<<8)|((u32)p[off+3]); k_be[i]=w; }
        u32 k_local[8];
        k_local[7]=k_be[0]; k_local[6]=k_be[1]; k_local[5]=k_be[2]; k_local[4]=k_be[3];
        k_local[3]=k_be[4]; k_local[2]=k_be[5]; k_local[1]=k_be[6]; k_local[0]=k_be[7];

        // Base point P0 = k*G (affine) - use optimized 16-bit window method
        u32 x0[8]; u32 y0[8]; point_mul_xy_w16(x0, y0, k_local, g16_table);
        
        // Store base point (x0, y0) to output buffer
        device u32 *out = base_points_out + gid * 16; // 16 u32s per point (8 for x, 8 for y)
        for (ushort i=0;i<8;++i){ out[i] = x0[i]; out[8+i] = y0[i]; }
    }

    // Optimized walker: uses precomputed base points to reduce register usage
    KERNEL_FQ void vanity_kernel_walk_compact(
        device const u32 *base_points_in             [[ buffer(0) ]],
        device uint *index_compact_out               [[ buffer(1) ]],
        device atomic_uint *out_count                [[ buffer(2) ]],
        device const WalkParams *params              [[ buffer(3) ]],
        uint gid                                     [[ thread_position_in_grid ]])
    {
        uint count = params->count; if (gid >= count) return;
        
        // Load precomputed base point (x0, y0) and initialize z0 in single loop
        device const u32 *base_point = base_points_in + gid * 16;
        u32 x0[8]; u32 y0[8]; u32 z0[8];
        for (ushort i=0;i<8;++i){ 
            x0[i] = base_point[i]; 
            y0[i] = base_point[8+i];
            z0[i] =  0;  // Initialize z0 inline
        }
        z0[0]=1;
        // Prepare delta = G (affine)
        u32 dx[8]; u32 dy[8]; 
        load_G_affine(dx, dy);

        const uint steps = params->steps;
        uint want = params->nibble & 0xF; 
        uint nibs = params->nibbleCount;
        uchar want_byte = (uchar)((want << 4) | want);
 // Stack arrays for this batch - fixed size to avoid overflow
        u32 xs[BATCH_WINDOW_SIZE][8];
        u32 ys[BATCH_WINDOW_SIZE][8]; 
        u32 zs[BATCH_WINDOW_SIZE][8];
        u32 pref[BATCH_WINDOW_SIZE][8];
        // Process in batches of BATCH_WINDOW_SIZE to avoid stack overflow
        uint processed = 0;
        while (processed < steps) {
            
            // Inline batch processing to avoid function call overhead
            {
               
                
                // Start from base point for this batch
                
                // Generate batch_size points by repeated addition
                for (uint i=0; i<BATCH_WINDOW_SIZE; ++i){
                    for (ushort k=0;k<8;++k){ xs[i][k]=x0[k]; ys[i][k]=y0[k]; zs[i][k]=z0[k]; }
                    point_add(x0, y0, z0, dx, dy);
                }
                
                
                // Batch inversion using Montgomery trick
                for (ushort k=0;k<8;++k){ pref[0][k] = zs[0][k]; }
                for (uint i=1; i<BATCH_WINDOW_SIZE; ++i){
                    mul_mod(pref[i], pref[i-1], zs[i]);
                }
                
                // Inverse total
                u32 inv_total[8];
                for (ushort k=0;k<8;++k){ inv_total[k] = pref[BATCH_WINDOW_SIZE-1][k]; }
                inv_mod(inv_total);
                
                // Backward pass and vanity check
                for (int ii=(int)BATCH_WINDOW_SIZE-1; ii>=0; --ii){
                    u32 inv_z[8];
                    if (ii == 0){
                        for (ushort k=0;k<8;++k){ inv_z[k]=inv_total[k]; }
                    } else {
                        mul_mod(inv_z, inv_total, pref[ii-1]);
                    }
                    mul_mod(inv_total, inv_total, zs[ii]);
                    
                    // Convert to affine coordinates
                    u32 z2[8]; for (ushort k=0;k<8;++k) z2[k]=inv_z[k];
                    mul_mod(z2, z2, z2);
                    u32 xa[8]; for (ushort k=0;k<8;++k) xa[k]=xs[ii][k]; mul_mod(xa, xa, z2);
                    u32 z3[8]; mul_mod(z3, z2, inv_z);
                    u32 ya[8]; for (ushort k=0;k<8;++k) ya[k]=ys[ii][k]; mul_mod(ya, ya, z3);
                    
                    // Pack public key and compute Keccak
                    uchar pub[64];
                    for (ushort k=0;k<8;++k){ u32 w=xa[7-k]; ushort off=k*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
                    for (ushort k=0;k<8;++k){ u32 w=ya[7-k]; ushort off=32+k*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
                    
                    uchar digest[32]; 
                    
                    keccak256_64(pub, digest);
                    
                    // Vanity check
                    bool ok = true;
                    uint full = nibs >> 1; 
                    uint rem = nibs & 1u;
                    for (uint b = 0; b < full; ++b) {
                     if (digest[12 + b] != want_byte) { ok = false; break; } 
                    }
                    if (ok && rem) { 
                    if ((digest[12 + full] >> 4) != want) ok = false; 
                    }
                    
                    if (ok){
                        uint idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
                        index_compact_out[0] = gid * steps + processed + (uint)ii;
                    }
                }
            }
            
            processed += BATCH_WINDOW_SIZE;
        }
    }

'''
        )

        source = "\n".join(src_parts)
        # Save template for per-steps specialization
        self._source_template = source

        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if library is None:
            raise RuntimeError(f"Metal library compile failed: {error}")

     
        # Try to compile w16 kernels
        self.pipeline_w16_compact = None
        fn_w16c = library.newFunctionWithName_("vanity_kernel_w16_compact")
        fn_builder = library.newFunctionWithName_("g16_builder_kernel")
        fn_compute_base_w16 = library.newFunctionWithName_("vanity_kernel_compute_basepoint_w16")
        fn_walk = library.newFunctionWithName_("vanity_kernel_walk_compact")
        if fn_w16c is not None:
            p_w16c, error = self.device.newComputePipelineStateWithFunction_error_(fn_w16c, None)
            if p_w16c is not None:
                self.pipeline_w16_compact = p_w16c
        self.pipeline_builder = None
      

        # Single command queue is sufficient; command buffers pipeline naturally
        self.queue = self.device.newCommandQueue()
        self.thread_execution_width = 16
        print(f'thread_execution_width: {self.thread_execution_width}')
        # Pipeline caches for specialized walkers
        self._walk_pipelines = {}
        # Keep no reusable buffers here; API supports overlapping jobs, so use per-job buffers
        # Load optional 16-bit window precomputed table from disk if available
        self.g16_buffer = None
        try:
            g16_path = os.path.join(secp_dir, "g16_precomp_le.bin")
            expected = 16 * 65536 * 64
            if os.path.isfile(g16_path) and os.path.getsize(g16_path) == expected:
                with open(g16_path, "rb") as f:
                    data = f.read()
                buf = self.device.newBufferWithLength_options_(expected, 0)
                buf.contents().as_buffer(expected)[:expected] = data
                self.g16_buffer = buf
        except Exception:
            self.g16_buffer = None

   

    # --- Compact-output pipelined API ---
    class VanityJobCompact:
        def __init__(self, cb, index_compact_buffer, out_count_buffer, capacity_count: int):
            self.cb = cb
            self.index_compact_buffer = index_compact_buffer
            self.out_count_buffer = out_count_buffer
            self.capacity_count = capacity_count
            # Encoded steps per thread used by the kernel that produced indices
            self.effective_steps_per_thread: int = 1
            self._done_event: threading.Event = threading.Event()
            self._indices: Optional[List[int]] = None
            self._error: Optional[Exception] = None
            # Timing
            self.cpu_encode_seconds: float = 0.0
            self.cpu_completion_seconds: float = 0.0
            self.gpu_start_time: float = -1.0
            self.gpu_end_time: float = -1.0

    def encode_and_commit_compact(self, privkeys_be32: List[bytes], nibble: int = 0x8, nibble_count: int = 7) -> "MetalVanity.VanityJobCompact":
        if not privkeys_be32:
            raise ValueError("privkeys_be32 must not be empty")
        for i, k in enumerate(privkeys_be32):
            if len(k) != 32:
                raise ValueError(f"privkey[{i}] must be 32 bytes")
        count = len(privkeys_be32)
        in_size = 32 * count
        idx_cap_bytes = 4 * count

        # Per-job buffers; overlapping jobs are expected
        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        index_compact_buffer = self.device.newBufferWithLength_options_(idx_cap_bytes, 0)
        out_count_buffer = self.device.newBufferWithLength_options_(4, 0)
        # Params will be set inline via setBytes to avoid allocating a buffer each call

        # Fill inputs without creating a large temporary joined bytes
        in_mv = in_buffer.contents().as_buffer(in_size)
        for i, k in enumerate(privkeys_be32):
            off = i * 32
            in_mv[off : off + 32] = k

        # Zero out_count
        out_count_buffer.contents().as_buffer(4)[:4] = (0).to_bytes(4, "little")

        # params
        p = bytearray(12)
        p[0:4] = int(count).to_bytes(4, "little")
        p[4:8] = int(nibble & 0xF).to_bytes(4, "little")
        p[8:12] = int(nibble_count).to_bytes(4, "little")

        t_cpu0 = time.perf_counter()
        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        use_w16 = True
        enc.setComputePipelineState_(self.pipeline_w16_compact)
        enc.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(index_compact_buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(out_count_buffer, 0, 2)
        # Prefer setBytes for small constant params; fall back to a temp buffer if unavailable
        try:
            enc.setBytes_length_atIndex_(bytes(p), 12, 3)
        except Exception:
            params_buffer = self.device.newBufferWithLength_options_(12, 0)
            params_buffer.contents().as_buffer(12)[:12] = bytes(p)
            enc.setBuffer_offset_atIndex_(params_buffer, 0, 3)
        if use_w16:
            enc.setBuffer_offset_atIndex_(self.g16_buffer, 0, 4)

        w = int(self.thread_execution_width)
        try:
            max_threads = int((self.pipeline_w16_compact).maxTotalThreadsPerThreadgroup())
        except Exception:
            max_threads = 256
        tpt = min(max_threads, max(w * 4, w), max(1, count))
        tg = MTLSizeMake(tpt, 1, 1)
        # Dispatch exactly 'count' threads; avoid padded idle threads
        grid = MTLSizeMake(count, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()

        job = MetalVanity.VanityJobCompact(cb, index_compact_buffer, out_count_buffer, count)
        job.effective_steps_per_thread = 1

        def _on_completed(inner_cb):
            cpu_start = time.perf_counter()
            try:
                # Read number of matches
                c_bytes = bytes(job.out_count_buffer.contents().as_buffer(4)[:4])
                out_count = int.from_bytes(c_bytes, "little")
                out_count = max(0, min(out_count, job.capacity_count))
                # Read compact indices
                idx_bytes = bytes(job.index_compact_buffer.contents().as_buffer(4)[:4])
                indices = [int.from_bytes(idx_bytes, "little")] if out_count > 0 else []
                job._indices = indices
                # GPU timing if supported
                try:
                    job.gpu_start_time = float(inner_cb.GPUStartTime())  # type: ignore[attr-defined]
                    job.gpu_end_time = float(inner_cb.GPUEndTime())      # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as e:
                job._error = e
            finally:
                job.cpu_completion_seconds = time.perf_counter() - cpu_start
                job._done_event.set()

        cb.addCompletedHandler_(_on_completed)
        cb.commit()
        job.cpu_encode_seconds = time.perf_counter() - t_cpu0
        return job

    def _ensure_walk_pipelines(self, steps_per_thread: int):
        key = int(steps_per_thread)
        # Return cached pipeline if present
        if key in self._walk_pipelines:
            return self._walk_pipelines[key]
        # Compile specialized library with T_STEPS macro
        specialized = f"#define T_STEPS {key}\n" + self._source_template
        library, error = self.device.newLibraryWithSource_options_error_(specialized, None, None)
        if library is None:
            raise RuntimeError(f"Metal library compile failed for steps={key}: {error}")
        
        # Compile both compute_base and walk functions
        fn_compute_base_w16 = library.newFunctionWithName_("vanity_kernel_compute_basepoint_w16")
        fn_walk = library.newFunctionWithName_("vanity_kernel_walk_compact")
        if fn_walk is None:
            raise RuntimeError("vanity_kernel_walk_compact not found in specialized library")
        
        # Try to compile w16 version if available
        p_cb_w16 = None
        if fn_compute_base_w16 is not None:
            p_cb_w16, error = self.device.newComputePipelineStateWithFunction_error_(fn_compute_base_w16, None)
        
        p_wc, error = self.device.newComputePipelineStateWithFunction_error_(fn_walk, None)
        self.thread_execution_width = p_wc.threadExecutionWidth()
        print(f'thread_execution_width: {self.thread_execution_width}')
        if p_wc is None:
            raise RuntimeError(f"Failed to create specialized walk pipeline for steps={key}: {error}")
        
        # Store both regular and w16 versions (w16 may be None)
        self._walk_pipelines[key] = (None, p_wc, p_cb_w16)
        return (None, p_wc, p_cb_w16)

    def wait_and_collect_compact(self, job: "MetalVanity.VanityJobCompact") -> Tuple[List[int], int]:
        job._done_event.wait()
        if job._error is not None:
            raise job._error
        return job._indices or [], job.effective_steps_per_thread


    def encode_and_commit_walk_compact(self, privkeys_be32: List[bytes], steps_per_thread: int = 8, nibble: int = 0x8, nibble_count: int = 7) -> "MetalVanity.VanityJobCompact":
        if not privkeys_be32:
            raise ValueError("privkeys_be32 must not be empty")
        if steps_per_thread <= 0:
            raise ValueError("steps_per_thread must be > 0")
        for i, k in enumerate(privkeys_be32):
            if len(k) != 32:
                raise ValueError(f"privkey[{i}] must be 32 bytes")
        count = len(privkeys_be32)
        in_size = 32 * count
        capacity_count = count * steps_per_thread

        # Buffers
        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        # Base points buffer (16 u32s per point: 8 for x, 8 for y)
        base_points_buffer = self.device.newBufferWithLength_options_(count * 16 * 4, 0)
        # 索引buff
        index_compact_buffer = self.device.newBufferWithLength_options_(4, 0)
        # 输出计数buff
        out_count_buffer = self.device.newBufferWithLength_options_(4, 0)

        # Fill inputs
        in_mv = in_buffer.contents().as_buffer(in_size)
        for i, k in enumerate(privkeys_be32):
            off = i * 32
            in_mv[off : off + 32] = k

        # Zero out_count
        out_count_buffer.contents().as_buffer(4)[:4] = (0).to_bytes(4, "little")

        # params: WalkParams {count, nibble, nibbleCount, steps}
        p = bytearray(16)
        p[0:4] = int(count).to_bytes(4, "little")
        p[4:8] = int(nibble & 0xF).to_bytes(4, "little")
        p[8:12] = int(nibble_count).to_bytes(4, "little")
        p[12:16] = int(steps_per_thread).to_bytes(4, "little")

        t_cpu0 = time.perf_counter()
        cb = self.queue.commandBuffer()
        
        # Build or reuse specialized pipelines for this steps_per_thread
        _, walk_pipeline, compute_base_w16_pipeline = self._ensure_walk_pipelines(steps_per_thread)

        # Stage 1: Compute base points from private keys
        enc1 = cb.computeCommandEncoder()
        # Use w16 version if available for better performance
        use_w16_compute = compute_base_w16_pipeline is not None and self.g16_buffer is not None
        if use_w16_compute:
            print("Using optimized vanity_kernel_compute_basepoint_w16 with g16 table")
        enc1.setComputePipelineState_(compute_base_w16_pipeline)
        enc1.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        enc1.setBuffer_offset_atIndex_(base_points_buffer, 0, 1)
        try:
            enc1.setBytes_length_atIndex_(bytes(p), 16, 2)
        except Exception:
            params_buffer = self.device.newBufferWithLength_options_(16, 0)
            params_buffer.contents().as_buffer(16)[:16] = bytes(p)
            enc1.setBuffer_offset_atIndex_(params_buffer, 0, 2)
        # Pass g16_table buffer if using w16 version
        if use_w16_compute:
            enc1.setBuffer_offset_atIndex_(self.g16_buffer, 0, 3)
        
        active_compute_pipeline = compute_base_w16_pipeline
        print(f'compute_base_pipeline thread_execution_width: {active_compute_pipeline.threadExecutionWidth()}')
        max_threads = int(active_compute_pipeline.maxTotalThreadsPerThreadgroup())
        print(f'compute_base_pipeline max_threads: {max_threads}')

        tpt = 8
        tg = MTLSizeMake(tpt, 1, 1)
        grid = MTLSizeMake(count, 1, 1)
        enc1.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc1.endEncoding()
        
        # Stage 2: Walker using precomputed base points
        enc2 = cb.computeCommandEncoder()
        enc2.setComputePipelineState_(walk_pipeline)
        enc2.setBuffer_offset_atIndex_(base_points_buffer, 0, 0)
        enc2.setBuffer_offset_atIndex_(index_compact_buffer, 0, 1)
        enc2.setBuffer_offset_atIndex_(out_count_buffer, 0, 2)
        try:
            enc2.setBytes_length_atIndex_(bytes(p), 16, 3)
        except Exception:
            if 'params_buffer' not in locals():
                params_buffer = self.device.newBufferWithLength_options_(16, 0)
                params_buffer.contents().as_buffer(16)[:16] = bytes(p)
            enc2.setBuffer_offset_atIndex_(params_buffer, 0, 3)
        print(f'walk_pipeline thread_execution_width: {walk_pipeline.threadExecutionWidth()}')
        max_threads = int(walk_pipeline.maxTotalThreadsPerThreadgroup())
        print(f'walk_pipeline max_threads: {max_threads}')
        tpt = 64
        print(f'tpt: {tpt}')
        tg = MTLSizeMake(tpt, 1, 1)
        grid = MTLSizeMake(count, 1, 1)
        enc2.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc2.endEncoding()

        job = MetalVanity.VanityJobCompact(cb, index_compact_buffer, out_count_buffer, capacity_count)
        job.effective_steps_per_thread = int(steps_per_thread)

        def _on_completed(inner_cb):
            cpu_start = time.perf_counter()
            try:
                c_bytes = bytes(job.out_count_buffer.contents().as_buffer(4)[:4])
                out_count = int.from_bytes(c_bytes, "little")
                out_count = max(0, min(out_count, job.capacity_count))
                idx_bytes = bytes(job.index_compact_buffer.contents().as_buffer(4)[:4])
                indices = [int.from_bytes(idx_bytes, "little")] if out_count > 0 else []
                job._indices = indices
                try:
                    job.gpu_start_time = float(inner_cb.GPUStartTime())
                    job.gpu_end_time = float(inner_cb.GPUEndTime())
                except Exception:
                    pass
            except Exception as e:
                job._error = e
            finally:
                job.cpu_completion_seconds = time.perf_counter() - cpu_start
                job._done_event.set()

        cb.addCompletedHandler_(_on_completed)
        cb.commit()
        job.cpu_encode_seconds = time.perf_counter() - t_cpu0
        return job
