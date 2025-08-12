import os
import time
import secrets
import threading
import math
from typing import List, Tuple, Optional

# Precompute curve order bytes once to speed privkey validity checks
try:
    from ecdsa import SECP256k1  # type: ignore
    SECP256K1_ORDER_BYTES = SECP256k1.order.to_bytes(32, "big")
except Exception:  # Fallback if ecdsa not importable at import-time
    SECP256K1_ORDER_BYTES = int(
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    ).to_bytes(32, "big")

ZERO_32 = b"\x00" * 32

from Metal import MTLCreateSystemDefaultDevice, MTLSizeMake

# Integer curve order for fast modular arithmetic
SECP256K1_ORDER_INT = int.from_bytes(SECP256K1_ORDER_BYTES, "big")

# Optional NumPy acceleration (SIMD on Apple Silicon)
try:
    import numpy as np  # type: ignore

    HAS_NUMPY = True
except Exception:  # Optional dependency
    HAS_NUMPY = False


def _load_text_no_includes(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return "\n".join(l for l in lines if not l.strip().startswith("#include"))


class MetalVanity:
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

KERNEL_FQ void vanity_kernel(
    device const uchar *priv_in          [[ buffer(0) ]],
    device uchar *addr_out               [[ buffer(1) ]],
    device uchar *flags_out              [[ buffer(2) ]],
    device const VanityParams *params    [[ buffer(3) ]],
    uint gid                              [[ thread_position_in_grid ]])
{
    uint count = params->count;
    if (gid >= count) return;
    const device uchar *p = priv_in + gid * 32;

    // load priv big-endian -> limbs
    u32 k_be[8];
    for (ushort i=0;i<8;++i){ ushort off=i*4; u32 w=((u32)p[off]<<24)|((u32)p[off+1]<<16)|((u32)p[off+2]<<8)|((u32)p[off+3]); k_be[i]=w; }
    u32 k_local[8];
    k_local[7]=k_be[0]; k_local[6]=k_be[1]; k_local[5]=k_be[2]; k_local[4]=k_be[3];
    k_local[3]=k_be[4]; k_local[2]=k_be[5]; k_local[1]=k_be[6]; k_local[0]=k_be[7];

    u32 x[8]; u32 y[8]; point_mul_xy(x, y, k_local, &G_PRECOMP);

    // pack pub (x||y) big-endian into thread buffer
    uchar pub[64];
    for (ushort i=0;i<8;++i){ u32 w=x[7-i]; ushort off=i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
    for (ushort i=0;i<8;++i){ u32 w=y[7-i]; ushort off=32+i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }

    // keccak
    uchar digest[32]; keccak256_64(pub, digest);
    // vanity check directly on digest (address = last 20 bytes)
    uint want = params->nibble & 0xF;
    uint nibs = params->nibbleCount;
    bool ok = true;
    uchar want_byte = (uchar)((want << 4) | want);
    uint full = nibs >> 1;            // number of full bytes to check
    uint rem = nibs & 1u;             // 1 if one extra high nibble
    for (uint i = 0; i < full; ++i) {
        if (digest[12 + i] != want_byte) { ok = false; break; }
    }
    if (ok && rem) {
        uchar b = digest[12 + full];
        if ((b >> 4) != want) ok = false;
    }
    flags_out[gid] = ok ? (uchar)1 : (uchar)0;
    if (ok) {
        device uchar *addr = addr_out + gid * 20;
        for (ushort i=0;i<20;++i) addr[i] = digest[12 + i];
    }
}

// Compact-output variant: write matches only using atomic compaction
KERNEL_FQ void vanity_kernel_compact(
    device const uchar *priv_in              [[ buffer(0) ]],
    device uchar *addr_compact_out           [[ buffer(1) ]],
    device uint *index_compact_out           [[ buffer(2) ]],
    device atomic_uint *out_count            [[ buffer(3) ]],
    device const VanityParams *params        [[ buffer(4) ]],
    uint gid                                 [[ thread_position_in_grid ]])
{
    uint count = params->count;
    if (gid >= count) return;
    const device uchar *p = priv_in + gid * 32;

    // load priv big-endian -> limbs
    u32 k_be[8];
    for (ushort i=0;i<8;++i){ ushort off=i*4; u32 w=((u32)p[off]<<24)|((u32)p[off+1]<<16)|((u32)p[off+2]<<8)|((u32)p[off+3]); k_be[i]=w; }
    u32 k_local[8];
    k_local[7]=k_be[0]; k_local[6]=k_be[1]; k_local[5]=k_be[2]; k_local[4]=k_be[3];
    k_local[3]=k_be[4]; k_local[2]=k_be[5]; k_local[1]=k_be[6]; k_local[0]=k_be[7];

    u32 x[8]; u32 y[8]; point_mul_xy(x, y, k_local, &G_PRECOMP);

    // pack pub (x||y) big-endian into thread buffer
    uchar pub[64];
    for (ushort i=0;i<8;++i){ u32 w=x[7-i]; ushort off=i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
    for (ushort i=0;i<8;++i){ u32 w=y[7-i]; ushort off=32+i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }

    // keccak
    uchar digest[32]; keccak256_64(pub, digest);
    // vanity check directly on digest (address = last 20 bytes)
    uint want = params->nibble & 0xF;
    uint nibs = params->nibbleCount;
    bool ok = true;
    uchar want_byte = (uchar)((want << 4) | want);
    uint full = nibs >> 1;            // number of full bytes to check
    uint rem = nibs & 1u;             // 1 if one extra high nibble
    for (uint i = 0; i < full; ++i) {
        if (digest[12 + i] != want_byte) { ok = false; break; }
    }
    if (ok && rem) {
        uchar b = digest[12 + full];
        if ((b >> 4) != want) ok = false;
    }
    if (ok) {
        uint idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
        device uchar *addr = addr_compact_out + idx * 20;
        for (ushort i=0;i<20;++i) addr[i] = digest[12 + i];
        index_compact_out[idx] = gid;
    }
}

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

    KERNEL_FQ void vanity_kernel_w16(
        device const uchar *priv_in              [[ buffer(0) ]],
        device uchar *addr_out                   [[ buffer(1) ]],
        device uchar *flags_out                  [[ buffer(2) ]],
        device const VanityParams *params        [[ buffer(3) ]],
        device const uchar *g16_table            [[ buffer(4) ]],
        uint gid                                 [[ thread_position_in_grid ]])
    {
        uint count = params->count;
        if (gid >= count) return;
        const device uchar *p = priv_in + gid * 32;
        u32 k_be[8];
        for (ushort i=0;i<8;++i){ ushort off=i*4; u32 w=((u32)p[off]<<24)|((u32)p[off+1]<<16)|((u32)p[off+2]<<8)|((u32)p[off+3]); k_be[i]=w; }
        u32 k_local[8];
        k_local[7]=k_be[0]; k_local[6]=k_be[1]; k_local[5]=k_be[2]; k_local[4]=k_be[3];
        k_local[3]=k_be[4]; k_local[2]=k_be[5]; k_local[1]=k_be[6]; k_local[0]=k_be[7];

        u32 x[8]; u32 y[8]; point_mul_xy_w16(x, y, k_local, g16_table);

        uchar pub[64];
        for (ushort i=0;i<8;++i){ u32 w=x[7-i]; ushort off=i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
        for (ushort i=0;i<8;++i){ u32 w=y[7-i]; ushort off=32+i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }

        uchar digest[32]; keccak256_64(pub, digest);
        uint want = params->nibble & 0xF; uint nibs = params->nibbleCount; bool ok = true;
        uchar want_byte = (uchar)((want << 4) | want); uint full = nibs >> 1; uint rem = nibs & 1u;
        for (uint i = 0; i < full; ++i) { if (digest[12 + i] != want_byte) { ok = false; break; } }
        if (ok && rem) { uchar b = digest[12 + full]; if ((b >> 4) != want) ok = false; }
        flags_out[gid] = ok ? (uchar)1 : (uchar)0;
        if (ok) { device uchar *addr = addr_out + gid * 20; for (ushort i=0;i<20;++i) addr[i] = digest[12 + i]; }
    }

    KERNEL_FQ void vanity_kernel_w16_compact(
        device const uchar *priv_in                  [[ buffer(0) ]],
        device uchar *addr_compact_out               [[ buffer(1) ]],
        device uint *index_compact_out               [[ buffer(2) ]],
        device atomic_uint *out_count                [[ buffer(3) ]],
        device const VanityParams *params            [[ buffer(4) ]],
        device const uchar *g16_table                [[ buffer(5) ]],
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
        if (ok) { uint idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed); device uchar *addr = addr_compact_out + idx * 20; for (ushort i=0;i<20;++i) addr[i] = digest[12 + i]; index_compact_out[idx] = gid; }
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
'''
        )

        source = "\n".join(src_parts)

        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if library is None:
            raise RuntimeError(f"Metal library compile failed: {error}")

        fn = library.newFunctionWithName_("vanity_kernel")
        if fn is None:
            raise RuntimeError("vanity_kernel not found")
        fn_compact = library.newFunctionWithName_("vanity_kernel_compact")
        if fn_compact is None:
            raise RuntimeError("vanity_kernel_compact not found")

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline: {error}")
        pipeline_compact, error = self.device.newComputePipelineStateWithFunction_error_(fn_compact, None)
        if pipeline_compact is None:
            raise RuntimeError(f"Failed to create compact pipeline: {error}")

        self.pipeline = pipeline
        self.pipeline_compact = pipeline_compact
        # Try to compile w16 kernels
        self.pipeline_w16 = None
        self.pipeline_w16_compact = None
        fn_w16 = library.newFunctionWithName_("vanity_kernel_w16")
        fn_w16c = library.newFunctionWithName_("vanity_kernel_w16_compact")
        fn_builder = library.newFunctionWithName_("g16_builder_kernel")
        if fn_w16 is not None and fn_w16c is not None:
            p_w16, error = self.device.newComputePipelineStateWithFunction_error_(fn_w16, None)
            if p_w16 is not None:
                self.pipeline_w16 = p_w16
            p_w16c, error = self.device.newComputePipelineStateWithFunction_error_(fn_w16c, None)
            if p_w16c is not None:
                self.pipeline_w16_compact = p_w16c
        self.pipeline_builder = None
        if fn_builder is not None:
            p_b, error = self.device.newComputePipelineStateWithFunction_error_(fn_builder, None)
            if p_b is not None:
                self.pipeline_builder = p_b

        # Single command queue is sufficient; command buffers pipeline naturally
        self.queue = self.device.newCommandQueue()
        self.thread_execution_width = self.pipeline.threadExecutionWidth()
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

    def build_g16_table(self, repo_root: str) -> None:
        # Build g16 table entirely on GPU; writes to memory, caller can persist to file if desired
        if self.pipeline_builder is None:
            raise RuntimeError("g16_builder_kernel pipeline not available")
        total_bytes = 16 * 65536 * 64
        if self.g16_buffer is None or int(self.g16_buffer.length()) != total_bytes:
            self.g16_buffer = self.device.newBufferWithLength_options_(total_bytes, 0)
        # Build per window in chunks to avoid absurd dispatch sizes
        for win in range(16):
            start = 0
            while start < 65536:
                chunk = min(65536 - start, 32768)  # at most 32k threads per dispatch
                params = bytearray(8)
                params[0:4] = int(win).to_bytes(4, "little")
                params[4:8] = int(start).to_bytes(4, "little")
                cb = self.queue.commandBuffer()
                enc = cb.computeCommandEncoder()
                enc.setComputePipelineState_(self.pipeline_builder)
                enc.setBuffer_offset_atIndex_(self.g16_buffer, 0, 0)
                try:
                    enc.setBytes_length_atIndex_(bytes(params), 8, 1)
                except Exception:
                    pbuf = self.device.newBufferWithLength_options_(8, 0)
                    pbuf.contents().as_buffer(8)[:8] = bytes(params)
                    enc.setBuffer_offset_atIndex_(pbuf, 0, 1)
                w = int(self.thread_execution_width)
                try:
                    max_threads = int(self.pipeline_builder.maxTotalThreadsPerThreadgroup())
                except Exception:
                    max_threads = 256
                tpt = min(max_threads, max(w * 4, w), max(1, chunk))
                tg = MTLSizeMake(tpt, 1, 1)
                grid = MTLSizeMake(chunk, 1, 1)
                enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
                enc.endEncoding()
                cb.commit()
                cb.waitUntilCompleted()  # sequential to limit VRAM pressure while building
                start += chunk
        # Optionally write to disk for future runs
        try:
            path = os.path.join(repo_root, "gen_eth", "secp256k1", "g16_precomp_le.bin")
            with open(path, "wb") as f:
                mv = self.g16_buffer.contents().as_buffer(total_bytes)
                f.write(bytes(mv[:total_bytes]))
        except Exception:
            pass
   # --- Pipelined API ---
    class VanityJob:
        def __init__(self, cb, addr_buffer, flags_buffer, out_addr_size: int, out_flag_size: int, count: int):
            self.cb = cb
            self.addr_buffer = addr_buffer
            self.flags_buffer = flags_buffer
            self.out_addr_size = out_addr_size
            self.out_flag_size = out_flag_size
            self.count = count
            # Async completion signaling
            self._done_event: threading.Event = threading.Event()
            self._addrs: Optional[List[bytes]] = None
            self._flags: Optional[List[int]] = None
            self._error: Optional[Exception] = None
            # Timing
            self.cpu_encode_seconds: float = 0.0
            self.gpu_start_time: float = -1.0
            self.gpu_end_time: float = -1.0

    def encode_and_commit(self, privkeys_be32: List[bytes], nibble: int = 0x8, nibble_count: int = 7) -> "MetalVanity.VanityJob":
        if not privkeys_be32:
            raise ValueError("privkeys_be32 must not be empty")
        for i, k in enumerate(privkeys_be32):
            if len(k) != 32:
                raise ValueError(f"privkey[{i}] must be 32 bytes")
        count = len(privkeys_be32)
        in_size = 32 * count
        out_addr_size = 20 * count
        out_flag_size = count

        # Per-job buffers (safe for overlapping command buffers)
        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        addr_buffer = self.device.newBufferWithLength_options_(out_addr_size, 0)
        flags_buffer = self.device.newBufferWithLength_options_(out_flag_size, 0)
        # Params will be set inline via setBytes to avoid allocating a buffer each call

        # Fill input buffer without creating a large temporary joined bytes
        in_mv = in_buffer.contents().as_buffer(in_size)
        for i, k in enumerate(privkeys_be32):
            off = i * 32
            in_mv[off : off + 32] = k

        p = bytearray(12)
        p[0:4] = int(count).to_bytes(4, "little")
        p[4:8] = int(nibble & 0xF).to_bytes(4, "little")
        p[8:12] = int(nibble_count).to_bytes(4, "little")
        t_cpu0 = time.perf_counter()
        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        use_w16 = self.g16_buffer is not None and self.pipeline_w16 is not None
        enc.setComputePipelineState_(self.pipeline_w16 if use_w16 else self.pipeline)
        enc.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(addr_buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(flags_buffer, 0, 2)
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
            max_threads = int((self.pipeline_w16 if use_w16 else self.pipeline).maxTotalThreadsPerThreadgroup())
        except Exception:
            max_threads = 256
        # Use a multiple of execution width for better occupancy, but respect device limits and problem size
        tpt = min(max_threads, max(w * 4, w), max(1, count))
        tg = MTLSizeMake(tpt, 1, 1)
        # Dispatch exactly 'count' threads; avoid padded idle threads
        grid = MTLSizeMake(count, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()

        job = MetalVanity.VanityJob(cb, addr_buffer, flags_buffer, out_addr_size, out_flag_size, count)

        # Attach completion handler so results are captured without explicit wait on GPU
        def _on_completed(inner_cb):
            try:
                addr_bytes = bytes(job.addr_buffer.contents().as_buffer(job.out_addr_size)[:job.out_addr_size])
                flags_bytes = bytes(job.flags_buffer.contents().as_buffer(job.out_flag_size)[:job.out_flag_size])
                job._addrs = [addr_bytes[i * 20 : (i + 1) * 20] for i in range(job.count)]
                job._flags = [flags_bytes[i] for i in range(job.count)]
                # GPU timing if supported
                try:
                    job.gpu_start_time = float(inner_cb.GPUStartTime())  # type: ignore[attr-defined]
                    job.gpu_end_time = float(inner_cb.GPUEndTime())      # type: ignore[attr-defined]
                except Exception:
                    pass
            except Exception as e:
                job._error = e
            finally:
                job._done_event.set()

        cb.addCompletedHandler_(_on_completed)
        # Submit
        cb.commit()
        job.cpu_encode_seconds = time.perf_counter() - t_cpu0

        return job

    def wait_and_collect(self, job: "MetalVanity.VanityJob") -> Tuple[List[bytes], List[int]]:
        # Wait for asynchronous completion handler to finish copying results
        job._done_event.wait()
        if job._error is not None:
            raise job._error
        # mypy: these are set once done_event is set
        return job._addrs or [], job._flags or []

    # --- Compact-output pipelined API ---
    class VanityJobCompact:
        def __init__(self, cb, addr_compact_buffer, index_compact_buffer, out_count_buffer, capacity_count: int):
            self.cb = cb
            self.addr_compact_buffer = addr_compact_buffer
            self.index_compact_buffer = index_compact_buffer
            self.out_count_buffer = out_count_buffer
            self.capacity_count = capacity_count
            self._done_event: threading.Event = threading.Event()
            self._addrs: Optional[List[bytes]] = None
            self._indices: Optional[List[int]] = None
            self._error: Optional[Exception] = None
            # Timing
            self.cpu_encode_seconds: float = 0.0
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
        # Worst-case we allocate full capacity; actual matches are typically few
        addr_cap_bytes = 20 * count
        idx_cap_bytes = 4 * count

        # Per-job buffers; overlapping jobs are expected
        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        addr_compact_buffer = self.device.newBufferWithLength_options_(addr_cap_bytes, 0)
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
        use_w16 = self.g16_buffer is not None and self.pipeline_w16_compact is not None
        enc.setComputePipelineState_(self.pipeline_w16_compact if use_w16 else self.pipeline_compact)
        enc.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(addr_compact_buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(index_compact_buffer, 0, 2)
        enc.setBuffer_offset_atIndex_(out_count_buffer, 0, 3)
        # Prefer setBytes for small constant params; fall back to a temp buffer if unavailable
        try:
            enc.setBytes_length_atIndex_(bytes(p), 12, 4)
        except Exception:
            params_buffer = self.device.newBufferWithLength_options_(12, 0)
            params_buffer.contents().as_buffer(12)[:12] = bytes(p)
            enc.setBuffer_offset_atIndex_(params_buffer, 0, 4)
        if use_w16:
            enc.setBuffer_offset_atIndex_(self.g16_buffer, 0, 5)

        w = int(self.thread_execution_width)
        try:
            max_threads = int((self.pipeline_w16_compact if use_w16 else self.pipeline_compact).maxTotalThreadsPerThreadgroup())
        except Exception:
            max_threads = 256
        tpt = min(max_threads, max(w * 4, w), max(1, count))
        tg = MTLSizeMake(tpt, 1, 1)
        # Dispatch exactly 'count' threads; avoid padded idle threads
        grid = MTLSizeMake(count, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()

        job = MetalVanity.VanityJobCompact(cb, addr_compact_buffer, index_compact_buffer, out_count_buffer, count)

        def _on_completed(inner_cb):
            try:
                # Read number of matches
                c_bytes = bytes(job.out_count_buffer.contents().as_buffer(4)[:4])
                out_count = int.from_bytes(c_bytes, "little")
                out_count = max(0, min(out_count, job.capacity_count))
                # Read compact addresses and indices
                addr_bytes = bytes(job.addr_compact_buffer.contents().as_buffer(20 * job.capacity_count)[: 20 * out_count])
                idx_bytes = bytes(job.index_compact_buffer.contents().as_buffer(4 * job.capacity_count)[: 4 * out_count])
                addrs = [addr_bytes[i * 20 : (i + 1) * 20] for i in range(out_count)]
                indices = [int.from_bytes(idx_bytes[i * 4 : (i + 1) * 4], "little") for i in range(out_count)]
                job._addrs = addrs
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
                job._done_event.set()

        cb.addCompletedHandler_(_on_completed)
        cb.commit()
        job.cpu_encode_seconds = time.perf_counter() - t_cpu0
        return job

    def wait_and_collect_compact(self, job: "MetalVanity.VanityJobCompact") -> Tuple[List[bytes], List[int]]:
        job._done_event.wait()
        if job._error is not None:
            raise job._error
        return job._addrs or [], job._indices or []


def generate_valid_privkeys(batch_size: int) -> List[bytes]:
    # Fast incremental generator to remove RNG bottleneck.
    if HAS_NUMPY:
        return _generate_privkeys_incremental_numpy(batch_size)
    raise NotImplementedError("NumPy is not installed")

def _generate_privkeys_incremental_python(batch_size: int, seq_len: int = 8192) -> List[bytes]:
    out: List[bytes] = []
    n = SECP256K1_ORDER_INT
    num_bases = math.ceil(batch_size / seq_len)
    for _ in range(num_bases):
        base = int.from_bytes(secrets.token_bytes(32), "big") % n
        if base == 0:
            base = 1
        # Derive a run of consecutive scalars in [1, n-1]
        for i in range(seq_len):
            k = base + i
            if k >= n:
                k -= n
            if k == 0:
                k = 1
            out.append(k.to_bytes(32, "big"))
            if len(out) >= batch_size:
                return out
    return out


def _generate_privkeys_incremental_numpy(batch_size: int, seq_len: int = 8192) -> List[bytes]:  # heavy vector path
    import numpy as np  # type: ignore

    # Build n as little-endian u64 limbs [w0,w1,w2,w3]
    n_be_u64 = np.frombuffer(SECP256K1_ORDER_BYTES, dtype=">u8").reshape(4)
    n_le_u64 = n_be_u64[::-1].byteswap().view(n_be_u64.dtype.newbyteorder())

    num_bases = int(math.ceil(batch_size / seq_len))
    # Draw random bases
    rnd = secrets.token_bytes(32 * num_bases)
    base_be_u64 = np.frombuffer(rnd, dtype=">u8").reshape(num_bases, 4)
    base_le = base_be_u64[:, ::-1].byteswap().view(base_be_u64.dtype.newbyteorder())

    # Reduce bases: if base >= n, subtract n
    ge_mask = _lex_ge_mask_le(base_le, n_le_u64)
    if ge_mask.any():
        base_le = _sub_256_le_inplace(base_le, n_le_u64, ge_mask)

    # Avoid zero
    zero_mask = (base_le == 0).all(axis=1)
    if zero_mask.any():
        base_le[zero_mask, 0] = 1

    # Vectorized increments over least-significant limb with carry propagation
    inc = np.arange(seq_len, dtype=np.uint64)[None, :]
    w0 = base_le[:, [0]] + inc
    carry = (w0 < base_le[:, [0]]).astype(np.uint64)
    w1 = base_le[:, [1]] + carry
    carry = (w1 < base_le[:, [1]]).astype(np.uint64)
    w2 = base_le[:, [2]] + carry
    carry = (w2 < base_le[:, [2]]).astype(np.uint64)
    w3 = base_le[:, [3]] + carry

    # Conditional subtract n if >= n
    ge = _lex_ge_mask_le_2d(w0, w1, w2, w3, n_le_u64)
    if ge.any():
        u0, u1, u2, u3 = _sub_256_le_broadcast(w0, w1, w2, w3, n_le_u64)
        w0 = np.where(ge, u0, w0)
        w1 = np.where(ge, u1, w1)
        w2 = np.where(ge, u2, w2)
        w3 = np.where(ge, u3, w3)

    # Avoid zero results (rare when base + inc == n)
    zero_mask_2d = (w0 == 0) & (w1 == 0) & (w2 == 0) & (w3 == 0)
    if zero_mask_2d.any():
        w0 = np.where(zero_mask_2d, np.uint64(1), w0)

    # Repack to big-endian 32-byte scalars
    B = num_bases
    S = seq_len
    vals = np.empty((B * S, 4), dtype=np.uint64)
    vals[:, 0] = w3.reshape(-1)
    vals[:, 1] = w2.reshape(-1)
    vals[:, 2] = w1.reshape(-1)
    vals[:, 3] = w0.reshape(-1)
    be = vals.byteswap().view(vals.dtype.newbyteorder())
    need = int(batch_size)
    be = be[:need]
    out_bytes = be.tobytes()
    return [out_bytes[i * 32 : (i + 1) * 32] for i in range(need)]


def _lex_ge_mask_le(a_le, n_le):
    # a_le: (N,4) little-endian limbs; n_le: (4,)
    m3 = a_le[:, 3] > n_le[3]
    e3 = a_le[:, 3] == n_le[3]
    m2 = a_le[:, 2] > n_le[2]
    e2 = a_le[:, 2] == n_le[2]
    m1 = a_le[:, 1] > n_le[1]
    e1 = a_le[:, 1] == n_le[1]
    m0 = a_le[:, 0] >= n_le[0]
    return m3 | (e3 & (m2 | (e2 & (m1 | (e1 & m0)))))


def _sub_256_le_inplace(a_le, n_le, select_mask):
    # a := a - n for rows where select_mask is True; little-endian limbs
    sel = select_mask
    if not sel.any():
        return a_le
    a0 = a_le[sel, 0]
    a1 = a_le[sel, 1]
    a2 = a_le[sel, 2]
    a3 = a_le[sel, 3]
    n0, n1, n2, n3 = (n_le[0], n_le[1], n_le[2], n_le[3])
    r0 = a0 - n0
    borrow = (a0 < n0).astype(a0.dtype)
    r1 = a1 - n1 - borrow
    borrow = ((a1 < n1) | ((a1 == n1) & (borrow == 1))).astype(a1.dtype)
    r2 = a2 - n2 - borrow
    borrow = ((a2 < n2) | ((a2 == n2) & (borrow == 1))).astype(a2.dtype)
    r3 = a3 - n3 - borrow
    a_le[sel, 0] = r0
    a_le[sel, 1] = r1
    a_le[sel, 2] = r2
    a_le[sel, 3] = r3
    return a_le


def _lex_ge_mask_le_2d(w0, w1, w2, w3, n_le):
    m3 = w3 > n_le[3]
    e3 = w3 == n_le[3]
    m2 = w2 > n_le[2]
    e2 = w2 == n_le[2]
    m1 = w1 > n_le[1]
    e1 = w1 == n_le[1]
    m0 = w0 >= n_le[0]
    return m3 | (e3 & (m2 | (e2 & (m1 | (e1 & m0)))))


def _sub_256_le_broadcast(w0, w1, w2, w3, n_le):
    n0, n1, n2, n3 = (n_le[0], n_le[1], n_le[2], n_le[3])
    r0 = w0 - n0
    borrow = (w0 < n0).astype(w0.dtype)
    r1 = w1 - n1 - borrow
    borrow = ((w1 < n1) | ((w1 == n1) & (borrow == 1))).astype(w1.dtype)
    r2 = w2 - n2 - borrow
    borrow = ((w2 < n2) | ((w2 == n2) & (borrow == 1))).astype(w2.dtype)
    r3 = w3 - n3 - borrow
    return r0, r1, r2, r3


