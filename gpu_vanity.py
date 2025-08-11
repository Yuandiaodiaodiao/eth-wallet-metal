import os
import secrets
from typing import List, Tuple

from Metal import MTLCreateSystemDefaultDevice, MTLSizeMake


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
        src_parts.append("#undef SECP256K1_TMPS_TYPE\n#define SECP256K1_TMPS_TYPE PRIVATE_AS\n")
        src_parts.append(_load_text_no_includes(os.path.join(secp_dir, "inc_types.h")))
        src_parts.append("\n")
        src_parts.append(_load_text_no_includes(os.path.join(secp_dir, "inc_ecc_secp256k1.h")))
        src_parts.append("\n")
        src_parts.append(_load_text_no_includes(os.path.join(secp_dir, "inc_ecc_secp256k1.cl")))

        # Append Keccak functions (fixed-size 64-byte input)
        src_parts.append(
            r'''
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
    ulong A[25]; for(ushort i=0;i<25;++i) A[i]=0UL;
    uchar block[136]; for(ushort i=0;i<136;++i) block[i]=0; for(ushort i=0;i<64;++i) block[i]=msg64[i];
    block[64]^=0x01; block[135]^=0x80;
    for(ushort i=0;i<17;++i){ const thread uchar* lane = block + i*8; A[i]^=load64_thread(lane); }
    keccak_f1600_inline(A);
    // squeeze first 32 bytes little-endian of lanes 0..3
    for(ushort i=0;i<4;++i){ store64_le(A[i], out32 + i*8); }
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

    secp256k1_t tmps; set_precomputed_basepoint_g(&tmps);
    u32 x[8]; u32 y[8]; point_mul_xy(x, y, k_local, &tmps);

    // pack pub (x||y) big-endian into thread buffer
    uchar pub[64];
    for (ushort i=0;i<8;++i){ u32 w=x[7-i]; ushort off=i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }
    for (ushort i=0;i<8;++i){ u32 w=y[7-i]; ushort off=32+i*4; pub[off]=w>>24; pub[off+1]=w>>16; pub[off+2]=w>>8; pub[off+3]=w; }

    // keccak
    uchar digest[32]; keccak256_64(pub, digest);
    // address = last 20 bytes
    device uchar *addr = addr_out + gid * 20;
    for (ushort i=0;i<20;++i) addr[i] = digest[12 + i];

    // vanity check: first nibbleCount hex nibbles equal to nibble value
    uint want = params->nibble & 0xF;
    uint nibs = params->nibbleCount;
    bool ok = true;
    for (uint n=0; n<nibs; ++n){
        uint byteIndex = n >> 1; bool high = ((n & 1u) == 0u);
        uchar b = addr[byteIndex]; uint nib = high ? (b >> 4) : (b & 0xF);
        if (nib != want) { ok = false; break; }
    }
    flags_out[gid] = ok ? (uchar)1 : (uchar)0;
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

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline: {error}")

        self.pipeline = pipeline
        # Use two distinct command queues and alternate to avoid sharing a single queue's buffer lifecycle
        self.queue_a = self.device.newCommandQueue()
        self.queue_b = self.device.newCommandQueue()
        self._use_queue_a = True
        self.thread_execution_width = self.pipeline.threadExecutionWidth()

    def run_batch(self, privkeys_be32: List[bytes], nibble: int = 0x8, nibble_count: int = 7) -> Tuple[List[bytes], List[int]]:
        if not privkeys_be32:
            return [], []
        for i, k in enumerate(privkeys_be32):
            if len(k) != 32:
                raise ValueError(f"privkey[{i}] must be 32 bytes")
        count = len(privkeys_be32)
        in_size = 32 * count
        out_addr_size = 20 * count
        out_flag_size = count

        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        addr_buffer = self.device.newBufferWithLength_options_(out_addr_size, 0)
        flags_buffer = self.device.newBufferWithLength_options_(out_flag_size, 0)
        params_buffer = self.device.newBufferWithLength_options_(12, 0)

        joined = b"".join(privkeys_be32)
        in_buffer.contents().as_buffer(in_size)[:in_size] = joined

        # params
        p = bytearray(12)
        p[0:4] = int(count).to_bytes(4, "little")
        p[4:8] = int(nibble & 0xF).to_bytes(4, "little")
        p[8:12] = int(nibble_count).to_bytes(4, "little")
        params_buffer.contents().as_buffer(12)[:12] = bytes(p)

        # Alternate command queues between jobs
        q = self.queue_a if self._use_queue_a else self.queue_b
        self._use_queue_a = not self._use_queue_a
        cb = q.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline)
        enc.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(addr_buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(flags_buffer, 0, 2)
        enc.setBuffer_offset_atIndex_(params_buffer, 0, 3)

        w = int(self.thread_execution_width)
        tpt = min(w, 256)
        tg = MTLSizeMake(tpt, 1, 1)
        grid = MTLSizeMake(((count + tpt - 1) // tpt) * tpt, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        addr_bytes = bytes(addr_buffer.contents().as_buffer(out_addr_size)[:out_addr_size])
        flags_bytes = bytes(flags_buffer.contents().as_buffer(out_flag_size)[:out_flag_size])
        addrs = [addr_bytes[i*20:(i+1)*20] for i in range(count)]
        flags = [flags_bytes[i] for i in range(count)]
        return addrs, flags

    # --- Pipelined API ---
    class VanityJob:
        def __init__(self, cb, addr_buffer, flags_buffer, out_addr_size: int, out_flag_size: int, count: int):
            self.cb = cb
            self.addr_buffer = addr_buffer
            self.flags_buffer = flags_buffer
            self.out_addr_size = out_addr_size
            self.out_flag_size = out_flag_size
            self.count = count

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

        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        addr_buffer = self.device.newBufferWithLength_options_(out_addr_size, 0)
        flags_buffer = self.device.newBufferWithLength_options_(out_flag_size, 0)
        params_buffer = self.device.newBufferWithLength_options_(12, 0)

        joined = b"".join(privkeys_be32)
        in_buffer.contents().as_buffer(in_size)[:in_size] = joined

        p = bytearray(12)
        p[0:4] = int(count).to_bytes(4, "little")
        p[4:8] = int(nibble & 0xF).to_bytes(4, "little")
        p[8:12] = int(nibble_count).to_bytes(4, "little")
        params_buffer.contents().as_buffer(12)[:12] = bytes(p)
        q = self.queue_a if self._use_queue_a else self.queue_b
        self._use_queue_a = not self._use_queue_a
        cb = q.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline)
        enc.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(addr_buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(flags_buffer, 0, 2)
        enc.setBuffer_offset_atIndex_(params_buffer, 0, 3)

        w = int(self.thread_execution_width)
        tpt = min(w, 256)
        tg = MTLSizeMake(tpt, 1, 1)
        grid = MTLSizeMake(((count + tpt - 1) // tpt) * tpt, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        # Non-blocking submit
        cb.commit()

        return MetalVanity.VanityJob(cb, addr_buffer, flags_buffer, out_addr_size, out_flag_size, count)

    def wait_and_collect(self, job: "MetalVanity.VanityJob") -> Tuple[List[bytes], List[int]]:
        job.cb.waitUntilCompleted()
        addr_bytes = bytes(job.addr_buffer.contents().as_buffer(job.out_addr_size)[:job.out_addr_size])
        flags_bytes = bytes(job.flags_buffer.contents().as_buffer(job.out_flag_size)[:job.out_flag_size])
        addrs = [addr_bytes[i*20:(i+1)*20] for i in range(job.count)]
        flags = [flags_bytes[i] for i in range(job.count)]
        return addrs, flags


def generate_valid_privkeys(batch_size: int) -> List[bytes]:
    # Ensure 1 <= k < n
    from ecdsa import SECP256k1
    n = SECP256k1.order
    out: List[bytes] = []
    while len(out) < batch_size:
        k = secrets.token_bytes(32)
        ki = int.from_bytes(k, "big")
        if 1 <= ki < n:
            out.append(k)
    return out


