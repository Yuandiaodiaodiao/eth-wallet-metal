import os
from typing import List

from Metal import MTLCreateSystemDefaultDevice, MTLSizeMake


def _load_text_no_includes(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    return "\n".join(l for l in lines if not l.strip().startswith("#include"))


class MetalSecp256k1:
    def __init__(self, repo_root: str) -> None:
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device available")

        secp_dir = os.path.join(repo_root, "gen_eth", "secp256k1")

        # Build a single Metal source by concatenating headers and CL code (strip includes)
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
        src_parts.append("\n")
        # Add our Metal kernel wrapper to call point_mul_xy and emit 64 bytes (x||y) big-endian
        src_parts.append(
            r'''
KERNEL_FQ void secp256k1_pubkey_kernel(
    device const uchar *priv_in   [[ buffer(0) ]],
    device uchar *pub_out         [[ buffer(1) ]],
    device const uint *count_ptr  [[ buffer(2) ]],
    uint gid                      [[ thread_position_in_grid ]])
{
    uint count = *count_ptr;
    if (gid >= count) return;

    // Load 32-byte big-endian private key and convert to 8x u32 words
    // big-endian bytes -> k_be[0]..k_be[7] (MSW..LSW)
    const device uchar *p = priv_in + gid * 32;
    u32 k_be[8];
    for (ushort i = 0; i < 8; ++i) {
        ushort off = i * 4;
        u32 w = ((u32)p[off + 0] << 24) | ((u32)p[off + 1] << 16) | ((u32)p[off + 2] << 8) | (u32)p[off + 3];
        k_be[i] = w;
    }

    // Convert to little-endian limb order expected by library: k_local[0]=LSW, k_local[7]=MSW
    u32 k_local[8];
    k_local[7] = k_be[0];
    k_local[6] = k_be[1];
    k_local[5] = k_be[2];
    k_local[4] = k_be[3];
    k_local[3] = k_be[4];
    k_local[2] = k_be[5];
    k_local[1] = k_be[6];
    k_local[0] = k_be[7];

    // Precompute basepoint table and multiply
    secp256k1_t tmps;
    set_precomputed_basepoint_g(&tmps);

    u32 x[8];
    u32 y[8];
    point_mul_xy(x, y, k_local, &tmps);

    // Store x||y in big-endian to pub_out
    device uchar *out = pub_out + gid * 64;
    // x: reverse limb order (little->big) and write bytes MSB first per limb
    for (ushort i = 0; i < 8; ++i) {
        u32 w = x[7 - i];
        out[i * 4 + 0] = (uchar)(w >> 24);
        out[i * 4 + 1] = (uchar)(w >> 16);
        out[i * 4 + 2] = (uchar)(w >> 8);
        out[i * 4 + 3] = (uchar)(w);
    }
    // y: same, offset by 32 bytes
    for (ushort i = 0; i < 8; ++i) {
        u32 w = y[7 - i];
        out[32 + i * 4 + 0] = (uchar)(w >> 24);
        out[32 + i * 4 + 1] = (uchar)(w >> 16);
        out[32 + i * 4 + 2] = (uchar)(w >> 8);
        out[32 + i * 4 + 3] = (uchar)(w);
    }
}
'''
        )

        source = "\n".join(src_parts)

        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if library is None:
            raise RuntimeError(f"Metal library compile failed: {error}")

        fn = library.newFunctionWithName_("secp256k1_pubkey_kernel")
        if fn is None:
            raise RuntimeError("secp256k1_pubkey_kernel not found")

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline: {error}")

        self.pipeline = pipeline
        self.queue = self.device.newCommandQueue()
        self.thread_execution_width = self.pipeline.threadExecutionWidth()

    def pubkey_many(self, privkeys_be32: List[bytes]) -> List[bytes]:
        if not privkeys_be32:
            return []
        for i, k in enumerate(privkeys_be32):
            if len(k) != 32:
                raise ValueError(f"privkey[{i}] must be 32 bytes")

        count = len(privkeys_be32)
        in_size = 32 * count
        out_size = 64 * count

        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        out_buffer = self.device.newBufferWithLength_options_(out_size, 0)
        count_buffer = self.device.newBufferWithLength_options_(4, 0)

        # Fill inputs
        joined = b"".join(privkeys_be32)
        in_buffer.contents().as_buffer(in_size)[:in_size] = joined
        count_buffer.contents().as_buffer(4)[:4] = int(count).to_bytes(4, "little")

        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipeline)
        enc.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        enc.setBuffer_offset_atIndex_(out_buffer, 0, 1)
        enc.setBuffer_offset_atIndex_(count_buffer, 0, 2)

        w = int(self.thread_execution_width)
        tpt = min(w, 256)
        tg = MTLSizeMake(tpt, 1, 1)
        grid = MTLSizeMake(((count + tpt - 1) // tpt) * tpt, 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        data = bytes(out_buffer.contents().as_buffer(out_size)[:out_size])
        return [data[i * 64 : (i + 1) * 64] for i in range(count)]


def secp256k1_pubkey_gpu(privkey32: bytes, repo_root: str) -> bytes:
    engine = MetalSecp256k1(repo_root)
    return engine.pubkey_many([privkey32])[0]


