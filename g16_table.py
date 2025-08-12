import os
from Metal import MTLSizeMake


def build_g16_table(metal_vanity_instance, repo_root: str) -> None:
    """Build g16 table entirely on GPU; writes to memory, caller can persist to file if desired"""
    if metal_vanity_instance.pipeline_builder is None:
        raise RuntimeError("g16_builder_kernel pipeline not available")
    total_bytes = 16 * 65536 * 64
    if metal_vanity_instance.g16_buffer is None or int(metal_vanity_instance.g16_buffer.length()) != total_bytes:
        metal_vanity_instance.g16_buffer = metal_vanity_instance.device.newBufferWithLength_options_(total_bytes, 0)
    # Build per window in chunks to avoid absurd dispatch sizes
    for win in range(16):
        start = 0
        while start < 65536:
            chunk = min(65536 - start, 32768)  # at most 32k threads per dispatch
            params = bytearray(8)
            params[0:4] = int(win).to_bytes(4, "little")
            params[4:8] = int(start).to_bytes(4, "little")
            cb = metal_vanity_instance.queue.commandBuffer()
            enc = cb.computeCommandEncoder()
            enc.setComputePipelineState_(metal_vanity_instance.pipeline_builder)
            enc.setBuffer_offset_atIndex_(metal_vanity_instance.g16_buffer, 0, 0)
            try:
                enc.setBytes_length_atIndex_(bytes(params), 8, 1)
            except Exception:
                pbuf = metal_vanity_instance.device.newBufferWithLength_options_(8, 0)
                pbuf.contents().as_buffer(8)[:8] = bytes(params)
                enc.setBuffer_offset_atIndex_(pbuf, 0, 1)
            w = int(metal_vanity_instance.thread_execution_width)
            try:
                max_threads = int(metal_vanity_instance.pipeline_builder.maxTotalThreadsPerThreadgroup())
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
            mv = metal_vanity_instance.g16_buffer.contents().as_buffer(total_bytes)
            f.write(bytes(mv[:total_bytes]))
    except Exception:
        pass