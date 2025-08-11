import os
import ctypes
from typing import List

from Metal import (
    MTLCreateSystemDefaultDevice,
    MTLSizeMake,
)


class MetalKeccak256:
    def __init__(self, metal_source_path: str) -> None:
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal device available. Requires macOS with Metal-capable GPU.")

        if not os.path.isfile(metal_source_path):
            raise FileNotFoundError(f"Metal source not found: {metal_source_path}")

        with open(metal_source_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Compile compute library
        library, error = self.device.newLibraryWithSource_options_error_(source, None, None)
        if library is None:
            raise RuntimeError(f"Metal library compile failed: {error}")

        fn = library.newFunctionWithName_("keccak256_kernel")
        if fn is None:
            raise RuntimeError("keccak256_kernel not found in Metal library")

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create compute pipeline: {error}")

        self.pipeline = pipeline
        self.queue = self.device.newCommandQueue()
        if self.queue is None:
            raise RuntimeError("Failed to create Metal command queue")

        self.thread_execution_width = self.pipeline.threadExecutionWidth()

    def keccak256_many(self, inputs: List[bytes]) -> List[bytes]:
        if not inputs:
            return []
        # All inputs must be exactly 64 bytes (uncompressed pubkey x||y)
        for idx, b in enumerate(inputs):
            if len(b) != 64:
                raise ValueError(f"Input at index {idx} is {len(b)} bytes; expected 64")

        count = len(inputs)
        in_size = 64 * count
        out_size = 32 * count

        # Allocate buffers
        in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
        out_buffer = self.device.newBufferWithLength_options_(out_size, 0)
        count_buffer = self.device.newBufferWithLength_options_(ctypes.sizeof(ctypes.c_uint), 0)
        if in_buffer is None or out_buffer is None:
            raise RuntimeError("Failed to allocate Metal buffers")
        if count_buffer is None:
            raise RuntimeError("Failed to allocate Metal count buffer")

        # Copy input bytes into GPU buffer
        # Concatenate inputs
        joined = b"".join(inputs)
        in_mv = in_buffer.contents().as_buffer(in_size)
        in_mv[:in_size] = joined

        # Build and encode compute command
        command_buffer = self.queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        encoder.setBuffer_offset_atIndex_(in_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(out_buffer, 0, 1)
        encoder.setBuffer_offset_atIndex_(count_buffer, 0, 2)

        # Write count to buffer
        count_mv = count_buffer.contents().as_buffer(4)
        count_mv[:4] = int(count).to_bytes(4, "little")

        # Grid config: one thread per item
        w = int(self.thread_execution_width)
        threads_per_tg = min(w, 256)
        tg_size = MTLSizeMake(threads_per_tg, 1, 1)
        num_tg = (count + threads_per_tg - 1) // threads_per_tg
        grid = MTLSizeMake(num_tg * threads_per_tg, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid, tg_size)
        encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Read back results
        out_mv = out_buffer.contents().as_buffer(out_size)
        out_bytes = bytes(out_mv[:out_size])
        return [out_bytes[i * 32 : (i + 1) * 32] for i in range(count)]


def keccak256_gpu_single(pubkey64: bytes, metal_path: str) -> bytes:
    engine = MetalKeccak256(metal_path)
    return engine.keccak256_many([pubkey64])[0]


