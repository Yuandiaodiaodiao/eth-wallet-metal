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
        self.max_threads_per_tg = getattr(self.pipeline, "maxTotalThreadsPerThreadgroup")()

        # Reusable buffers to reduce allocation overhead
        self._in_buffer = None
        self._out_buffer = None
        self._in_capacity = 0
        self._out_capacity = 0

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

        # Ensure and reuse buffers
        if self._in_buffer is None or self._in_capacity < in_size:
            self._in_buffer = self.device.newBufferWithLength_options_(in_size, 0)
            self._in_capacity = in_size
        if self._out_buffer is None or self._out_capacity < out_size:
            self._out_buffer = self.device.newBufferWithLength_options_(out_size, 0)
            self._out_capacity = out_size
        if self._in_buffer is None or self._out_buffer is None:
            raise RuntimeError("Failed to allocate Metal buffers")

        # Copy input bytes into GPU buffer
        # Concatenate inputs
        joined = b"".join(inputs)
        in_mv = self._in_buffer.contents().as_buffer(in_size)
        in_mv[:in_size] = joined

        # Build and encode compute command
        command_buffer = self.queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()
        encoder.setComputePipelineState_(self.pipeline)
        encoder.setBuffer_offset_atIndex_(self._in_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(self._out_buffer, 0, 1)

        # Grid config: one thread per item
        w = int(self.thread_execution_width)
        # choose the largest multiple of execution width that fits the device limit, max 512 for safety
        limit = int(self.max_threads_per_tg) if self.max_threads_per_tg else 512
        threads_per_tg = min(max(w, 1) * max(1, (limit // max(w, 1))), limit)
        if threads_per_tg == 0:
            threads_per_tg = w or 64
        tg_size = MTLSizeMake(threads_per_tg, 1, 1)
        grid = MTLSizeMake(count, 1, 1)
        encoder.dispatchThreads_threadsPerThreadgroup_(grid, tg_size)
        encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Read back results
        out_mv = self._out_buffer.contents().as_buffer(out_size)
        out_bytes = bytes(out_mv[:out_size])
        return [out_bytes[i * 32 : (i + 1) * 32] for i in range(count)]


def keccak256_gpu_single(pubkey64: bytes, metal_path: str) -> bytes:
    engine = MetalKeccak256(metal_path)
    return engine.keccak256_many([pubkey64])[0]


