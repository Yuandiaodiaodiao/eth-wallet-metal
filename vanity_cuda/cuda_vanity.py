"""
CUDA-based Ethereum vanity address generator using cuda-python
Following best practices from cuda-python examples
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

# Use cuda.core for modern CUDA programming
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch, MemoryResource
import cupy as cp
import concurrent.futures
from dataclasses import dataclass

@dataclass
class AsyncGpuResult:
    """Result from async GPU computation"""
    future: concurrent.futures.Future
    stream: 'Device.Stream'
    start_event: 'Device.Event'
    end_event: 'Device.Event'
    d_found_indices: cp.ndarray
    d_found_count: cp.ndarray
    batch_size: int
    steps_per_thread: int

class CudaVanity:
    """CUDA-accelerated vanity address generator"""
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA device and compile kernels"""
        # Set up device
        self.device = Device(device_id)
        self.device.set_current()
        
        # Create streams for async operations
        self.stream = self.device.create_stream()
        self.stream_g16 = self.device.create_stream()  # Separate stream for G16 table operations
        
        # Additional streams for parallel processing
        self.stream_compute = self.device.create_stream()  # Main computation stream
        self.stream_transfer = self.device.create_stream() # Data transfer stream
        self.stream_prefetch = self.device.create_stream() # Prefetch next batch stream
        
        # Events for synchronization
        self.transfer_complete_event = self.device.create_event()
        self.compute_complete_event = self.device.create_event()
        
        # Thread pool for async operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Get device properties
        self.compute_capability = self.device.compute_capability
        self.arch = "".join(f"{i}" for i in self.compute_capability)
        
        print(f"Using CUDA device: {self.device.name}")
        print(f"Compute capability: {self.compute_capability}")
        print(f"Architecture: sm_{self.arch}")
        
        # Load and compile kernels
        self._load_kernels()
        
        # G16 precomputed table
        self.g16_table = None
        self.g16_table_size = 16 * 65536 * 64  # 16 windows * 65536 entries * 64 bytes
        
        # Load or build G16 table
        self._init_g16_table()
        
    def _load_kernels(self):
        """Load and compile CUDA kernels"""
        kernel_dir = Path(__file__).parent
        
        # Read kernel source files
        with open(kernel_dir / "kernels.cu", "r") as f:
            kernel_source = f.read()
        

        
        # Combine sources (headers are included via #include in kernels.cu)
        # For cuda.core, we need to provide the full source with includes resolved
        full_source = kernel_source
        
        # Compile options with minimal includes
        cuda_include_path = os.path.join(os.environ.get("CUDA_PATH", ""), "include")
        include_paths = [str(kernel_dir)]
        if os.path.exists(cuda_include_path):
            include_paths.append(cuda_include_path)
        
        program_options = ProgramOptions(
            arch=f"sm_{self.arch}",
            include_path=include_paths,
            use_fast_math=True
        )
        
        # Create program and compile
        self.program = Program(full_source, code_type="c++", options=program_options)
        
        # Compile to cubin and get kernel functions
        kernel_names = [
            "vanity_kernel_g16",
            "compute_basepoints_g16", 
            "vanity_walker_kernel",
            "build_g16_table_kernel"
        ]
        
        self.module = self.program.compile("cubin", name_expressions=kernel_names)
        
        # Get kernel references
        self.kernel_vanity_g16 = self.module.get_kernel("vanity_kernel_g16")
        self.kernel_compute_base = self.module.get_kernel("compute_basepoints_g16")
        self.kernel_walker = self.module.get_kernel("vanity_walker_kernel")
        self.kernel_build_g16 = self.module.get_kernel("build_g16_table_kernel")
        
    def _init_g16_table(self):
        """Initialize or load G16 precomputed table"""
        g16_path = Path(__file__).parent.parent / "vanity_cuda" / "g16_precomp_le.bin"
        
        if g16_path.exists() and g16_path.stat().st_size == self.g16_table_size:
            # Load precomputed table from file
            print("Loading G16 table from file...")
            with open(g16_path, "rb") as f:
                table_data = np.frombuffer(f.read(), dtype=np.uint8)
            self.g16_table = cp.asarray(table_data, dtype=cp.uint8)
            print(f"G16 table loaded: {self.g16_table_size} bytes")
        else:
            # Build G16 table on GPU
            print("Building G16 table on GPU...")
            self.g16_table = self._build_g16_table()
            
            # Save to file for future use
            print("Saving G16 table to file...")
            g16_path.parent.mkdir(parents=True, exist_ok=True)
            with open(g16_path, "wb") as f:
                f.write(self.g16_table.get().tobytes())
            print(f"G16 table saved: {self.g16_table_size} bytes")
    
    def _build_g16_table(self) -> cp.ndarray:
        """Build G16 precomputed table on GPU"""
        # Allocate table
        g16_table = cp.zeros(self.g16_table_size, dtype=cp.uint8)
        
        # Build table window by window
        entries_per_window = 65536
        entries_per_kernel = 4096  # Process in chunks to avoid timeout
        
        for window in range(16):
            print(f"Building G16 table window {window}/16...")
            
            for start_idx in range(0, entries_per_window, entries_per_kernel):
                count = min(entries_per_kernel, entries_per_window - start_idx)
                
                # Configure kernel launch
                block_size = 256
                grid_size = (count + block_size - 1) // block_size
                config = LaunchConfig(grid=grid_size, block=block_size)
                
                # Launch kernel
                launch(
                    self.stream_g16,
                    config,
                    self.kernel_build_g16,
                    g16_table.data.ptr,      # g16_table
                    cp.uint32(window),       # window
                    cp.uint32(start_idx),    # start_idx
                    cp.uint32(count)         # count
                )
        
        # Wait for completion
        self.stream_g16.sync()
        return g16_table
    
    def generate_vanity_simple(self, 
                              privkeys: List[bytes],
                              head_pattern: str = "",
                              tail_pattern: str = "") -> Tuple[List[int], float, bytes]:
        """
        Generate vanity addresses using simple kernel (one address per thread)
        
        Args:
            privkeys: List of 32-byte private keys (big-endian)
            head_pattern: Hex pattern for address prefix (e.g., "888" for 0x888...)
            tail_pattern: Hex pattern for address suffix (e.g., "abc" for ...abc)
            
        Returns:
            Tuple of (found_indices, gpu_time_seconds, addresses_bytes)
        """
        print(f"Generating {len(privkeys)} private keys...")
        num_keys = len(privkeys)
        
        # Convert pattern strings to byte arrays
        def pattern_to_bytes(pattern: str) -> Tuple[cp.ndarray, int]:
            if not pattern:
                return cp.zeros(0, dtype=cp.uint8), 0
            
            pattern = pattern.lower()
            nibble_count = len(pattern)
            byte_count = (nibble_count + 1) // 2
            
            # Convert hex string to bytes (packed nibbles)
            pattern_bytes = np.zeros(byte_count, dtype=np.uint8)
            for i in range(0, len(pattern), 2):
                if i + 1 < len(pattern):
                    # Two nibbles -> one byte
                    pattern_bytes[i // 2] = (int(pattern[i], 16) << 4) | int(pattern[i + 1], 16)
                else:
                    # One nibble -> half byte (left-padded)
                    pattern_bytes[i // 2] = int(pattern[i], 16) << 4
            
            return cp.asarray(pattern_bytes, dtype=cp.uint8), nibble_count
        
        d_head_pattern, head_nibbles = pattern_to_bytes(head_pattern)
        d_tail_pattern, tail_nibbles = pattern_to_bytes(tail_pattern)
        
        # Ensure we have valid pointers for CUDA (use dummy arrays if pattern is empty)
        if head_nibbles == 0:
            d_head_pattern = cp.zeros(1, dtype=cp.uint8)
        if tail_nibbles == 0:
            d_tail_pattern = cp.zeros(1, dtype=cp.uint8)
        
        # Prepare input data
        privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
        d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
        
        # Allocate output buffers
        max_found = min(1, num_keys)  # Limit output size
        d_found_indices = cp.zeros(max_found, dtype=cp.uint32)
        d_found_count = cp.zeros(1, dtype=cp.uint32)
        d_outbuffer = cp.zeros(num_keys * 20, dtype=cp.uint8)  # 20 bytes per address
        
        # Configure kernel launch
        block_size = 256
        grid_size = (num_keys + block_size - 1) // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        
        # Ensure previous operations are complete
        self.device.sync()
        
        # Launch kernel
        start_time = time.perf_counter()
        launch(
            self.stream,
            config,
            self.kernel_vanity_g16,
            d_privkeys.data.ptr,        # privkeys
            d_found_indices.data.ptr,   # found_indices
            d_found_count.data.ptr,     # found_count
            d_outbuffer.data.ptr,       # outbuffer
            self.g16_table.data.ptr,    # g16_table
            cp.uint32(num_keys),        # num_keys
            d_head_pattern.data.ptr,    # head_pattern
            cp.uint32(head_nibbles),    # head_nibbles
            d_tail_pattern.data.ptr,    # tail_pattern
            cp.uint32(tail_nibbles)     # tail_nibbles
        )
        
        # Wait for completion
        self.stream.sync()
        gpu_time = time.perf_counter() - start_time
        
        # Get results
        found_count = int(d_found_count.get()[0])
        found_indices = d_found_indices[:found_count].get().tolist() if found_count > 0 else []
        addresses = d_outbuffer.get()  # Get all addresses
        
        return found_indices, gpu_time, addresses
    
    def generate_vanity_walker(self,
                               privkeys: List[bytes],
                               steps_per_thread: int = 256,
                               head_pattern: str = "",
                               tail_pattern: str = "") -> Tuple[List[int], float]:
        """
        Generate vanity addresses using walker kernel (multiple addresses per thread)
        
        Args:
            privkeys: List of 32-byte private keys (big-endian)
            steps_per_thread: Number of addresses to check per thread
            head_pattern: Hex pattern for address prefix (e.g., "888" for 0x888...)
            tail_pattern: Hex pattern for address suffix (e.g., "abc" for ...abc)
            
        Returns:
            Tuple of (found_indices, gpu_time_seconds, middle_gpu_time_seconds)
        """
        
        print(f"Generating {len(privkeys)} private keys... {steps_per_thread} steps per thread, head: '{head_pattern}', tail: '{tail_pattern}'")
             
        num_keys = len(privkeys)
        
        # Convert pattern strings to byte arrays
        def pattern_to_bytes(pattern: str) -> Tuple[cp.ndarray, int]:
            if not pattern:
                return cp.zeros(0, dtype=cp.uint8), 0
            
            pattern = pattern.lower()
            nibble_count = len(pattern)
            byte_count = (nibble_count + 1) // 2
            
            # Convert hex string to bytes (packed nibbles)
            pattern_bytes = np.zeros(byte_count, dtype=np.uint8)
            for i in range(0, len(pattern), 2):
                if i + 1 < len(pattern):
                    # Two nibbles -> one byte
                    pattern_bytes[i // 2] = (int(pattern[i], 16) << 4) | int(pattern[i + 1], 16)
                else:
                    # One nibble -> half byte (left-padded)
                    pattern_bytes[i // 2] = int(pattern[i], 16) << 4
            
            return cp.asarray(pattern_bytes, dtype=cp.uint8), nibble_count
        
        d_head_pattern, head_nibbles = pattern_to_bytes(head_pattern)
        d_tail_pattern, tail_nibbles = pattern_to_bytes(tail_pattern)
        
        # Ensure we have valid pointers for CUDA (use dummy arrays if pattern is empty)
        if head_nibbles == 0:
            d_head_pattern = cp.zeros(1, dtype=cp.uint8)
        if tail_nibbles == 0:
            d_tail_pattern = cp.zeros(1, dtype=cp.uint8)
        
        # Prepare input data
        privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
        d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
        
        # Allocate intermediate buffer for base points
        d_basepoints = cp.zeros(num_keys * 16, dtype=cp.uint32)  # 16 uint32s per point
        
        # Allocate output buffers
        max_found = min(1, num_keys * steps_per_thread)
        d_found_indices = cp.zeros(max_found, dtype=cp.uint32)
        d_found_count = cp.zeros(1, dtype=cp.uint32)
        
        # Ensure previous operations are complete
        self.device.sync()
        
        start_time = time.perf_counter()
        
        # Stage 1: Compute base points
        block_size = 64
        grid_size = (num_keys + block_size - 1) // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        print('launch compute base')
        launch(
            self.stream,
            config,
            self.kernel_compute_base,
            d_privkeys.data.ptr,        # privkeys
            d_basepoints.data.ptr,      # basepoints
            self.g16_table.data.ptr,    # g16_table
            cp.uint32(num_keys)         # num_keys
        )
        print('launch walker')
        middle_gpu_time = time.perf_counter() - start_time

        # Stage 2: Walker kernel
        # Use smaller block size for walker due to higher register usage
        block_size = 32
        grid_size = (num_keys + block_size - 1) // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        print(f'launch walker {grid_size} {block_size}')
        launch(
            self.stream,
            config,
            self.kernel_walker,
            d_basepoints.data.ptr,      # basepoints
            d_found_indices.data.ptr,   # found_indices
            d_found_count.data.ptr,     # found_count
            cp.uint32(num_keys),        # num_keys
            cp.uint32(steps_per_thread), # steps_per_thread
            d_head_pattern.data.ptr,    # head_pattern
            cp.uint32(head_nibbles),    # head_nibbles
            d_tail_pattern.data.ptr,    # tail_pattern
            cp.uint32(tail_nibbles)     # tail_nibbles
        )
        print('sync')
        # Wait for completion
        self.stream.sync()
        print('sync done')
        gpu_time = time.perf_counter() - start_time
        
        # Get results
        found_count = int(d_found_count.get()[0])
        found_indices = d_found_indices[:found_count].get().tolist() if found_count > 0 else []
        
        return found_indices, gpu_time , middle_gpu_time
    
    def generate_vanity_walker_async(self,
                                   privkeys: List[bytes],
                                   steps_per_thread: int = 256,
                                   head_pattern: str = "",
                                   tail_pattern: str = "") -> AsyncGpuResult:
        """
        Generate vanity addresses using walker kernel with async execution
        
        Args:
            privkeys: List of 32-byte private keys (big-endian)
            steps_per_thread: Number of addresses to check per thread
            head_pattern: Hex pattern for address prefix
            tail_pattern: Hex pattern for address suffix
            
        Returns:
            AsyncGpuResult with future for getting results
        """
        
        print(f"Starting async generation of {len(privkeys)} private keys...")
        num_keys = len(privkeys)
        
        # Convert pattern strings to byte arrays (same as sync version)
        def pattern_to_bytes(pattern: str) -> Tuple[cp.ndarray, int]:
            if not pattern:
                return cp.zeros(0, dtype=cp.uint8), 0
            
            pattern = pattern.lower()
            nibble_count = len(pattern)
            byte_count = (nibble_count + 1) // 2
            
            pattern_bytes = np.zeros(byte_count, dtype=np.uint8)
            for i in range(0, len(pattern), 2):
                if i + 1 < len(pattern):
                    pattern_bytes[i // 2] = (int(pattern[i], 16) << 4) | int(pattern[i + 1], 16)
                else:
                    pattern_bytes[i // 2] = int(pattern[i], 16) << 4
            
            return cp.asarray(pattern_bytes, dtype=cp.uint8), nibble_count
        
        d_head_pattern, head_nibbles = pattern_to_bytes(head_pattern)
        d_tail_pattern, tail_nibbles = pattern_to_bytes(tail_pattern)
        
        if head_nibbles == 0:
            d_head_pattern = cp.zeros(1, dtype=cp.uint8)
        if tail_nibbles == 0:
            d_tail_pattern = cp.zeros(1, dtype=cp.uint8)
        
        # Prepare input data on transfer stream
        privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
        d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
        
        # Allocate buffers
        d_basepoints = cp.zeros(num_keys * 16, dtype=cp.uint32)
        max_found = min(10, num_keys * steps_per_thread)  # Increased capacity
        d_found_indices = cp.zeros(max_found, dtype=cp.uint32)
        d_found_count = cp.zeros(1, dtype=cp.uint32)
        
        # Create events for this async operation
        start_event = self.device.create_event()
        end_event = self.device.create_event()
        
        # Define async computation function
        def compute_async():
            start_time = time.perf_counter()
            
            # Record start
            start_event.record(self.stream_compute)
            
            # Stage 1: Compute base points
            block_size = 64
            grid_size = (num_keys + block_size - 1) // block_size
            config = LaunchConfig(grid=grid_size, block=block_size)
            
            launch(
                self.stream_compute,
                config,
                self.kernel_compute_base,
                d_privkeys.data.ptr,
                d_basepoints.data.ptr,
                self.g16_table.data.ptr,
                cp.uint32(num_keys)
            )
            
            # Wait for base computation to complete before walker
            self.compute_complete_event.record(self.stream_compute)
            self.stream_compute.wait(self.compute_complete_event)
            
            middle_time = time.perf_counter() - start_time
            
            # Stage 2: Walker kernel
            block_size = 32
            grid_size = (num_keys + block_size - 1) // block_size
            config = LaunchConfig(grid=grid_size, block=block_size)
            
            launch(
                self.stream_compute,
                config,
                self.kernel_walker,
                d_basepoints.data.ptr,
                d_found_indices.data.ptr,
                d_found_count.data.ptr,
                cp.uint32(num_keys),
                cp.uint32(steps_per_thread),
                d_head_pattern.data.ptr,
                cp.uint32(head_nibbles),
                d_tail_pattern.data.ptr,
                cp.uint32(tail_nibbles)
            )
            
            # Record end
            end_event.record(self.stream_compute)
            
            # Sync and get results
            self.stream_compute.sync()
            total_time = time.perf_counter() - start_time
            
            found_count = int(d_found_count.get()[0])
            found_indices = d_found_indices[:found_count].get().tolist() if found_count > 0 else []
            
            return found_indices, total_time, middle_time
        
        # Submit async computation
        future = self.executor.submit(compute_async)
        
        return AsyncGpuResult(
            future=future,
            stream=self.stream_compute,
            start_event=start_event,
            end_event=end_event,
            d_found_indices=d_found_indices,
            d_found_count=d_found_count,
            batch_size=num_keys,
            steps_per_thread=steps_per_thread
        )
    
    def prefetch_privkeys_async(self, privkeys: List[bytes]) -> cp.ndarray:
        """
        Asynchronously transfer private keys to GPU memory
        
        Args:
            privkeys: List of 32-byte private keys
            
        Returns:
            GPU array with private keys
        """
        privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
        
        # Use transfer stream for async memory copy
        with cp.cuda.Stream(non_blocking=True) as stream:
            d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
            
        # Record transfer complete
        self.transfer_complete_event.record(self.stream_transfer)
        
        return d_privkeys
    
    def benchmark(self, num_keys: int = 10000, steps_per_thread: int = 256):
        """Run a benchmark test"""
        print(f"\nBenchmarking with {num_keys} keys...")
        
        # Generate random private keys
        privkeys = [os.urandom(32) for _ in range(num_keys)]
        
        # Test simple kernel
        print("\nSimple kernel (1 address per thread):")
        indices, gpu_time, _ = self.generate_vanity_simple(privkeys, head_pattern="0")
        addresses_checked = num_keys
        throughput = addresses_checked / gpu_time / 1e6  # Million addresses per second
        print(f"  GPU time: {gpu_time:.3f} seconds")
        print(f"  Addresses checked: {addresses_checked:,}")
        print(f"  Throughput: {throughput:.2f} MAddr/s")
        print(f"  Found: {len(indices)} matches")
        
        # Test walker kernel
        print(f"\nWalker kernel ({steps_per_thread} addresses per thread):")
        indices, gpu_time, _ = self.generate_vanity_walker(
            privkeys, steps_per_thread=steps_per_thread, 
            head_pattern="0"
        )
        addresses_checked = num_keys * steps_per_thread
        throughput = addresses_checked / gpu_time / 1e6
        print(f"  GPU time: {gpu_time:.3f} seconds")
        print(f"  Addresses checked: {addresses_checked:,}")
        print(f"  Throughput: {throughput:.2f} MAddr/s")
        print(f"  Found: {len(indices)} matches")


def main():
    """Main entry point for testing"""
    # Initialize CUDA vanity generator
    generator = CudaVanity(device_id=0)
    
    # Run benchmark
    generator.benchmark(num_keys=10000, steps_per_thread=256)
    
    # Example: Generate vanity address with specific pattern
    print("\n" + "="*60)
    print("Generating vanity address with pattern 0x888...")
    
    # Use a specific private key or generate random ones
    test_keys = [os.urandom(32) for _ in range(1000)]
    
    # Search for addresses starting with 0x888
    indices, gpu_time, _ = generator.generate_vanity_walker(
        test_keys, 
        steps_per_thread=1000,
        head_pattern="888"
    )
    
    if indices:
        print(f"Found {len(indices)} matching addresses in {gpu_time:.3f} seconds")
        # Convert index to key number and step
        for idx in indices[:5]:  # Show first 5 matches
            key_idx = idx // 1000
            step = idx % 1000
            print(f"  Match at key {key_idx}, step {step}")
    else:
        print(f"No matches found in {gpu_time:.3f} seconds")


if __name__ == "__main__":
    main()