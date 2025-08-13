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
                              target_nibble: int = 0x8,
                              nibble_count: int = 7) -> Tuple[List[int], float]:
        """
        Generate vanity addresses using simple kernel (one address per thread)
        
        Args:
            privkeys: List of 32-byte private keys (big-endian)
            target_nibble: Target nibble value (0x0-0xF)
            nibble_count: Number of nibbles to match
            
        Returns:
            Tuple of (found_indices, gpu_time_seconds)
        """
        
        num_keys = len(privkeys)
        
        # Prepare input data
        privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
        d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
        
        # Allocate output buffers
        max_found = min(1, num_keys)  # Limit output size
        d_found_indices = cp.zeros(max_found, dtype=cp.uint32)
        d_found_count = cp.zeros(1, dtype=cp.uint32)
        
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
            self.g16_table.data.ptr,    # g16_table
            cp.uint32(num_keys),        # num_keys
            cp.uint8(target_nibble),    # target_nibble
            cp.uint32(nibble_count)     # nibble_count
        )
        
        # Wait for completion
        self.stream.sync()
        gpu_time = time.perf_counter() - start_time
        
        # Get results
        found_count = int(d_found_count.get()[0])
        found_indices = d_found_indices[:found_count].get().tolist() if found_count > 0 else []
        
        return found_indices, gpu_time
    
    def generate_vanity_walker(self,
                               privkeys: List[bytes],
                               steps_per_thread: int = 256,
                               target_nibble: int = 0x8,
                               nibble_count: int = 7) -> Tuple[List[int], float]:
        """
        Generate vanity addresses using walker kernel (multiple addresses per thread)
        
        Args:
            privkeys: List of 32-byte private keys (big-endian)
            steps_per_thread: Number of addresses to check per thread
            target_nibble: Target nibble value (0x0-0xF)
            nibble_count: Number of nibbles to match
            
        Returns:
            Tuple of (found_indices, gpu_time_seconds)
        """
        
        
             
        num_keys = len(privkeys)
        
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
        block_size = 16
        grid_size = (num_keys + block_size - 1) // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        
        launch(
            self.stream,
            config,
            self.kernel_compute_base,
            d_privkeys.data.ptr,        # privkeys
            d_basepoints.data.ptr,      # basepoints
            self.g16_table.data.ptr,    # g16_table
            cp.uint32(num_keys)         # num_keys
        )
        
        # Stage 2: Walker kernel
        # Use smaller block size for walker due to higher register usage
        block_size = 256
        grid_size = (num_keys + block_size - 1) // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        
        launch(
            self.stream,
            config,
            self.kernel_walker,
            d_basepoints.data.ptr,      # basepoints
            d_found_indices.data.ptr,   # found_indices
            d_found_count.data.ptr,     # found_count
            cp.uint32(num_keys),        # num_keys
            cp.uint32(steps_per_thread), # steps_per_thread
            cp.uint8(target_nibble),    # target_nibble
            cp.uint32(nibble_count)     # nibble_count
        )
        
        # Wait for completion
        self.stream.sync()
        gpu_time = time.perf_counter() - start_time
        
        # Get results
        found_count = int(d_found_count.get()[0])
        found_indices = d_found_indices[:found_count].get().tolist() if found_count > 0 else []
        
        return found_indices, gpu_time
    
    def benchmark(self, num_keys: int = 10000, steps_per_thread: int = 256):
        """Run a benchmark test"""
        print(f"\nBenchmarking with {num_keys} keys...")
        
        # Generate random private keys
        privkeys = [os.urandom(32) for _ in range(num_keys)]
        
        # Test simple kernel
        print("\nSimple kernel (1 address per thread):")
        indices, gpu_time = self.generate_vanity_simple(privkeys, target_nibble=0x0, nibble_count=1)
        addresses_checked = num_keys
        throughput = addresses_checked / gpu_time / 1e6  # Million addresses per second
        print(f"  GPU time: {gpu_time:.3f} seconds")
        print(f"  Addresses checked: {addresses_checked:,}")
        print(f"  Throughput: {throughput:.2f} MAddr/s")
        print(f"  Found: {len(indices)} matches")
        
        # Test walker kernel
        print(f"\nWalker kernel ({steps_per_thread} addresses per thread):")
        indices, gpu_time = self.generate_vanity_walker(
            privkeys, steps_per_thread=steps_per_thread, 
            target_nibble=0x0, nibble_count=1
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
    indices, gpu_time = generator.generate_vanity_walker(
        test_keys, 
        steps_per_thread=1000,
        target_nibble=0x8,
        nibble_count=3
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