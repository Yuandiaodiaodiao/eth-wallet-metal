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
        
        # Pattern-optimized kernel cache
        self.pattern_kernel_cache = {}
        
        # Kernel source for pattern-optimized compilation
        self.base_kernel_source = None
        
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
        
        # Store base kernel source for pattern-optimized compilation
        self.base_kernel_source = kernel_source
        
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
    
    def _get_nibble_access(self, byte_idx: int, is_high_nibble: bool) -> str:
        """Generate C expression to access a nibble"""
        if is_high_nibble:
            return f"(addr[{byte_idx}] >> 4)"
        else:
            return f"(addr[{byte_idx}] & 0xF)"
    
    def _optimize_literal_checks(self, literal_checks: List[str]) -> List[str]:
        """Optimize literal checks by merging adjacent nibble checks for the same byte"""
        if not literal_checks:
            return literal_checks
        
        # Parse checks to extract byte index, nibble position, and value
        check_info = []
        for check in literal_checks:
            # Parse patterns like "((addr[0] >> 4) == 0x1)" or "((addr[0] & 0xF) == 0x2)"
            import re
            high_match = re.match(r'\(\(addr\[(\d+)\] >> 4\) == 0x([0-9a-fA-F])\)', check)
            low_match = re.match(r'\(\(addr\[(\d+)\] & 0xF\) == 0x([0-9a-fA-F])\)', check)
            
            if high_match:
                byte_idx = int(high_match.group(1))
                value = int(high_match.group(2), 16)
                check_info.append((byte_idx, 'high', value, check))
            elif low_match:
                byte_idx = int(low_match.group(1))
                value = int(low_match.group(2), 16)
                check_info.append((byte_idx, 'low', value, check))
            else:
                # Keep unknown formats as-is
                check_info.append((None, None, None, check))
        
        # Group by byte index and try to merge
        byte_groups = {}
        standalone_checks = []
        
        for byte_idx, nibble_pos, value, original_check in check_info:
            if byte_idx is not None:
                if byte_idx not in byte_groups:
                    byte_groups[byte_idx] = {}
                byte_groups[byte_idx][nibble_pos] = (value, original_check)
            else:
                # Keep non-parseable checks as-is
                standalone_checks.append(original_check)
        
        # Merge checks for bytes that have both high and low nibbles
        optimized_checks = []
        
        for byte_idx in sorted(byte_groups.keys()):
            group = byte_groups[byte_idx]
            
            if 'high' in group and 'low' in group:
                # Both nibbles present - merge into single byte check
                high_val, _ = group['high']
                low_val, _ = group['low']
                merged_val = (high_val << 4) | low_val
                optimized_check = f"(addr[{byte_idx}] == 0x{merged_val:02x})"
                optimized_checks.append(optimized_check)
            else:
                # Only one nibble - keep original checks
                if 'high' in group:
                    _, original_check = group['high']
                    optimized_checks.append(original_check)
                if 'low' in group:
                    _, original_check = group['low']
                    optimized_checks.append(original_check)
        
        # Add standalone checks
        optimized_checks.extend(standalone_checks)
        
        return optimized_checks
    
    def _optimize_wildcard_checks(self, wildcard_positions: List[int]) -> List[str]:
        """Optimize wildcard checks by merging adjacent wildcards for the same byte"""
        if not wildcard_positions:
            return []
        
        # Group wildcard positions by byte and analyze patterns
        byte_wildcards = {}
        for nibble_idx in wildcard_positions:
            byte_idx = nibble_idx // 2
            is_high = (nibble_idx % 2) == 0
            if byte_idx not in byte_wildcards:
                byte_wildcards[byte_idx] = {}
            byte_wildcards[byte_idx][nibble_idx] = is_high
        
        # Find consecutive full-byte wildcards (both high and low nibbles present)
        full_byte_wildcards = []
        partial_nibbles = []
        
        for byte_idx in sorted(byte_wildcards.keys()):
            nibbles_in_byte = byte_wildcards[byte_idx]
            
            # Check if this byte has both high and low nibbles as wildcards
            expected_high = byte_idx * 2
            expected_low = byte_idx * 2 + 1
            
            if expected_high in nibbles_in_byte and expected_low in nibbles_in_byte:
                full_byte_wildcards.append(byte_idx)
            else:
                # Add individual nibbles for partial bytes
                for nibble_idx in nibbles_in_byte:
                    partial_nibbles.append(nibble_idx)
        
        # Generate optimized checks
        optimized_checks = []
        
        if not wildcard_positions:
            return optimized_checks
        
        # Choose reference: prefer partial nibbles first, then full bytes
        if partial_nibbles:
            ref_nibble_idx = partial_nibbles[0]
            ref_byte_idx = ref_nibble_idx // 2
        elif full_byte_wildcards:
            ref_byte_idx = full_byte_wildcards[0] 
            ref_nibble_idx = ref_byte_idx * 2  # Use high nibble as reference
        else:
            return optimized_checks
        
        ref_is_high = (ref_nibble_idx % 2) == 0
        reference_expr = self._get_nibble_access(ref_byte_idx, ref_is_high)
        
        # If reference byte is a full wildcard, ensure its nibbles are equal
        if ref_byte_idx in full_byte_wildcards:
            optimized_checks.append(f"({reference_expr} == (addr[{ref_byte_idx}] & 0xF))")
        
        # Process other full-byte wildcards - optimized approach
        for byte_idx in full_byte_wildcards:
            if byte_idx != ref_byte_idx:
                # For full bytes, we only need to compare the whole byte to reference
                # No need for internal consistency check since equality with reference implies consistency
                optimized_checks.append(f"(addr[{byte_idx}] == addr[{ref_byte_idx}])")
        
        # Process remaining partial wildcards
        for nibble_idx in partial_nibbles:
            if nibble_idx != ref_nibble_idx:  # Skip reference nibble
                byte_idx = nibble_idx // 2
                is_high = (nibble_idx % 2) == 0
                access_expr = self._get_nibble_access(byte_idx, is_high)
                optimized_checks.append(f"({reference_expr} == {access_expr})")
        
        return optimized_checks
    
    def _parse_pattern_segments(self, pattern: str, is_tail: bool = False):
        """Parse pattern into segments of wildcards and literals
        
        Returns:
            List of tuples (type, value, global_nibble_positions)
            where type is 'wildcard' or 'literal'
        """
        if not pattern:
            return []
        
        segments = []
        current_segment = {'type': None, 'chars': [], 'start_pos': None}
        
        for i, char in enumerate(pattern):
            if char == '*':
                if current_segment['type'] == 'literal':
                    # End current literal segment
                    segments.append(('literal', ''.join(current_segment['chars']), current_segment['start_pos']))
                    current_segment = {'type': 'wildcard', 'chars': [char], 'start_pos': i}
                else:
                    # Continue or start wildcard segment
                    current_segment['type'] = 'wildcard'
                    current_segment['chars'].append(char)
                    if 'start_pos' not in current_segment or current_segment['start_pos'] is None:
                        current_segment['start_pos'] = i
            else:
                if current_segment['type'] == 'wildcard':
                    # End current wildcard segment
                    segments.append(('wildcard', ''.join(current_segment['chars']), current_segment['start_pos']))
                    current_segment = {'type': 'literal', 'chars': [char], 'start_pos': i}
                else:
                    # Continue or start literal segment
                    current_segment['type'] = 'literal'
                    current_segment['chars'].append(char)
                    if 'start_pos' not in current_segment or current_segment['start_pos'] is None:
                        current_segment['start_pos'] = i
        
        # Add final segment
        if current_segment['chars']:
            segments.append((current_segment['type'], ''.join(current_segment['chars']), current_segment['start_pos']))
        
        # Convert to global nibble positions
        result = []
        for seg_type, value, start_pos in segments:
            if is_tail:
                # For tail pattern, calculate positions from the end
                # Tail patterns are positioned from the right side of the address
                global_positions = []
                for i in range(len(value)):
                    nibble_from_end = len(pattern) - start_pos - i - 1
                    global_nibble_idx = 40 - 1 - nibble_from_end  # 40 nibbles total (20 bytes * 2)
                    
                    # Safety check to prevent out-of-bounds access
                    if global_nibble_idx < 0 or global_nibble_idx >= 40:
                        raise ValueError(f"Invalid global nibble index {global_nibble_idx} for pattern '{pattern}' "
                                       f"at segment start_pos={start_pos}, char {i}. "
                                       f"nibble_from_end={nibble_from_end}")
                    
                    global_positions.append(global_nibble_idx)
            else:
                # For head pattern, positions start from 0
                global_positions = list(range(start_pos, start_pos + len(value)))
            
            result.append((seg_type, value, global_positions))
        
        return result
    
    def _generate_pattern_macros(self, head_pattern: str, tail_pattern: str) -> str:
        """Generate compile-time macros for pattern matching with wildcard support"""
        macros = []
        
        # Parse both patterns to identify wildcards and literals
        head_segments = self._parse_pattern_segments(head_pattern, is_tail=False)
        tail_segments = self._parse_pattern_segments(tail_pattern, is_tail=True)
        
        # Collect all wildcard positions globally
        wildcard_positions = []
        literal_checks = []
        
        # Process head pattern segments
        for seg_type, value, positions in head_segments:
            if seg_type == 'wildcard':
                wildcard_positions.extend(positions)
            else:
                # Generate literal checks for head
                for i, char in enumerate(value):
                    nibble_idx = positions[i]
                    byte_idx = nibble_idx // 2
                    is_high = (nibble_idx % 2) == 0
                    nibble_val = int(char, 16)
                    
                    access_expr = self._get_nibble_access(byte_idx, is_high)
                    literal_checks.append(f"({access_expr} == 0x{nibble_val:x})")
        
        # Process tail pattern segments
        for seg_type, value, positions in tail_segments:
            if seg_type == 'wildcard':
                wildcard_positions.extend(positions)
            else:
                # Generate literal checks for tail
                for i, char in enumerate(value):
                    nibble_idx = positions[i]
                    byte_idx = nibble_idx // 2
                    is_high = (nibble_idx % 2) == 0
                    nibble_val = int(char, 16)
                    
                    access_expr = self._get_nibble_access(byte_idx, is_high)
                    literal_checks.append(f"({access_expr} == 0x{nibble_val:x})")
        
        # Generate optimized wildcard matching checks
        wildcard_checks = self._optimize_wildcard_checks(wildcard_positions)
        
        # Optimize literal checks by merging adjacent nibble checks
        optimized_literal_checks = self._optimize_literal_checks(literal_checks)
        
        # Combine all checks
        all_checks = wildcard_checks + optimized_literal_checks
        
        if all_checks:
            # Split checks between head and tail macros for clarity
            # For now, put all checks in head macro and make tail macro always true
            head_macro = f"#define CHECK_HEAD_PATTERN(addr) ({' && '.join(all_checks)})"
            tail_macro = "#define CHECK_TAIL_PATTERN(addr) (true)"
        else:
            # No patterns specified
            head_macro = "#define CHECK_HEAD_PATTERN(addr) (true)"
            tail_macro = "#define CHECK_TAIL_PATTERN(addr) (true)"
        
        macros.append(head_macro)
        macros.append(tail_macro)
        
        return '\n'.join(macros)
    
    def _get_pattern_kernel(self, head_pattern: str, tail_pattern: str):
        """Get or compile pattern-optimized kernel"""
        pattern_key = (head_pattern, tail_pattern)
        
        # Check cache
        if pattern_key in self.pattern_kernel_cache:
            return self.pattern_kernel_cache[pattern_key]
        
        # Generate pattern-specific macros
        pattern_macros = self._generate_pattern_macros(head_pattern, tail_pattern)
        
        # Create modified kernel source with pattern macros
        # Insert macros before the includes
        lines = self.base_kernel_source.split('\n')
        
        # Find the position after includes but before kernel definitions
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_pos = i + 1
            elif line.strip().startswith('//') and 'CUDA Kernel Functions' in line:
                break
        
        # Insert our optimization macros after includes
        optimized_lines = (
            lines[:insert_pos] + 
            ['', '// Pattern optimization macros'] +
            pattern_macros.split('\n') +
            ['#define USE_PATTERN_OPTIMIZATION', '', 
             '// Ultra-fast pattern check with compile-time patterns',
             '__device__ __forceinline__ bool check_vanity_pattern_optimized(const uint8_t* addr20) {',
             '    return CHECK_HEAD_PATTERN(addr20) && CHECK_TAIL_PATTERN(addr20);',
             '}', ''] +
            lines[insert_pos:]
        )
        
        optimized_source = '\n'.join(optimized_lines)
        
        # Compile options
        kernel_dir = Path(__file__).parent
        cuda_include_path = os.path.join(os.environ.get("CUDA_PATH", ""), "include")
        include_paths = [str(kernel_dir)]
        if os.path.exists(cuda_include_path):
            include_paths.append(cuda_include_path)
        
        program_options = ProgramOptions(
            arch=f"sm_{self.arch}",
            include_path=include_paths,
            use_fast_math=True
        )
        
        # Create and compile program
        program = Program(optimized_source, code_type="c++", options=program_options)
        
        kernel_names = [
            "vanity_kernel_g16",
            "compute_basepoints_g16", 
            "vanity_walker_kernel",
            "build_g16_table_kernel"
        ]
        
        module = program.compile("cubin", name_expressions=kernel_names)
        
        # Get optimized kernel references
        optimized_kernels = {
            'vanity_g16': module.get_kernel("vanity_kernel_g16"),
            'compute_base': module.get_kernel("compute_basepoints_g16"),
            'walker': module.get_kernel("vanity_walker_kernel"),
            'build_g16': module.get_kernel("build_g16_table_kernel")
        }
        
        # Cache the compiled kernels
        self.pattern_kernel_cache[pattern_key] = optimized_kernels
        
        print(f"Compiled optimized kernel for pattern: head='{head_pattern}', tail='{tail_pattern}'")
        
        # Debug: print generated macros
        print("Generated macros:")
        for line in pattern_macros.split('\n'):
            print(f"  {line}")
        
        # Debug: save optimized source to file for inspection
        # Escape special characters in pattern strings for safe filename
        safe_head = head_pattern.replace('*', 'STAR').replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_') if head_pattern else 'EMPTY'
        safe_tail = tail_pattern.replace('*', 'STAR').replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_') if tail_pattern else 'EMPTY'
        debug_path = kernel_dir / f"debug_optimized_{safe_head}_{safe_tail}.cu"
        with open(debug_path, "w") as f:
            f.write(optimized_source)
        print(f"Debug: Saved optimized source to {debug_path}")
        
        return optimized_kernels
    
    def _pattern_to_bytes(self, pattern: str) -> Tuple[cp.ndarray, int]:
        """Convert hex pattern string to byte array for GPU processing
        
        Args:
            pattern: Hex pattern string (may contain wildcards '*')
            
        Returns:
            Tuple of (pattern_bytes, nibble_count)
        """
        if not pattern:
            return cp.zeros(0, dtype=cp.uint8), 0
        
        # Check if pattern contains wildcards
        if '*' in pattern:
            # For wildcard patterns, we use optimized kernels that don't need pattern bytes
            return cp.zeros(1, dtype=cp.uint8), 0  # Dummy values
        
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
        
        # Convert pattern strings to byte arrays (only for non-wildcard patterns)
        d_head_pattern, head_nibbles = self._pattern_to_bytes(head_pattern)
        d_tail_pattern, tail_nibbles = self._pattern_to_bytes(tail_pattern)
        
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
        
        # Get pattern-optimized kernels
        if head_pattern or tail_pattern:
            optimized_kernels = self._get_pattern_kernel(head_pattern, tail_pattern)
            kernel_compute_base = optimized_kernels['compute_base']
            kernel_walker = optimized_kernels['walker']
            print(f"Using optimized kernel for head='{head_pattern}', tail='{tail_pattern}'")
        else:
            # Use default kernels for no pattern
            kernel_compute_base = self.kernel_compute_base
            kernel_walker = self.kernel_walker
        
        # Convert pattern strings to byte arrays (only for non-wildcard patterns)
        d_head_pattern, head_nibbles = self._pattern_to_bytes(head_pattern)
        d_tail_pattern, tail_nibbles = self._pattern_to_bytes(tail_pattern)
        
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
            kernel_compute_base,
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
        # Use optimized kernel without pattern parameters if optimization is enabled
        if head_pattern or tail_pattern:
            # Optimized kernel doesn't need pattern parameters
            launch(
                self.stream,
                config,
                kernel_walker,
                d_basepoints.data.ptr,      # basepoints
                d_found_indices.data.ptr,   # found_indices
                d_found_count.data.ptr,     # found_count
                cp.uint32(num_keys),        # num_keys
                cp.uint32(steps_per_thread), # steps_per_thread
            )
        else:
            # Standard kernel with pattern parameters
            launch(
                self.stream,
                config,
                kernel_walker,
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
    

    
  

