"""
CUDA-based Ethereum vanity address generator using cuda-python
Following best practices from cuda-python examples
"""

import os
import time
import numpy as np
from typing import List, Tuple
from pathlib import Path

# Use cuda.core for modern CUDA programming
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch
import cupy as cp
from dataclasses import dataclass
from enum import Enum
from collections import deque
import threading

class TaskState(Enum):
    READY = "ready"           # 任务准备就绪
    COMPUTE_LOADING = "compute_loading"  # compute_base正在数据加载和排队
    COMPUTE_RUNNING = "compute_running"  # compute_base kernel运行中
    COMPUTE_DONE = "compute_done"        # compute_base完成，等待walker
    WALKER_RUNNING = "walker_running"    # walker kernel运行中
    COMPLETED = "completed"              # 任务完成
    FAILED = "failed"                    # 任务失败

@dataclass
class GPUTask:
    """GPU任务数据结构"""
    task_id: int
    privkeys_data: cp.ndarray
    basepoints: cp.ndarray
    found_indices: cp.ndarray
    found_count: cp.ndarray
    num_keys: int
    steps_per_thread: int
    state: TaskState = TaskState.READY
    compute_event: object = None
    walker_event: object = None
    start_time: float = 0.0
    compute_done_time: float = 0.0
    walker_done_time: float = 0.0

class GPUTaskQueue:
    """GPU任务队列管理器，实现三缓冲pipeline"""
    
    def __init__(self, max_concurrent_tasks: int = 2):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks = deque()  # 任务队列
        self.active_tasks = {}  # 正在执行的任务 {task_id: task}
        self.completed_tasks = deque(maxlen=10)  # 已完成任务历史
        self.next_task_id = 0
        self.lock = threading.Lock()
        
        # 全局compute_base执行锁 - 确保只有一个compute_base在数据加载/排队
        self.compute_global_lock = threading.Lock()
        self.compute_loading_task_id = None  # 当前正在加载的任务ID
        
    def create_task(self, privkeys_data: cp.ndarray, num_keys: int, steps_per_thread: int, device) -> GPUTask:
        """创建新的GPU任务"""
        with self.lock:
            task_id = self.next_task_id
            self.next_task_id += 1
            
            # 分配GPU内存buffers
            basepoints = cp.zeros(num_keys * 16, dtype=cp.uint32)
            max_found = min(1, num_keys * steps_per_thread)
            found_indices = cp.zeros(max_found, dtype=cp.uint32)
            found_count = cp.zeros(1, dtype=cp.uint32)
            
            # 创建同步事件
            compute_event = device.create_event()
            walker_event = device.create_event()
            
            task = GPUTask(
                task_id=task_id,
                privkeys_data=privkeys_data,
                basepoints=basepoints,
                found_indices=found_indices,
                found_count=found_count,
                num_keys=num_keys,
                steps_per_thread=steps_per_thread,
                compute_event=compute_event,
                walker_event=walker_event,
                start_time=time.perf_counter()
            )
            
            self.tasks.append(task)
            return task
    
    def get_ready_tasks(self) -> List[GPUTask]:
        """获取可以开始compute的任务"""
        with self.lock:
            ready_tasks = []
            for task in list(self.tasks):
                if task.state == TaskState.READY and len(self.active_tasks) < self.max_concurrent_tasks:
                    ready_tasks.append(task)
                    self.tasks.remove(task)
                    self.active_tasks[task.task_id] = task
                    # 注意：这里不直接设置为COMPUTE_RUNNING，而是在获取锁后设置为COMPUTE_LOADING
            return ready_tasks
    
    def try_acquire_compute_lock(self, task_id: int) -> bool:
        """尝试获取compute_base执行锁"""
        if self.compute_global_lock.acquire(blocking=False):
            with self.lock:
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    task.state = TaskState.COMPUTE_LOADING
                    self.compute_loading_task_id = task_id
                    return True
                else:
                    # 任务不存在，释放锁
                    self.compute_global_lock.release()
                    return False
        return False
    
    def release_compute_lock(self, task_id: int):
        """释放compute_base执行锁"""
        with self.lock:
            if self.compute_loading_task_id == task_id:
                self.compute_loading_task_id = None
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    task.state = TaskState.COMPUTE_RUNNING
        self.compute_global_lock.release()
        
    def get_loading_tasks(self) -> List[GPUTask]:
        """获取可以开始实际执行compute kernel的任务（已获得锁的）"""
        with self.lock:
            loading_tasks = []
            for task in self.active_tasks.values():
                if task.state == TaskState.COMPUTE_LOADING:
                    loading_tasks.append(task)
            return loading_tasks
    
    def get_compute_done_tasks(self) -> List[GPUTask]:
        """获取compute完成，可以开始walker的任务"""
        with self.lock:
            compute_done_tasks = []
            for task in self.active_tasks.values():
                if task.state == TaskState.COMPUTE_DONE:
                    compute_done_tasks.append(task)
                    task.state = TaskState.WALKER_RUNNING
            return compute_done_tasks
    
    def mark_compute_done(self, task_id: int):
        """标记compute阶段完成"""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.state = TaskState.COMPUTE_DONE
                task.compute_done_time = time.perf_counter()
    
    def mark_walker_done(self, task_id: int):
        """标记walker阶段完成"""
        with self.lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.state = TaskState.COMPLETED
                task.walker_done_time = time.perf_counter()
    
    def get_completed_tasks(self) -> List[GPUTask]:
        """获取已完成的任务"""
        with self.lock:
            completed = []
            for task_id in list(self.active_tasks.keys()):
                task = self.active_tasks[task_id]
                if task.state == TaskState.COMPLETED:
                    completed.append(task)
                    del self.active_tasks[task_id]
                    self.completed_tasks.append(task)
            return completed
    
    def get_stats(self) -> dict:
        """获取队列统计信息"""
        with self.lock:
            state_counts = {}
            for task in self.active_tasks.values():
                state = task.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
            
            return {
                'queued_tasks': len(self.tasks),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.completed_tasks),
                'active_states': state_counts,
                'compute_loading_task': self.compute_loading_task_id
            }

class CudaVanity:
    """CUDA-accelerated vanity address generator"""
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA device and compile kernels"""
        # Set up device
        self.device = Device(device_id)
        self.device.set_current()
        
        # Create streams for pipeline operations
        self.stream = self.device.create_stream()
        self.stream_g16 = self.device.create_stream()  # Separate stream for G16 table operations
        self.stream_compute = self.device.create_stream()  # For compute_base kernels
        self.stream_walker = self.device.create_stream()   # For walker kernels
        
        # Create events for synchronization
        self.event_compute_done = self.device.create_event()
        self.event_walker_done = self.device.create_event()
        
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
        self._load_kernels()
        # Load and compile kernels
        self._load_g16_kernel()
        
        # G16 precomputed table
        self.g16_table = None
        self.g16_table_size = 16 * 65536 * 64  # 16 windows * 65536 entries * 64 bytes
        
        # Load or build G16 table
        self._init_g16_table()
        
        # Pipeline state management
        self.task_queue = None
        self._init_task_queue()
        
    def _load_kernels(self):
        """Load and compile CUDA kernels"""
        kernel_dir = Path(__file__).parent
        
        # Read main kernel source files
        with open(kernel_dir / "kernels.cu", "r") as f:
            kernel_source = f.read()
        
        # Store base kernel source for pattern-optimized compilation
        self.base_kernel_source = kernel_source
        
      
        
        # Separately load G16 table kernel
    
    def _load_g16_kernel(self):
        """Load and compile G16 table builder kernel separately"""
        kernel_dir = Path(__file__).parent
        
        # Read G16 kernel source
        with open(kernel_dir / "g16_table.cu", "r") as f:
            g16_source = f.read()
        
        # Compile options
        cuda_include_path = os.path.join(os.environ.get("CUDA_PATH", ""), "include")
        include_paths = [str(kernel_dir)]
        if os.path.exists(cuda_include_path):
            include_paths.append(cuda_include_path)
        
        program_options = ProgramOptions(
            arch=f"sm_{self.arch}",
            include_path=include_paths,
            use_fast_math=True
        )
        
        # Create and compile G16 program
        self.g16_program = Program(g16_source, code_type="c++", options=program_options)
        self.g16_module = self.g16_program.compile("cubin", name_expressions=["build_g16_table_kernel"])
        
        # Get G16 kernel reference
        self.kernel_build_g16 = self.g16_module.get_kernel("build_g16_table_kernel")
        
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
                # For full bytes, we need both:
                # 1. Internal consistency check (high nibble == low nibble)
                # 2. Equality with reference byte
                optimized_checks.append(f"((addr[{byte_idx}] >> 4) == (addr[{byte_idx}] & 0xF))")
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
           # Escape special characters in pattern strings for safe filename
        safe_head = head_pattern.replace('*', 'STAR').replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_') if head_pattern else 'EMPTY'
        safe_tail = tail_pattern.replace('*', 'STAR').replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_') if tail_pattern else 'EMPTY'
        kernel_dir = Path(__file__).parent
        debug_path = kernel_dir / f"debug_optimized_{safe_head}_{safe_tail}.cu"
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
        cuda_include_path = os.path.join(os.environ.get("CUDA_PATH", ""), "include")
        include_paths = [str(kernel_dir)]
        if os.path.exists(cuda_include_path):
            include_paths.append(cuda_include_path)
        
        program_options = ProgramOptions(
            arch=f"sm_{self.arch}",
            include_path=include_paths,
            use_fast_math=True,
            split_compile=8,
            device_int128=True,
            ptxas_options=["-O3"],
            device_code_optimize=True,
            extra_device_vectorization=True,
        )
    
        with open(debug_path, "w") as f:
            f.write(optimized_source)
            print(f"Debug: Saved optimized source to {debug_path}")
        # Create and compile program
        program = Program(optimized_source, code_type="c++", options=program_options)
        
        kernel_names = [
            "vanity_kernel_g16",
            "compute_basepoints_g16", 
            "vanity_walker_kernel"
        ]
        
        module = program.compile("cubin", name_expressions=kernel_names)
        
        # Get optimized kernel references
        optimized_kernels = {
            'vanity_g16': module.get_kernel("vanity_kernel_g16"),
            'compute_base': module.get_kernel("compute_basepoints_g16"),
            'walker': module.get_kernel("vanity_walker_kernel")
        }
        
        # Cache the compiled kernels
        self.pattern_kernel_cache[pattern_key] = optimized_kernels
        
        print(f"Compiled optimized kernel for pattern: head='{head_pattern}', tail='{tail_pattern}'")
        
        # Debug: print generated macros
        print("Generated macros:")
        for line in pattern_macros.split('\n'):
            print(f"  {line}")
        
        # Debug: save optimized source to file for inspection
     
       
        
        return optimized_kernels
    
    def _init_task_queue(self):
        """Initialize the GPU task queue for pipeline execution"""
        self.task_queue = GPUTaskQueue()
    
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
        
        # Always get pattern-optimized kernels
        optimized_kernels = self._get_pattern_kernel(head_pattern, tail_pattern)
        kernel_vanity_g16 = optimized_kernels['vanity_g16']
        print(f"Using optimized kernel for head='{head_pattern}', tail='{tail_pattern}'")
        
        # Pattern optimization mode doesn't need pattern byte arrays
        # (patterns are compiled into the kernel as macros)
        
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
            kernel_vanity_g16,
            d_privkeys.data.ptr,        # privkeys
            d_found_indices.data.ptr,   # found_indices
            d_found_count.data.ptr,     # found_count
            d_outbuffer.data.ptr,       # outbuffer
            self.g16_table.data.ptr,    # g16_table
            cp.uint32(num_keys)         # num_keys
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
        
        # Always get pattern-optimized kernels
        optimized_kernels = self._get_pattern_kernel(head_pattern, tail_pattern)
        kernel_compute_base = optimized_kernels['compute_base']
        kernel_walker = optimized_kernels['walker']
        print(f"Using optimized kernel for head='{head_pattern}', tail='{tail_pattern}'")
        
        # Pattern optimization mode doesn't need pattern byte arrays
        # (patterns are compiled into the kernel as macros)
        
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
        block_size = 128
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
        block_size = 128
        grid_size = num_keys // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        print(f'launch walker {grid_size} {block_size}')
        # Always use optimized kernel (no pattern parameters needed)
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
        print('sync')
        # Wait for completion
        self.stream.sync()
        print('sync done')
        gpu_time = time.perf_counter() - start_time
        
        # Get results
        found_count = int(d_found_count.get()[0])
        found_indices = d_found_indices[:found_count].get().tolist() if found_count > 0 else []
        
        return found_indices, gpu_time, middle_gpu_time
    
    def generate_vanity_walker_pipeline(self,
                                        initial_privkeys_batch: List[List[bytes]],
                                        steps_per_thread: int = 256,
                                        head_pattern: str = "",
                                        tail_pattern: str = "",
                                        max_iterations: int = 1000) -> Tuple[List[int], float, dict]:
        """
        Pipeline版本的vanity生成器，实现GPU满载运行
        
        Args:
            initial_privkeys_batch: 初始私钥批次列表
            steps_per_thread: 每个线程检查的地址数
            head_pattern: 地址前缀模式
            tail_pattern: 地址后缀模式  
            max_iterations: 最大迭代次数
            
        Returns:
            Tuple of (found_indices, total_time_seconds, performance_stats)
        """
        print(f"Starting pipeline walker with {len(initial_privkeys_batch)} batches, pattern: '{head_pattern}', '{tail_pattern}'")
        
        # 获取优化内核
        optimized_kernels = self._get_pattern_kernel(head_pattern, tail_pattern)
        kernel_compute_base = optimized_kernels['compute_base']
        kernel_walker = optimized_kernels['walker']
        
        # 重置任务队列
        self.task_queue = GPUTaskQueue(max_concurrent_tasks=2)
        
        # 初始化性能统计
        stats = {
            'total_keys_processed': 0,
            'compute_launches': 0,
            'walker_launches': 0,
            'pipeline_efficiency': 0.0,
            'avg_gpu_utilization': 0.0,
            'total_gpu_time': 0.0
        }
        
        start_time = time.perf_counter()
        iteration = 0
        
        # 初始化前两个任务
        for i, privkeys in enumerate(initial_privkeys_batch[:2]):
            privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
            d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
            task = self.task_queue.create_task(d_privkeys, len(privkeys), steps_per_thread, self.device)
            print(f"Created initial task {task.task_id} with {len(privkeys)} keys")
        
        batch_index = 2  # 下一个要处理的批次索引
        
        while iteration < max_iterations:
            iteration += 1
            
            # 1. 检查准备就绪的任务并尝试获取compute锁
            ready_tasks = self.task_queue.get_ready_tasks()
            for task in ready_tasks:
                if self.task_queue.try_acquire_compute_lock(task.task_id):
                    print(f"Task {task.task_id} acquired compute lock")
                else:
                    print(f"Task {task.task_id} waiting for compute lock")
            
            # 2. 启动已获得锁的compute任务
            loading_tasks = self.task_queue.get_loading_tasks()
            for task in loading_tasks:
                self._launch_compute_base(task, kernel_compute_base)
                stats['compute_launches'] += 1
                print(f"Launched compute_base for task {task.task_id}")
            
            # 3. 启动compute完成的walker任务
            compute_done_tasks = self.task_queue.get_compute_done_tasks()
            for task in compute_done_tasks:
                self._launch_walker(task, kernel_walker)
                stats['walker_launches'] += 1
                print(f"Launched walker for task {task.task_id}")
            
            # 4. 检查完成的任务
            completed_tasks = self.task_queue.get_completed_tasks()
            for task in completed_tasks:
                found_count = int(task.found_count.get()[0])
                if found_count > 0:
                    found_indices = task.found_indices[:found_count].get().tolist()
                    total_time = time.perf_counter() - start_time
                    stats['total_keys_processed'] += task.num_keys * task.steps_per_thread
                    print(f"Found match in task {task.task_id}: indices {found_indices}")
                    return found_indices, total_time, stats
                
                stats['total_keys_processed'] += task.num_keys * task.steps_per_thread
                print(f"Task {task.task_id} completed, no matches found")
                
                # 如果还有数据，创建新任务替换完成的任务
                if batch_index < len(initial_privkeys_batch):
                    privkeys = initial_privkeys_batch[batch_index]
                    privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
                    d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
                    new_task = self.task_queue.create_task(d_privkeys, len(privkeys), steps_per_thread, self.device)
                    batch_index += 1
                    print(f"Created new task {new_task.task_id} to replace completed task")
            
            # 5. 检查事件状态更新
            self._update_task_states()
            
            # 6. 简单的CPU睡眠避免忙等待
            time.sleep(0.001)  # 1ms
            
            # 每100次迭代输出统计
            if iteration % 100 == 0:
                queue_stats = self.task_queue.get_stats()
                print(f"Iteration {iteration}: {queue_stats}")
        
        # 超过最大迭代次数，返回统计
        total_time = time.perf_counter() - start_time
        return [], total_time, stats
    
    def _launch_compute_base(self, task: GPUTask, kernel_compute_base):
        """启动compute_base kernel"""
        block_size = 128
        grid_size = (task.num_keys + block_size - 1) // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        
        launch(
            self.stream_compute,
            config,
            kernel_compute_base,
            task.privkeys_data.data.ptr,    # privkeys
            task.basepoints.data.ptr,       # basepoints
            self.g16_table.data.ptr,        # g16_table
            cp.uint32(task.num_keys)        # num_keys
        )
        
        # 记录事件标记compute完成
        self.stream_compute.record(task.compute_event)
        
        # 释放compute锁，允许其他任务获取锁
        self.task_queue.release_compute_lock(task.task_id)
    
    def _launch_walker(self, task: GPUTask, kernel_walker):
        """启动walker kernel"""
        block_size = 128
        grid_size = task.num_keys // block_size
        config = LaunchConfig(grid=grid_size, block=block_size)
        
        launch(
            self.stream_walker,
            config,
            kernel_walker,
            task.basepoints.data.ptr,       # basepoints
            task.found_indices.data.ptr,    # found_indices
            task.found_count.data.ptr,      # found_count
            cp.uint32(task.num_keys),       # num_keys
            cp.uint32(task.steps_per_thread) # steps_per_thread
        )
        
        # 记录事件标记walker完成
        self.stream_walker.record(task.walker_event)
    
    def _update_task_states(self):
        """更新任务状态基于事件完成情况"""
        for task in self.task_queue.active_tasks.values():
            if task.state == TaskState.COMPUTE_RUNNING and task.compute_event.is_done:
                self.task_queue.mark_compute_done(task.task_id)
            elif task.state == TaskState.WALKER_RUNNING and task.walker_event.is_done:
                self.task_queue.mark_walker_done(task.task_id)
    
    def get_gpu_utilization_stats(self) -> dict:
        """获取GPU利用率统计信息"""
        queue_stats = self.task_queue.get_stats()
        
        # 计算各个阶段的任务数
        active_states = queue_stats.get('active_states', {})
        compute_loading = active_states.get('compute_loading', 0)
        compute_running = active_states.get('compute_running', 0)
        walker_running = active_states.get('walker_running', 0) 
        total_active = queue_stats['active_tasks']
        
        # GPU利用率估算（简化版）
        utilization = 0.0
        if total_active > 0:
            # compute_loading不占用GPU，只有running状态才占用
            utilization = (compute_running + walker_running) / max(1, total_active) * 100
        
        return {
            'gpu_utilization_percent': utilization,
            'compute_kernels_loading': compute_loading,
            'compute_kernels_running': compute_running,
            'walker_kernels_running': walker_running,
            'total_active_tasks': total_active,
            'queued_tasks': queue_stats['queued_tasks'],
            'compute_loading_task': queue_stats.get('compute_loading_task'),
            'pipeline_efficiency': min(100.0, total_active / 2.0 * 100)  # 2个并发任务为100%
        }
    
    def create_continuous_pipeline(self, 
                                   privkey_generator,
                                   batch_size: int = 4096,
                                   steps_per_thread: int = 256,
                                   head_pattern: str = "",
                                   tail_pattern: str = ""):
        """
        创建连续的pipeline执行器，可以与外部私钥生成器集成
        
        Args:
            privkey_generator: 私钥生成器函数，返回List[bytes]
            batch_size: 每批次私钥数量
            steps_per_thread: 每个线程检查的地址数  
            head_pattern: 地址前缀模式
            tail_pattern: 地址后缀模式
            
        Yields:
            生成器，产出(found_indices, task_stats)或None
        """
        print(f"Starting continuous pipeline: batch_size={batch_size}, steps_per_thread={steps_per_thread}")
        
        # 获取优化内核
        optimized_kernels = self._get_pattern_kernel(head_pattern, tail_pattern)
        kernel_compute_base = optimized_kernels['compute_base']
        kernel_walker = optimized_kernels['walker']
        
        # 重置任务队列
        self.task_queue = GPUTaskQueue(max_concurrent_tasks=2)
        
        # 初始化两个任务
        for i in range(2):
            privkeys = privkey_generator(batch_size)
            privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in privkeys])
            d_privkeys = cp.asarray(privkeys_data, dtype=cp.uint8)
            task = self.task_queue.create_task(d_privkeys, len(privkeys), steps_per_thread, self.device)
            print(f"Created initial pipeline task {task.task_id}")
        
        iteration = 0
        
        while True:
            iteration += 1
            
            # Pipeline执行步骤
            # 1. 检查准备就绪的任务并尝试获取compute锁
            ready_tasks = self.task_queue.get_ready_tasks()
            for task in ready_tasks:
                self.task_queue.try_acquire_compute_lock(task.task_id)
            
            # 2. 启动已获得锁的compute任务
            loading_tasks = self.task_queue.get_loading_tasks()
            for task in loading_tasks:
                self._launch_compute_base(task, kernel_compute_base)
            
            # 3. 启动compute完成的walker任务
            compute_done_tasks = self.task_queue.get_compute_done_tasks()
            for task in compute_done_tasks:
                self._launch_walker(task, kernel_walker)
            
            completed_tasks = self.task_queue.get_completed_tasks()
            for task in completed_tasks:
                found_count = int(task.found_count.get()[0])
                
                # 计算任务统计
                task_stats = {
                    'task_id': task.task_id,
                    'keys_processed': task.num_keys * task.steps_per_thread,
                    'total_time': task.walker_done_time - task.start_time,
                    'compute_time': task.compute_done_time - task.start_time,
                    'walker_time': task.walker_done_time - task.compute_done_time
                }
                
                if found_count > 0:
                    found_indices = task.found_indices[:found_count].get().tolist()
                    # 添加私钥恢复信息
                    match_info = {
                        'indices': found_indices,
                        'task_id': task.task_id,
                        'batch_privkeys': task.privkeys_data.get(),  # 完整的私钥数据
                        'num_keys': task.num_keys,
                        'steps_per_thread': task.steps_per_thread
                    }
                    yield match_info, task_stats
                    return  # 找到结果后停止
                else:
                    yield None, task_stats
                
                # 创建新任务替换完成的任务
                new_privkeys = privkey_generator(batch_size)
                new_privkeys_data = np.concatenate([np.frombuffer(k, dtype=np.uint8) for k in new_privkeys])
                new_d_privkeys = cp.asarray(new_privkeys_data, dtype=cp.uint8)
                new_task = self.task_queue.create_task(new_d_privkeys, len(new_privkeys), steps_per_thread, self.device)
            
            # 更新任务状态
            self._update_task_states()
            
            # 防止忙等待
            time.sleep(0.001)
    
    def recover_private_key_from_index(self, match_info: dict) -> bytes:
        """从匹配索引恢复私钥"""
        indices = match_info['indices']
        batch_privkeys = match_info['batch_privkeys']
        num_keys = match_info['num_keys']
        steps_per_thread = match_info['steps_per_thread']
        
        if not indices:
            raise ValueError("No indices found in match_info")
        
        # 使用第一个匹配索引
        idx = indices[0]
        
        # 计算私钥偏移
        key_idx = idx // steps_per_thread
        step = idx % steps_per_thread
        
        if key_idx >= num_keys:
            raise ValueError(f"Key index {key_idx} out of range (max: {num_keys-1})")
        
        # 计算基私钥（每个私钥是32字节）
        base_privkey_start = key_idx * 32
        base_privkey_bytes = batch_privkeys[base_privkey_start:base_privkey_start + 32]
        
        # 计算最终私钥
        base = int.from_bytes(base_privkey_bytes, "big")
        k = base + step
        
        # 应用secp256k1曲线阶数模运算
        SECP256K1_ORDER_INT = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        if k >= SECP256K1_ORDER_INT:
            k -= SECP256K1_ORDER_INT
        if k == 0:
            k = 1
            
        return k.to_bytes(32, "big")