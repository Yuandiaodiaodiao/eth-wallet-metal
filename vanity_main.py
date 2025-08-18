import os
import sys
import time
from typing import Optional

from vanity.gpu_vanity import MetalVanity
from vanity.privkey_gen import generate_valid_privkeys, SECP256K1_ORDER_INT

def hex_addr(b: bytes) -> str:
    return "0x" + b.hex()



def main(batch_size: int = 384*8, max_batches: Optional[int] = None, steps_per_thread: int = 256*16, prefix_hex: Optional[str] = "000000", suffix_hex: Optional[str] = "000000") -> None:
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vanity")
    engine = MetalVanity(here)
    batches = 0
    # Triple-buffering with compact GPU output to maximize GPU utilization
    start_time = time.perf_counter()
    total_keys = 0
    
    # Initialize first three buffers
    t_gen0 = time.perf_counter()
    privs_0 = generate_valid_privkeys(batch_size, steps_per_thread, 128)
    gen_0_sec = time.perf_counter() - t_gen0
    job_0 = engine.encode_and_commit_walk_compact(privs_0, steps_per_thread=steps_per_thread, prefix_hex=prefix_hex, suffix_hex=suffix_hex)
    
    t_gen1 = time.perf_counter()
    privs_1 = generate_valid_privkeys(batch_size, steps_per_thread, 128)
    gen_1_sec = time.perf_counter() - t_gen1
    job_1 = engine.encode_and_commit_walk_compact(privs_1, steps_per_thread=steps_per_thread, prefix_hex=prefix_hex, suffix_hex=suffix_hex)
    
    # Circular buffer indices
    current_buffer = 0
    privs = [privs_0, privs_1]
    jobs = [job_0, job_1]
    gen_times = [gen_0_sec, gen_1_sec]

    process_birth = time.perf_counter()
    while True:
        batches += 1
        # Wait for the oldest job to complete
        oldest_idx = current_buffer
        indices, steps_effective = engine.wait_and_collect_compact(jobs[oldest_idx])
        print(f'batch: {batches}')
        total_keys += batch_size * steps_per_thread
        avg_elapsed = max(time.perf_counter() - start_time, 1e-9)
        avg_rate = total_keys / avg_elapsed
        
        # Timing diagnostics
        cpu_encode = jobs[oldest_idx].cpu_encode_seconds
        gpu_ms = -1.0
        if getattr(jobs[oldest_idx], 'gpu_start_time', -1.0) and getattr(jobs[oldest_idx], 'gpu_end_time', -1.0):
            try:
                gpu_ms = max(0.0, (jobs[oldest_idx].gpu_end_time - jobs[oldest_idx].gpu_start_time) * 1e3)
            except Exception:
                gpu_ms = -1.0
                
        print(
            f"avg rate: {avg_rate:,.2f} keys/s ({avg_rate/1e6:.3f} MH/s) | "
            f"CPU gen: {gen_times[oldest_idx]*1e3:.2f} ms, CPU encode: {cpu_encode*1e3:.2f} ms, GPU: {gpu_ms:.2f} ms"
        )

        if indices:
            print(f"indices: {indices}")
            idx = indices[0]
            # Use the steps used by the GPU job that produced this result
            gid = idx // max(1, steps_effective)
            off = idx % max(1, steps_effective)
            base = int.from_bytes(privs[oldest_idx][gid], "big")
            k = base + off
            n = SECP256K1_ORDER_INT
            if k >= n:
                k -= n
            if k == 0:
                k = 1
            k_bytes = k.to_bytes(32, "big")
            # Verify address for this priv using a direct compact kernel call
            verify_job = engine.encode_and_commit_walk_compact([k_bytes], steps_per_thread=1, prefix_hex=None, suffix_hex=None)
            v_indices, _ = engine.wait_and_collect_compact(verify_job)
            print(f'walk indices: {v_indices}')
            
            print("\nTesting compact:")
            verify_job_correct = engine.encode_and_commit_compact([k_bytes], prefix_hex=None, suffix_hex=None)
            indices, _ = engine.wait_and_collect_compact(verify_job_correct)
            
            print(f'compact indices: {indices}')
            
            
            print(f'私钥: {k_bytes.hex()}')
            # Check if g16 buffer is available
            print(f"g16_buffer available: {engine.g16_buffer is not None}")
            print(f"pipeline_w16_compact available: {engine.pipeline_w16_compact is not None}")
            # print(f"pipeline_walk_w16_compact available: {engine.pipeline_walk_w16_compact is not None}")
            return
            
        # Generate new keys and submit new job to replace the completed one
        t_gen = time.perf_counter()
        privs[oldest_idx] = generate_valid_privkeys(batch_size, steps_per_thread, 128)
        gen_times[oldest_idx] = time.perf_counter() - t_gen
        jobs[oldest_idx] = engine.encode_and_commit_walk_compact(privs[oldest_idx], steps_per_thread=steps_per_thread, prefix_hex=prefix_hex, suffix_hex=suffix_hex)
        
        # Move to next buffer in circular fashion
        current_buffer = (current_buffer + 1) % len(jobs)
        
        # Periodic self-fork-and-restart every 5 minutes to mitigate any long-lived leaks
        if time.perf_counter() - process_birth >= 300:
            print("[self-restart] 5 minutes elapsed, forking a fresh worker and exiting old process...")
            try:
                pid = os.fork()
            except OSError:
                pid = -1
            if pid == 0:
                # Child: exec a fresh interpreter running this script
                script_path = os.path.abspath(__file__)
                os.execv(sys.executable, [sys.executable, script_path])
            else:
                # Parent: exit immediately
                os._exit(0)
        
        if max_batches is not None and batches >= max_batches:
            print("No match in", batches, "batches")
            return


if __name__ == "__main__":
    # default: find address with first 7 hex nibbles == 8
    main()


