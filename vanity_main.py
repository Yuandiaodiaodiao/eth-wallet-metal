import os
import time
from typing import Optional

from gpu_vanity import MetalVanity, generate_valid_privkeys


def hex_addr(b: bytes) -> str:
    return "0x" + b.hex()



def main(batch_size: int = 4096*8, nibble: int = 0x8, nibble_count: int = 6, max_batches: Optional[int] = None) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    engine = MetalVanity(here)
    batches = 0
    # Double-buffering with compact GPU output to minimize readback
    t_gen0 = time.perf_counter()
    privs_prev = generate_valid_privkeys(batch_size)
    gen_prev_sec = time.perf_counter() - t_gen0
    job_prev = engine.encode_and_commit_compact(privs_prev, nibble=nibble, nibble_count=nibble_count)
    start_time = time.perf_counter()
    total_keys = 0

    while True:
        batches += 1
        # print(f'batch {batches}')
        # Prepare and immediately commit the next batch so a command is always in-flight
        t_gen = time.perf_counter()
        privs_next = generate_valid_privkeys(batch_size)
        gen_next_sec = time.perf_counter() - t_gen
        job_next = engine.encode_and_commit_compact(privs_next, nibble=nibble, nibble_count=nibble_count)

        # Now wait and collect the previous job while the next one is queued/running
        addrs, indices = engine.wait_and_collect_compact(job_prev)
        total_keys += batch_size
        avg_elapsed = max(time.perf_counter() - start_time, 1e-9)
        avg_rate = total_keys / avg_elapsed
        # Timing diagnostics
        cpu_encode = job_prev.cpu_encode_seconds
        gpu_ms = -1.0
        if getattr(job_prev, 'gpu_start_time', -1.0) and getattr(job_prev, 'gpu_end_time', -1.0):
            try:
                gpu_ms = max(0.0, (job_prev.gpu_end_time - job_prev.gpu_start_time) * 1e3)
            except Exception:
                gpu_ms = -1.0
        # Which generation time to attribute to the batch we just completed?
        # gen_prev_sec corresponds to prev batch's key generation
        print(
            f"avg rate: {avg_rate:,.2f} keys/s ({avg_rate/1e6:.3f} MH/s) | "
            f"CPU gen: {gen_prev_sec*1e3:.2f} ms, CPU encode: {cpu_encode*1e3:.2f} ms, GPU: {gpu_ms:.2f} ms"
        )

        if indices:
            i = indices[0]
            print("FOUND:")
            print("priv:", privs_prev[i].hex())
            print("addr:", hex_addr(addrs[0]))
            return
        # Shift next -> prev for the next iteration
        privs_prev = privs_next
        job_prev = job_next
        gen_prev_sec = gen_next_sec
        if max_batches is not None and batches >= max_batches:
            print("No match in", batches, "batches")
            return


if __name__ == "__main__":
    # default: find address with first 7 hex nibbles == 8
    main()


