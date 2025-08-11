import os
import time
from typing import Optional

from gpu_vanity import MetalVanity, generate_valid_privkeys


def hex_addr(b: bytes) -> str:
    return "0x" + b.hex()


def main(batch_size: int = 4096*16, nibble: int = 0x8, nibble_count: int = 5, max_batches: Optional[int] = None) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    engine = MetalVanity(here)
    batches = 0
    # Double-buffering with compact GPU output to minimize readback
    privs_prev = generate_valid_privkeys(batch_size)
    job_prev = engine.encode_and_commit_compact(privs_prev, nibble=nibble, nibble_count=nibble_count)
    start_time = time.perf_counter()
    total_keys = 0

    while True:
        batches += 1
        # print(f'batch {batches}')
        # Prepare and immediately commit the next batch so a command is always in-flight
        privs_next = generate_valid_privkeys(batch_size)
        job_next = engine.encode_and_commit_compact(privs_next, nibble=nibble, nibble_count=nibble_count)

        # Now wait and collect the previous job while the next one is queued/running
        addrs, indices = engine.wait_and_collect_compact(job_prev)
        total_keys += batch_size
        avg_elapsed = max(time.perf_counter() - start_time, 1e-9)
        avg_rate = total_keys / avg_elapsed
        print(f"avg rate: {avg_rate:,.2f} keys/s ({avg_rate/1e6:.3f} MH/s)")

        if indices:
            i = indices[0]
            print("FOUND:")
            print("priv:", privs_prev[i].hex())
            print("addr:", hex_addr(addrs[0]))
            return
        # Shift next -> prev for the next iteration
        privs_prev = privs_next
        job_prev = job_next
        if max_batches is not None and batches >= max_batches:
            print("No match in", batches, "batches")
            return


if __name__ == "__main__":
    # default: find address with first 7 hex nibbles == 8
    main()


