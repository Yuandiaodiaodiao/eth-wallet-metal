import os
import time
from typing import Optional

from gpu_vanity import MetalVanity
from privkey_gen import generate_valid_privkeys, SECP256K1_ORDER_INT


def hex_addr(b: bytes) -> str:
    return "0x" + b.hex()



def main(batch_size: int = 4096, nibble: int = 0x8, nibble_count: int = 5, max_batches: Optional[int] = None, steps_per_thread: int = 8) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    engine = MetalVanity(here)
    k_bytes = bytes.fromhex('96210e2bc86ee70d072362702f995366cab530cd40bdba115f96350258a21126')
    
    print("Testing walk_compact:")
    verify_job = engine.encode_and_commit_walk_compact([k_bytes], steps_per_thread=1, nibble=0x0, nibble_count=0)
    v_indices, steps = engine.wait_and_collect_compact(verify_job)
    print(f'walk indices: {v_indices}')
    
    print("\nTesting compact:")
    verify_job_correct = engine.encode_and_commit_compact([k_bytes], nibble=0x0, nibble_count=0)
    indices, steps = engine.wait_and_collect_compact(verify_job_correct)
    
    print(f'compact indices: {indices}')
    
    # Check if g16 buffer is available
    print(f"g16_buffer available: {engine.g16_buffer is not None}")
    print(f"pipeline_w16_compact available: {engine.pipeline_w16_compact is not None}")
    print(f"pipeline_compute_base_w16 available: {engine.pipeline_compute_base_w16 is not None}")


if __name__ == "__main__":
    # default: find address with first 7 hex nibbles == 8
    main()


