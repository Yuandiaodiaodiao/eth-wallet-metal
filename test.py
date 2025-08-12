import os
import time
from typing import Optional

from gpu_vanity import MetalVanity, generate_valid_privkeys, SECP256K1_ORDER_INT


def hex_addr(b: bytes) -> str:
    return "0x" + b.hex()



def main(batch_size: int = 4096, nibble: int = 0x8, nibble_count: int = 5, max_batches: Optional[int] = None, steps_per_thread: int = 8) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    engine = MetalVanity(here)
    k_bytes = bytes.fromhex('96210e2bc86ee70d072362702f995366cab530cd40bdba115f96350258a21126')
    verify_job = engine.encode_and_commit_walk_compact([k_bytes], steps_per_thread=1, nibble=0x0, nibble_count=0)
    v_addrs, v_indices, _ = engine.wait_and_collect_compact(verify_job)
    verified_addr = v_addrs[0] if v_addrs else b""
    print(f'indices: {v_indices}')
    print("addr:", hex_addr(verified_addr))
    
    verify_job_correct = engine.encode_and_commit_compact([k_bytes], nibble=0x0, nibble_count=0)
    addrs, indices, _ = engine.wait_and_collect_compact(verify_job_correct)
    
    print(f'indices: {indices}')
    print(f"addr:", hex_addr(addrs[0]))


if __name__ == "__main__":
    # default: find address with first 7 hex nibbles == 8
    main()


