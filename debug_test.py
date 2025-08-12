import os
from gpu_vanity import MetalVanity

def hex_addr(b: bytes) -> str:
    return "0x" + b.hex()

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    engine = MetalVanity(here)
    k_bytes = bytes.fromhex('96210e2bc86ee70d072362702f995366cab530cd40bdba115f96350258a21126')
    
    # Test with different scenarios
    print("=== Testing with g16 disabled ===")
    
    # Temporarily disable g16 for both
    orig_g16 = engine.g16_buffer
    engine.g16_buffer = None
    
    print("Testing walk_compact (no g16):")
    verify_job = engine.encode_and_commit_walk_compact([k_bytes], steps_per_thread=1, nibble=0x0, nibble_count=0)
    v_addrs, v_indices, _ = engine.wait_and_collect_compact(verify_job)
    walk_addr = v_addrs[0] if v_addrs else b""
    print(f'walk addr: {hex_addr(walk_addr)}')
    
    print("Testing compact (no g16):")
    verify_job_correct = engine.encode_and_commit_compact([k_bytes], nibble=0x0, nibble_count=0)
    addrs, indices, _ = engine.wait_and_collect_compact(verify_job_correct)
    compact_addr = addrs[0] if addrs else b""
    print(f'compact addr: {hex_addr(compact_addr)}')
    
    print(f"Match (no g16): {hex_addr(walk_addr) == hex_addr(compact_addr)}")
    
    # Restore g16 buffer
    engine.g16_buffer = orig_g16
    
    print("\n=== Testing with g16 enabled ===")
    
    print("Testing walk_compact (with g16):")
    verify_job = engine.encode_and_commit_walk_compact([k_bytes], steps_per_thread=1, nibble=0x0, nibble_count=0)
    v_addrs, v_indices, _ = engine.wait_and_collect_compact(verify_job)
    walk_addr_g16 = v_addrs[0] if v_addrs else b""
    print(f'walk addr: {hex_addr(walk_addr_g16)}')
    
    print("Testing compact (with g16):")
    verify_job_correct = engine.encode_and_commit_compact([k_bytes], nibble=0x0, nibble_count=0)
    addrs, indices, _ = engine.wait_and_collect_compact(verify_job_correct)
    compact_addr_g16 = addrs[0] if addrs else b""
    print(f'compact addr: {hex_addr(compact_addr_g16)}')
    
    print(f"Match (with g16): {hex_addr(walk_addr_g16) == hex_addr(compact_addr_g16)}")
    
    print(f"\nCompact consistency: {hex_addr(compact_addr) == hex_addr(compact_addr_g16)}")
    print(f"Walk consistency: {hex_addr(walk_addr) == hex_addr(walk_addr_g16)}")

if __name__ == "__main__":
    main()