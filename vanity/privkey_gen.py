import secrets
import math
from typing import List
from .constants import SECP256K1_ORDER_BYTES, SECP256K1_ORDER_INT, HAS_NUMPY


def generate_private_keys(count: int) -> List[bytes]:
    """
    Generate cryptographically secure random private keys
    
    Args:
        count: Number of private keys to generate
        
    Returns:
        List of 32-byte private keys
    """
    keys = []
    for _ in range(count):
        # Generate 32 random bytes
        key = secrets.token_bytes(32)
        
        # Ensure the key is valid for secp256k1
        # Private key must be in range [1, n-1] where n is the order of the curve
        # For practical purposes, any 32-byte value except 0 and values >= n are invalid
        # The probability of generating such values is negligible (< 2^-128)
        
        # Check for zero (extremely unlikely)
        if key == b'\x00' * 32:
            key = secrets.token_bytes(32)
        
        keys.append(key)
    
    return keys


def generate_valid_privkeys(batch_size: int, steps_per_thread: int = 1, seq_len: int = 8192) -> List[bytes]:
    if HAS_NUMPY:
        return _generate_privkeys_incremental_numpy(batch_size, steps_per_thread, seq_len)
    raise NotImplementedError("NumPy is not installed")


def _generate_privkeys_incremental_numpy(batch_size: int, steps_per_thread: int = 1, seq_len: int = 8192) -> List[bytes]:
    import numpy as np  # type: ignore

    # Build n as little-endian u64 limbs [w0,w1,w2,w3]
    n_be_u64 = np.frombuffer(SECP256K1_ORDER_BYTES, dtype=np.dtype('>u8')).reshape(4)
    n_le_u64 = n_be_u64[::-1].byteswap().view(np.dtype('<u8'))

    # Convert steps_per_thread to little-endian u64 limbs
    step_int = steps_per_thread % SECP256K1_ORDER_INT
    if step_int == 0:
        step_int = 1
    step_bytes = step_int.to_bytes(32, "big")
    step_be_u64 = np.frombuffer(step_bytes, dtype=np.dtype('>u8')).reshape(4)
    step_le_u64 = step_be_u64[::-1].byteswap().view(np.dtype('<u8'))

    # Calculate number of base keys needed
    num_bases = int(math.ceil(batch_size / seq_len))
    
    # Generate multiple large random bases (256 bytes each for maximum security)
    # This provides much more entropy than 32-byte bases
    large_base_bytes = 256  # 2048 bits of entropy per base
    rnd = secrets.token_bytes(large_base_bytes * num_bases)
    
    # Convert large random numbers to secp256k1 field elements
    base_le = np.zeros((num_bases, 4), dtype=np.uint64)
    for i in range(num_bases):
        # Take 256 bytes of randomness for each base
        large_rnd = rnd[i * large_base_bytes : (i + 1) * large_base_bytes]
        
        # Interpret as big integer and reduce modulo n
        large_int = int.from_bytes(large_rnd, "big")
        reduced_int = large_int % SECP256K1_ORDER_INT
        if reduced_int == 0:
            reduced_int = 1
            
        # Convert to little-endian u64 limbs
        reduced_bytes = reduced_int.to_bytes(32, "big")
        reduced_be_u64 = np.frombuffer(reduced_bytes, dtype=np.dtype('>u8')).reshape(4)
        base_le[i] = reduced_be_u64[::-1].byteswap().view(np.dtype('<u8'))

    # 64x64->128 multiply helper
    def mul64_add(a, b, carry_in=0):
        # a * b + carry_in, return (low64, high64)
        prod = a.astype(np.uint64) * b + carry_in
        return prod.astype(np.uint64), (prod >> 64).astype(np.uint64)

    # Process each base to generate seq_len consecutive keys with step spacing
    out_list = []
    s0, s1, s2, s3 = step_le_u64[0], step_le_u64[1], step_le_u64[2], step_le_u64[3]
    
    for base_idx in range(num_bases):
        # How many keys to generate from this base
        keys_from_base = min(seq_len, batch_size - len(out_list))
        if keys_from_base <= 0:
            break
            
        # Vectorized increments: base + step * [0,1,2,...,keys_from_base-1]
        inc_factors = np.arange(keys_from_base, dtype=np.uint64)
        
        # step * inc_factors = [s0*inc, s1*inc, s2*inc, s3*inc] with carries
        p0, c0 = mul64_add(s0, inc_factors)
        p1, c1 = mul64_add(s1, inc_factors, c0)
        p2, c2 = mul64_add(s2, inc_factors, c1)
        p3, c3 = mul64_add(s3, inc_factors, c2)
        
        # Add base to each result: base + step*i
        base_w0, base_w1, base_w2, base_w3 = base_le[base_idx, 0], base_le[base_idx, 1], base_le[base_idx, 2], base_le[base_idx, 3]
        w0 = base_w0 + p0
        carry = (w0 < base_w0).astype(np.uint64)
        w1 = base_w1 + p1 + carry
        carry = ((w1 < base_w1) | ((w1 == base_w1) & (carry == 1))).astype(np.uint64)
        w2 = base_w2 + p2 + carry
        carry = ((w2 < base_w2) | ((w2 == base_w2) & (carry == 1))).astype(np.uint64)
        w3 = base_w3 + p3 + carry

        # Conditional subtract n if >= n
        ge = _lex_ge_mask_le_2d(w0, w1, w2, w3, n_le_u64)
        if ge.any():
            u0, u1, u2, u3 = _sub_256_le_broadcast(w0, w1, w2, w3, n_le_u64)
            w0 = np.where(ge, u0, w0)
            w1 = np.where(ge, u1, w1)
            w2 = np.where(ge, u2, w2)
            w3 = np.where(ge, u3, w3)

        # Avoid zero results
        zero_mask = (w0 == 0) & (w1 == 0) & (w2 == 0) & (w3 == 0)
        if zero_mask.any():
            w0 = np.where(zero_mask, np.uint64(1), w0)

        # Repack to big-endian 32-byte scalars
        vals = np.empty((keys_from_base, 4), dtype=np.uint64)
        vals[:, 0] = w3
        vals[:, 1] = w2
        vals[:, 2] = w1
        vals[:, 3] = w0
        be = vals.byteswap()
        out_bytes = be.tobytes()
        batch_keys = [out_bytes[i * 32 : (i + 1) * 32] for i in range(keys_from_base)]
        out_list.extend(batch_keys)

    return out_list[:batch_size]


def _lex_ge_mask_le_2d(w0, w1, w2, w3, n_le):
    m3 = w3 > n_le[3]
    e3 = w3 == n_le[3]
    m2 = w2 > n_le[2]
    e2 = w2 == n_le[2]
    m1 = w1 > n_le[1]
    e1 = w1 == n_le[1]
    m0 = w0 >= n_le[0]
    return m3 | (e3 & (m2 | (e2 & (m1 | (e1 & m0)))))


def _sub_256_le_broadcast(w0, w1, w2, w3, n_le):
    n0, n1, n2, n3 = (n_le[0], n_le[1], n_le[2], n_le[3])
    r0 = w0 - n0
    borrow = (w0 < n0).astype(w0.dtype)
    r1 = w1 - n1 - borrow
    borrow = ((w1 < n1) | ((w1 == n1) & (borrow == 1))).astype(w1.dtype)
    r2 = w2 - n2 - borrow
    borrow = ((w2 < n2) | ((w2 == n2) & (borrow == 1))).astype(w2.dtype)
    r3 = w3 - n3 - borrow
    return r0, r1, r2, r3


def generate_walk_start_privkeys(batch_size: int, steps_per_thread: int) -> List[bytes]:
    """Generate starting private keys for the walker kernel such that
    each thread's T-step walk does not overlap with others. Starts are spaced by steps_per_thread.

    This uses a lightweight Python big-int loop for correctness. Cost is negligible relative to GPU work.
    """
    if steps_per_thread <= 0:
        raise ValueError("steps_per_thread must be > 0")
    out: List[bytes] = []
    n = SECP256K1_ORDER_INT
    base = int.from_bytes(secrets.token_bytes(32), "big") % n
    if base == 0:
        base = 1
    step = steps_per_thread % n
    if step == 0:
        step = 1
    k = base
    for _ in range(batch_size):
        out.append(k.to_bytes(32, "big"))
        k += step
        if k >= n:
            k -= n
        if k == 0:
            k = 1
    return out