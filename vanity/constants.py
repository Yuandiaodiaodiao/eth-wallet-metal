try:
    from ecdsa import SECP256k1  # type: ignore
    SECP256K1_ORDER_BYTES = SECP256k1.order.to_bytes(32, "big")
except Exception:  # Fallback if ecdsa not importable at import-time
    SECP256K1_ORDER_BYTES = int(
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    ).to_bytes(32, "big")

ZERO_32 = b"\x00" * 32

# Integer curve order for fast modular arithmetic
SECP256K1_ORDER_INT = int.from_bytes(SECP256K1_ORDER_BYTES, "big")

# Optional NumPy acceleration (SIMD on Apple Silicon)
try:
    import numpy as np  # type: ignore
    HAS_NUMPY = True
except Exception:  # Optional dependency
    HAS_NUMPY = False