import os
import secrets
import binascii
from typing import Tuple

from ecdsa import SigningKey, SECP256k1

from gpu_keccak import keccak256_gpu_single
from gpu_secp256k1 import secp256k1_pubkey_gpu


def priv_to_pub_uncompressed_xy(privkey32: bytes) -> bytes:
    sk = SigningKey.from_string(privkey32, curve=SECP256k1, hashfunc=None)
    vk = sk.get_verifying_key()
    # Uncompressed point without 0x04 prefix: 32-byte X || 32-byte Y
    x = vk.pubkey.point.x()
    y = vk.pubkey.point.y()
    x_bytes = int(x).to_bytes(32, "big")
    y_bytes = int(y).to_bytes(32, "big")
    return x_bytes + y_bytes


def eth_address_from_pubkey64_gpu(pubkey64: bytes, metal_src_path: str) -> str:
    digest = keccak256_gpu_single(pubkey64, metal_src_path)
    # Ethereum address = last 20 bytes of Keccak-256(pubkey) (no 0x04 prefix)
    addr = digest[-20:]
    return "0x" + addr.hex()


def generate_random_privkey() -> bytes:
    # 32 random bytes in range [1, n-1] — retry if invalid (rare)
    n = SECP256k1.order
    while True:
        k = secrets.token_bytes(32)
        ki = int.from_bytes(k, "big")
        if 1 <= ki < n:
            return k


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    metal_src = os.path.join(here, "keccak256.metal")

    priv = generate_random_privkey()
    # GPU-derived public key (x||y)
    pubxy = secp256k1_pubkey_gpu(priv, here)
    addr = eth_address_from_pubkey64_gpu(pubxy, metal_src)

    print("Private key:", priv.hex())
    print("Public key (x||y):", pubxy.hex())
    print("Ethereum address:", addr)


if __name__ == "__main__":
    main()


