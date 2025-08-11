import os
from ecdsa import SigningKey, SECP256k1
import sha3
from gpu_keccak import MetalKeccak256
from gpu_secp256k1 import secp256k1_pubkey_gpu

def main():
    priv_hex = "a341d05f6775643fb38c053ab6a7adb6b37fce31bcaca0b2ae434f14f18ab58b"
    priv = bytes.fromhex(priv_hex)
    # GPU pubkey
    pub = secp256k1_pubkey_gpu(priv, os.path.dirname(os.path.abspath(__file__)))

    # CPU reference (Ethereum Keccak)
    k = sha3.keccak_256(); k.update(pub)
    cpud = k.digest()
    cpu_addr = '0x' + cpud[-20:].hex()

    # GPU result
    engine = MetalKeccak256("/Users/ai/Documents/gpuaddress2/keccak256.metal")
    gpud = engine.keccak256_many([pub])[0]
    gpu_addr = '0x' + gpud[-20:].hex()

    print("CPU:", cpu_addr)
    print("GPU:", gpu_addr)

if __name__ == '__main__':
    main()


