"""
CUDA-accelerated Ethereum vanity address generator
"""

from .cuda_vanity import CudaVanity
from .vanity_main import VanityAddressGenerator
from .privkey_gen import generate_private_keys, generate_valid_privkeys

__version__ = "1.0.0"
__all__ = [
    "CudaVanity",
    "VanityAddressGenerator",
    "generate_private_keys",
    "generate_valid_privkeys"
]