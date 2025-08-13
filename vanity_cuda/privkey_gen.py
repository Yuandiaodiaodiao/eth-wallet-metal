"""
Private key generation utilities for Ethereum vanity address generation
"""

import os
import secrets
from typing import List

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


def generate_sequential_keys(start: int, count: int) -> List[bytes]:
    """
    Generate sequential private keys starting from a given value
    Useful for testing and debugging
    
    Args:
        start: Starting value
        count: Number of keys to generate
        
    Returns:
        List of 32-byte private keys
    """
    keys = []
    for i in range(count):
        value = start + i
        # Convert to 32-byte big-endian
        key = value.to_bytes(32, byteorder='big')
        keys.append(key)
    
    return keys


def load_keys_from_file(filename: str) -> List[bytes]:
    """
    Load private keys from a file
    
    Args:
        filename: Path to file containing hex-encoded private keys (one per line)
        
    Returns:
        List of 32-byte private keys
    """
    keys = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove 0x prefix if present
                if line.startswith('0x'):
                    line = line[2:]
                
                # Convert hex to bytes
                try:
                    key = bytes.fromhex(line)
                    if len(key) == 32:
                        keys.append(key)
                    else:
                        print(f"Warning: Skipping invalid key length: {len(key)} bytes")
                except ValueError as e:
                    print(f"Warning: Skipping invalid hex: {line}")
    
    return keys


def save_keys_to_file(keys: List[bytes], filename: str):
    """
    Save private keys to a file
    
    Args:
        keys: List of private keys
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for key in keys:
            f.write(f"0x{key.hex()}\n")