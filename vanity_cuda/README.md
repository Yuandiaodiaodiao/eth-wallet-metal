# CUDA Ethereum Vanity Address Generator

High-performance Ethereum vanity address generator using CUDA and cuda-python.

## Features

- **Optimized CUDA Kernels**: Leverages GPU parallel processing for massive throughput
- **16-bit Window Scalar Multiplication**: Fast elliptic curve operations using precomputed tables
- **Batch Processing**: Process thousands of keys simultaneously
- **Walker Mode**: Generate multiple addresses per thread for improved efficiency
- **Montgomery's Trick**: Batch inversion for efficient Jacobian to affine conversion

## Requirements

- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit 12.0 or higher
- Python 3.8+

## Installation

1. Install CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
# Basic usage - search for addresses starting with 0x888
python vanity_main.py --pattern 888

# Specify batch size and steps
python vanity_main.py --pattern 888 --batch-size 50000 --steps 512

# Run for specific time (seconds)
python vanity_main.py --pattern 888 --time 60

# Run benchmark
python vanity_main.py --benchmark
```

### Python API

```python
from vanity_cuda import CudaVanity

# Initialize generator
generator = CudaVanity(device_id=0)

# Generate random private keys
import os
privkeys = [os.urandom(32) for _ in range(10000)]

# Search for vanity addresses
indices, gpu_time = generator.generate_vanity_walker(
    privkeys,
    steps_per_thread=256,
    target_nibble=0x8,
    nibble_count=3  # Search for 0x888...
)

print(f"Found {len(indices)} matches in {gpu_time:.3f} seconds")
```

## Performance

Performance depends on GPU model and configuration:

- **RTX 4090**: ~500-1000 MAddr/s
- **RTX 3080**: ~300-600 MAddr/s
- **RTX 2080**: ~200-400 MAddr/s

Walker mode with optimal steps_per_thread (typically 256-512) provides best throughput.

## Architecture

### Kernel Design

1. **vanity_kernel_g16**: Simple kernel, one address per thread
2. **vanity_walker_kernel**: Advanced kernel, multiple addresses per thread
3. **build_g16_table_kernel**: Builds precomputed scalar multiplication table

### Optimizations

- **G16 Precomputed Table**: 16-bit window scalar multiplication
- **Batch Inversion**: Montgomery's trick for efficient coordinate conversion
- **Shared Memory**: Optimized memory access patterns
- **PTX Assembly**: Hand-optimized 256-bit arithmetic operations

### Memory Layout

- Private keys: 32 bytes, big-endian
- Public keys: 64 bytes (32 bytes X, 32 bytes Y), big-endian
- G16 table: 64 MB (16 windows × 65536 entries × 64 bytes)

## File Structure

```
vanity_cuda/
├── kernels.cu          # CUDA kernel implementations
├── secp256k1.cuh       # Elliptic curve operations
├── keccak256.cuh       # Keccak-256 hash function
├── cuda_vanity.py      # Python wrapper using cuda-python
├── vanity_main.py      # CLI interface
├── privkey_gen.py      # Private key generation utilities
└── requirements.txt    # Python dependencies
```

## Vanity Pattern Format

Patterns are specified as hexadecimal strings:
- `"8"` - Addresses starting with 0x8
- `"888"` - Addresses starting with 0x888
- `"dead"` - Addresses starting with 0xdead

Longer patterns exponentially increase search time:
- 1 nibble: ~16 addresses
- 2 nibbles: ~256 addresses
- 3 nibbles: ~4,096 addresses
- 4 nibbles: ~65,536 addresses

## Security Notes

- Private keys are generated using cryptographically secure random sources
- Never share or expose private keys
- Vanity addresses provide no additional security
- Always verify generated addresses before use

## License

MIT License - See LICENSE file for details