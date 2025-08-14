
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage


```bash
python vanity_main.py 

```

### Change Config

```python
def main():
    """Main entry point"""
    # Configuration variables
    pattern = "8888,8888"  # Can be "888" for head, ",abc" for tail, or "888,abc" for both
    batch_size = 4096*32
    steps = 512*8
    device = 0
    benchmark = False
    useWalker = True
    use_parallel = True 

```

## Performance

Performance depends on GPU model and configuration:

- **RTX 5090**: ~1700 MAddr/s
- **RTX 3080TI**: ~500 MAddr/s
- **H800**: ~700 MAddr/s
