
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
### 性能调优

1.逐步调高 @kernels.cu 中的 #define BATCH_WINDOW_SIZE 512    
512会占用 12G显存 , 占用越多性能释放越好  
2.拉高batch_size直到没有明显收益  
3. 调整 Stage 2: Walker kernel 后面的 block_size 逐渐调高直到哈希速率开始下降  




## Performance

Performance depends on GPU model and configuration:

- **RTX 5090**: ~1700 MAddr/s
- **RTX 3080TI**: ~500 MAddr/s
- **H800**: ~700 MAddr/s
