#!/usr/bin/env python3
"""
Main entry point for CUDA-based Ethereum vanity address generator
Compatible with the existing vanity address generation workflow
"""
try:
    from ecdsa import SECP256k1  # type: ignore
    SECP256K1_ORDER_BYTES = SECP256k1.order.to_bytes(32, "big")
except Exception:  # Fallback if ecdsa not importable at import-time
    SECP256K1_ORDER_BYTES = int(
        0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    ).to_bytes(32, "big")


# Integer curve order for fast modular arithmetic
SECP256K1_ORDER_INT = int.from_bytes(SECP256K1_ORDER_BYTES, "big")

import os
import sys
import time
import json
import threading
import queue
from pathlib import Path
from typing import List, Dict, Any, Optional

# Try to import web3 for address generation
try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from cuda_vanity import CudaVanity
from privkey_gen import generate_valid_privkeys

class PrivkeyGenerator(threading.Thread):
    """Background thread for continuous private key generation"""
    
    def __init__(self, batch_size: int, steps_per_thread: int, queue_maxsize: int = 3):
        super().__init__(daemon=True)
        self.batch_size = batch_size
        self.steps_per_thread = steps_per_thread
        self.privkey_queue = queue.Queue(maxsize=queue_maxsize)
        self._stop_event = threading.Event()
        self.generation_count = 0
        
    def run(self):
        """Generate private keys in background"""
        while not self._stop_event.is_set():
            try:
                # Generate private keys
                start_time = time.perf_counter()
                privkeys = generate_valid_privkeys(self.batch_size, self.steps_per_thread, 128)
                generation_time = time.perf_counter() - start_time
                
                # Put in queue (this will block if queue is full)
                self.privkey_queue.put((privkeys, generation_time), timeout=1.0)
                self.generation_count += 1
                
            except queue.Full:
                # Queue is full, GPU is processing fast enough
                continue
            except Exception as e:
                print(f"Error in privkey generation: {e}")
                break
                
    def get_privkeys(self, timeout=None):
        """Get next batch of private keys"""
        try:
            return self.privkey_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None
            
    def stop(self):
        """Stop the generator thread"""
        self._stop_event.set()
        
    def queue_size(self):
        """Get current queue size"""
        return self.privkey_queue.qsize()

class VanityAddressGenerator:
    """High-level interface for vanity address generation"""
    
    def __init__(self, device_id: int = 0):
        """Initialize the vanity address generator"""
        self.cuda_generator = CudaVanity(device_id=device_id)
        self.stats = {
            "total_keys_processed": 0,
            "total_addresses_checked": 0,
            "total_matches_found": 0,
            "total_gpu_time": 0.0,
            "start_time": time.time()
        }
    


    def run_continuous_pipeline(self,
                               target_pattern: str = "888",
                               batch_size: int = 10000,
                               steps_per_thread: int = 256):
        """
        使用新的GPU pipeline系统运行连续vanity地址生成
        
        Args:
            target_pattern: 要搜索的十六进制模式
            batch_size: 每批次的密钥数量  
            steps_per_thread: walker kernel每个线程的步数
        """
        print(f"Starting PIPELINE generation for pattern: {target_pattern}")
        print(f"Batch size: {batch_size}, Steps per thread: {steps_per_thread}")
        print(f"Press Ctrl+C to stop\n")
        
        # 解析模式  
        if "," in target_pattern:
            head_pattern, tail_pattern = target_pattern.split(",", 1)
        else:
            head_pattern = target_pattern
            tail_pattern = ""
        
        # 创建私钥生成器函数
        def privkey_generator_func(size):
            return generate_valid_privkeys(size, steps_per_thread, 128)
        
        # 使用pipeline生成器
        pipeline = self.cuda_generator.create_continuous_pipeline(
            privkey_generator=privkey_generator_func,
            batch_size=batch_size,
            steps_per_thread=steps_per_thread,
            head_pattern=head_pattern,
            tail_pattern=tail_pattern
        )
        
        all_matches = []
        batch_count = 0
        total_task_time = 0.0
        start_time = time.perf_counter()
        
        try:
            print("Starting pipeline execution...")
            
            for result, task_stats in pipeline:
                batch_count += 1
                
                # 更新统计信息
                addresses_checked = task_stats['keys_processed']
                self.stats["total_keys_processed"] += batch_size
                self.stats["total_addresses_checked"] += addresses_checked
                self.stats["total_gpu_time"] += task_stats['total_time']
                total_task_time += task_stats['total_time']
                
                # 处理当前批次的结果
                results = {"matches": [], "batch_stats": {
                    "keys_processed": batch_size,
                    "addresses_checked": addresses_checked,
                    "matches_found": 0,
                    "total_time": task_stats['total_time'],
                    "compute_time": task_stats['compute_time'],
                    "walker_time": task_stats['walker_time'],
                    "throughput": addresses_checked / task_stats['total_time'] / 1e6 if task_stats['total_time'] > 0 else 0
                }}
                
                if result is not None:
                    # 找到匹配，恢复私钥
                    indices = result['indices']
                    results["batch_stats"]["matches_found"] = len(indices)
                    self.stats["total_matches_found"] += len(indices)
                    
                    try:
                        # 恢复私钥
                        recovered_privkey = self.cuda_generator.recover_private_key_from_index(result)
                        
                        for idx in indices:
                            results["matches"].append({
                                "private_key": recovered_privkey.hex(),
                                "found_index": idx,
                                "task_id": task_stats['task_id'],
                                "head_pattern": head_pattern if head_pattern else None,
                                "tail_pattern": tail_pattern if tail_pattern else None,
                                "pattern": target_pattern
                            })
                    except Exception as e:
                        print(f"Error recovering private key: {e}")
                        for idx in indices:
                            results["matches"].append({
                                "found_index": idx,
                                "task_id": task_stats['task_id'],
                                "error": str(e),
                                "pattern": target_pattern
                            })
                
                # Store matches
                all_matches.extend(results["matches"])
                
                # 获取GPU利用率统计  
                gpu_stats = self.cuda_generator.get_gpu_utilization_stats()
                avg_throughput = self.stats["total_addresses_checked"] / self.stats["total_gpu_time"] / 1e6 if self.stats["total_gpu_time"] > 0 else 0
                
                print(f"Batch {batch_count}: "
                      f"Found {len(results['matches'])} matches, "
                      f"Total: {self.stats['total_matches_found']}, "
                      f"Throughput: {results['batch_stats']['throughput']:.2f} MAddr/s, "
                      f"Avg: {avg_throughput:.2f} MAddr/s")
                print(f"    Pipeline Timing - Total: {task_stats['total_time']*1000:.1f}ms, "
                      f"Compute: {task_stats['compute_time']*1000:.1f}ms, "
                      f"Walker: {task_stats['walker_time']*1000:.1f}ms")
                print(f"    GPU Stats - Utilization: {gpu_stats['gpu_utilization_percent']:.1f}%, "
                      f"Pipeline Efficiency: {gpu_stats['pipeline_efficiency']:.1f}%, "
                      f"Active Tasks: {gpu_stats['total_active_tasks']}")
                
                # 检查是否找到匹配
                if len(all_matches) > 0:
                    match = all_matches[0]
                    print(f"\n*** MATCH FOUND! ***")
                    if 'private_key' in match:
                        print(f"Private key: {match['private_key']}")
                        
                        # 如果有web3，生成地址
                        if WEB3_AVAILABLE:
                            try:
                                account = Account.from_key(match['private_key'])
                                print(f"Address: {account.address}")
                                print(f"\nResult: private_key='{match['private_key']}', address='{account.address}'")
                            except Exception as e:
                                print(f"Error generating address: {e}")
                    else:
                        print(f"Found index: {match['found_index']}")
                        print(f"Error: {match.get('error', 'Unknown error')}")
                    
                    print(f"Task ID: {match['task_id']}")
                    print(f"Pattern: {match['pattern']}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nStopping pipeline generation...")
        except StopIteration:
            print("\nPipeline completed")
        
        # 打印最终的pipeline统计信息
        self._print_final_stats_pipeline(total_task_time, start_time)
    

    
    def _print_final_stats_pipeline(self, total_task_time: float, start_time: float):
        """打印pipeline执行的最终统计信息"""
        elapsed = time.perf_counter() - start_time
        
        print("\n" + "="*60)
        print("Final Statistics (Pipeline Execution):")
        print(f"  Total elapsed time: {elapsed:.2f} seconds")
        print(f"  Total GPU time: {self.stats['total_gpu_time']:.2f} seconds")
        print(f"  Total task time: {total_task_time:.2f} seconds")
        print(f"  Keys processed: {self.stats['total_keys_processed']:,}")
        print(f"  Addresses checked: {self.stats['total_addresses_checked']:,}")
        print(f"  Matches found: {self.stats['total_matches_found']:,}")
        
        if self.stats["total_gpu_time"] > 0:
            avg_throughput = self.stats["total_addresses_checked"] / self.stats["total_gpu_time"] / 1e6
            print(f"  Average throughput: {avg_throughput:.2f} MAddr/s")
            
            # Pipeline效率指标
            gpu_utilization = self.stats["total_gpu_time"] / elapsed * 100
            pipeline_efficiency = total_task_time / elapsed * 100
            
            print(f"  GPU utilization: {gpu_utilization:.1f}%")
            print(f"  Pipeline efficiency: {pipeline_efficiency:.1f}%")
            
            # 获取最终GPU统计
            final_gpu_stats = self.cuda_generator.get_gpu_utilization_stats()
            print(f"  Final pipeline state: {final_gpu_stats['pipeline_efficiency']:.1f}% efficient")
            print(f"  Active tasks at end: {final_gpu_stats['total_active_tasks']}")
        
        if self.stats["total_matches_found"] > 0:
            probability = self.stats["total_matches_found"] / self.stats["total_addresses_checked"]
            print(f"  Match probability: {probability:.6e}")


def main():
    """Main entry point"""
    # Configuration variables
    pattern = "****,****"  # Can be "888" for head, ",abc" for tail, or "888,abc" for both
    batch_size = 4096*32
    steps = 512*16
    device = 0
    useWalker = True
    
    # Initialize generator
    generator = VanityAddressGenerator(device_id=device)
    
    print(f"Configuration:")
    print(f"  Pattern: {pattern}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per thread: {steps}")
    print(f"  Use Walker: {useWalker}")
    print()
    
    
    generator.run_continuous_pipeline(
        target_pattern=pattern,
        batch_size=batch_size,
        steps_per_thread=steps
    )
 


if __name__ == "__main__":
    main()