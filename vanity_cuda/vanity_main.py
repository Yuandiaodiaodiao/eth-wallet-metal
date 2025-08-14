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
    
    def generate_batch(self,
                      batch_size: int = 10000,
                      steps_per_thread: int = 256,
                      target_pattern: str = "888",
                      use_walker: bool = True) -> Dict[str, Any]:
        """
        Generate a batch of vanity addresses
        
        Args:
            batch_size: Number of private keys to process
            steps_per_thread: Number of addresses to check per thread (walker mode)
            target_pattern: Pattern to search for. Formats:
                           - "888" = head pattern only (0x888...)
                           - ",abc" = tail pattern only (...abc)
                           - "888,abc" = both head and tail (0x888...abc)
            use_walker: Use walker kernel for better performance
            
        Returns:
            Dictionary with results and statistics
        """
        # Parse target pattern (head,tail format)
        if "," in target_pattern:
            head_pattern, tail_pattern = target_pattern.split(",", 1)
        else:
            head_pattern = target_pattern
            tail_pattern = ""
        
        # Validate patterns
        if head_pattern and not all(c in "0123456789abcdef" for c in head_pattern.lower()):
            raise ValueError(f"Invalid head hex pattern: {head_pattern}")
        if tail_pattern and not all(c in "0123456789abcdef" for c in tail_pattern.lower()):
            raise ValueError(f"Invalid tail hex pattern: {tail_pattern}")
        
        if not head_pattern and not tail_pattern:
            raise ValueError("At least one of head or tail pattern must be provided")
        
        # Generate private keys
        print(f"Generating {batch_size} private keys...")
        privkeys = generate_valid_privkeys(batch_size,steps_per_thread,128)
        # privkeys = [bytes.fromhex("801b58f6029d6514ac85f20db88f919b4b26fc3b72128c379cd7f7f790974c61")]
        # for i in range(len(privkeys)):
        #     print(f"privkeys[{i}] = {privkeys[i].hex()}")
        # Run GPU kernel
        middle_gpu_time=0
        if use_walker:
            print('use walker')
            indices, gpu_time, middle_gpu_time = self.cuda_generator.generate_vanity_walker(
                privkeys,
                steps_per_thread=steps_per_thread,
                head_pattern=head_pattern,
                tail_pattern=tail_pattern
            )
            addresses_checked = batch_size * steps_per_thread
            all_addresses = None  # Walker kernel doesn't support address output yet
        else:
            indices, gpu_time, all_addresses = self.cuda_generator.generate_vanity_simple(
                privkeys,
                head_pattern=head_pattern,
                tail_pattern=tail_pattern
            )
            addresses_checked = batch_size
        
        # Update statistics
        self.stats["total_keys_processed"] += batch_size
        self.stats["total_addresses_checked"] += addresses_checked
        self.stats["total_matches_found"] += len(indices)
        self.stats["total_gpu_time"] += gpu_time
        
        # Process results
        results = {
            "matches": [],
            "all_addresses": all_addresses,
            "batch_stats": {
                "keys_processed": batch_size,
                "addresses_checked": addresses_checked,
                "matches_found": len(indices),
                "gpu_time": gpu_time,
                "middle_gpu_time": middle_gpu_time,
                "throughput": addresses_checked / gpu_time / 1e6  # MAddr/s
            }
        }
        
        # Extract matching private keys
        for idx in indices:

            if use_walker:
                key_idx = idx // steps_per_thread
                step = idx % steps_per_thread
            else:
                key_idx = idx
                step = 0

            if key_idx < len(privkeys):
                base = int.from_bytes(privkeys[key_idx], "big")
                k = base + step
                n = SECP256K1_ORDER_INT
                if k >= n:
                    k -= n
                if k == 0:
                    k = 1
                # Calculate actual private key (base_key + step * G)
                base_key = k.to_bytes(32, "big")
                # For simplicity, we store the base key and step
                # In production, you'd compute the actual private key
                results["matches"].append({
                    "private_key": base_key.hex(),
                    "key_index": key_idx,
                    "step": step,
                    "head_pattern": head_pattern if head_pattern else None,
                    "tail_pattern": tail_pattern if tail_pattern else None,
                    "pattern": target_pattern
                })
        
        # Print addresses only when not using walker mode
        if not use_walker and all_addresses is not None:
            pass
            # print(f"\nAddresses from batch:")
            # for i in range(min(10, batch_size)):  # Show first 10 addresses
            #     addr_bytes = all_addresses[i*20:(i+1)*20]
            #     addr_hex = "0x" + addr_bytes.tobytes().hex()
            #     privkey_hex = privkeys[i].hex()
            #     print(f"  privkey[{i}]: {privkey_hex}")
            #     print(f"  address[{i}]: {addr_hex}")
        
        return results
    
    def run_continuous(self,
                       target_pattern: str = "888",
                       batch_size: int = 10000,
                       steps_per_thread: int = 256,
                       useWalker: bool = True):
        """
        Run continuous vanity address generation (serial version)
        
        Args:
            target_pattern: Hex pattern to search for
            batch_size: Number of keys per batch
            steps_per_thread: Steps per thread for walker kernel
        """
        print(f"Starting continuous generation for pattern: {target_pattern}")
        print(f"Batch size: {batch_size}, Steps per thread: {steps_per_thread}")
        print(f"Press Ctrl+C to stop\n")
        
        all_matches = []
        batch_count = 0
        
        try:
            while True:
                
                # Generate batch
                batch_count += 1
                results = self.generate_batch(
                    batch_size=batch_size,
                    steps_per_thread=steps_per_thread,
                    target_pattern=target_pattern,
                    use_walker=useWalker
                )
                
                # Store matches
                all_matches.extend(results["matches"])
                
                # Print progress
                avg_throughput = self.stats["total_addresses_checked"] / self.stats["total_gpu_time"] / 1e6
                
                print(f"Batch {batch_count}: "
                      f"Found {len(results['matches'])} matches, "
                      f"Total: {self.stats['total_matches_found']}, "
                      f"Throughput: {results['batch_stats']['throughput']:.2f} MAddr/s, "
                      f"Avg: {avg_throughput:.2f} MAddr/s, "
                      f"GPU Time: {results['batch_stats']['gpu_time']:.2f} seconds, "
                      f"Middle GPU Time: {results['batch_stats']['middle_gpu_time']:.2f} seconds")
                

                # Save results periodically
                if len(all_matches) > 0:
                    print(f"ans = {all_matches[0]}")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nStopping generation...")
        
      
        
        # Print final statistics
        self._print_final_stats()
        
    def run_continuous_parallel(self,
                               target_pattern: str = "888",
                               batch_size: int = 10000,
                               steps_per_thread: int = 256,
                               useWalker: bool = True):
        """
        Run continuous vanity address generation with CPU-GPU parallelism
        
        Args:
            target_pattern: Hex pattern to search for
            batch_size: Number of keys per batch
            steps_per_thread: Steps per thread for walker kernel
        """
        print(f"Starting PARALLEL generation for pattern: {target_pattern}")
        print(f"Batch size: {batch_size}, Steps per thread: {steps_per_thread}")
        print(f"Press Ctrl+C to stop\n")
        
        # Start background private key generator
        privkey_generator = PrivkeyGenerator(batch_size, steps_per_thread)
        privkey_generator.start()
        
        all_matches = []
        batch_count = 0
        total_cpu_time = 0.0
        total_queue_wait_time = 0.0
        
        try:
            # Wait for first batch to be ready
            print("Waiting for first batch of private keys...")
            privkeys, cpu_time = privkey_generator.get_privkeys(timeout=10.0)
            if privkeys is None:
                print("Timeout waiting for first batch")
                return
            
            while True:
                queue_wait_start = time.perf_counter()
                
                # Start generating next batch asynchronously while processing current
                if privkey_generator.queue_size() == 0:
                    print("Warning: GPU waiting for CPU (queue empty)")
                
                # Process current batch on GPU
                batch_count += 1
                
                # Parse target pattern
                if "," in target_pattern:
                    head_pattern, tail_pattern = target_pattern.split(",", 1)
                else:
                    head_pattern = target_pattern
                    tail_pattern = ""
                
                # Run GPU kernel
                middle_gpu_time = 0
                if useWalker:
                    indices, gpu_time,middle_gpu_time = self.cuda_generator.generate_vanity_walker(
                        privkeys,
                        steps_per_thread=steps_per_thread,
                        head_pattern=head_pattern,
                        tail_pattern=tail_pattern
                    )
                    addresses_checked = batch_size * steps_per_thread
                else:
                    indices, gpu_time, all_addresses = self.cuda_generator.generate_vanity_simple(
                        privkeys,
                        head_pattern=head_pattern,
                        tail_pattern=tail_pattern
                    )
                    addresses_checked = batch_size
                
                # Update statistics
                self.stats["total_keys_processed"] += batch_size
                self.stats["total_addresses_checked"] += addresses_checked
                self.stats["total_matches_found"] += len(indices)
                self.stats["total_gpu_time"] += gpu_time
                total_cpu_time += cpu_time
                
                # Get next batch (non-blocking, should be ready)
                next_privkeys, next_cpu_time = privkey_generator.get_privkeys(timeout=0.1)
                queue_wait_time = time.perf_counter() - queue_wait_start
                total_queue_wait_time += queue_wait_time
                
                # Process results for current batch
                results = {"matches": [],"batch_stats": {
                    "keys_processed": batch_size,
                    "addresses_checked": addresses_checked,
                    "matches_found": len(indices),
                    "gpu_time": gpu_time,
                    "cpu_time": cpu_time,
                    "middle_gpu_time": middle_gpu_time,
                    "queue_wait_time": queue_wait_time,
                    "throughput": addresses_checked / gpu_time / 1e6
                }}
                
                # Extract matching private keys
                for idx in indices:
                    if useWalker:
                        key_idx = idx // steps_per_thread
                        step = idx % steps_per_thread
                    else:
                        key_idx = idx
                        step = 0

                    if key_idx < len(privkeys):
                        base = int.from_bytes(privkeys[key_idx], "big")
                        k = base + step
                        n = SECP256K1_ORDER_INT
                        if k >= n:
                            k -= n
                        if k == 0:
                            k = 1
                        base_key = k.to_bytes(32, "big")
                        results["matches"].append({
                            "private_key": base_key.hex(),
                            "key_index": key_idx,
                            "step": step,
                            "head_pattern": head_pattern if head_pattern else None,
                            "tail_pattern": tail_pattern if tail_pattern else None,
                            "pattern": target_pattern
                        })
                
                # Store matches
                all_matches.extend(results["matches"])
                
                # Print progress with parallel statistics
                avg_throughput = self.stats["total_addresses_checked"] / self.stats["total_gpu_time"] / 1e6
                avg_cpu_time = total_cpu_time / batch_count
                avg_queue_wait = total_queue_wait_time / batch_count
                cpu_gpu_ratio = total_cpu_time / self.stats["total_gpu_time"] if self.stats["total_gpu_time"] > 0 else 0
                
                print(f"Batch {batch_count}: "
                      f"Found {len(results['matches'])} matches, "
                      f"Total: {self.stats['total_matches_found']}, "
                      f"Throughput: {results['batch_stats']['throughput']:.2f} MAddr/s, "
                      f"Avg: {avg_throughput:.2f} MAddr/s")
                print(f"    Timing - GPU: {gpu_time:.3f}s, CPU: {cpu_time:.3f}s, "
                      f"Queue wait: {queue_wait_time:.3f}s, Queue size: {privkey_generator.queue_size()}")
                print(f"    Avg - CPU: {avg_cpu_time:.3f}s, Queue wait: {avg_queue_wait:.3f}s, "
                      f"CPU/GPU ratio: {cpu_gpu_ratio:.2f}")
                
                # Check if we found matches
                if len(all_matches) > 0:
                    print(f"ans = {all_matches[0]}")
                    break
                
                # Use next batch
                if next_privkeys is not None:
                    privkeys, cpu_time = next_privkeys, next_cpu_time
                else:
                    # Fallback: wait for next batch
                    print("GPU waiting for next batch...")
                    privkeys, cpu_time = privkey_generator.get_privkeys(timeout=5.0)
                    if privkeys is None:
                        print("Timeout waiting for next batch")
                        break
                    
        except KeyboardInterrupt:
            print("\n\nStopping generation...")
        finally:
            privkey_generator.stop()
            privkey_generator.join(timeout=1.0)
        
        # Print final statistics with parallel info
        self._print_final_stats_parallel(total_cpu_time, total_queue_wait_time)
    
    def _print_final_stats(self):
        """Print final statistics"""
        elapsed = time.time() - self.stats["start_time"]
        
        print("\n" + "="*60)
        print("Final Statistics:")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  GPU time: {self.stats['total_gpu_time']:.2f} seconds")
        print(f"  Keys processed: {self.stats['total_keys_processed']:,}")
        print(f"  Addresses checked: {self.stats['total_addresses_checked']:,}")
        print(f"  Matches found: {self.stats['total_matches_found']:,}")
        
        if self.stats["total_gpu_time"] > 0:
            avg_throughput = self.stats["total_addresses_checked"] / self.stats["total_gpu_time"] / 1e6
            print(f"  Average throughput: {avg_throughput:.2f} MAddr/s")
        
        if self.stats["total_matches_found"] > 0:
            probability = self.stats["total_matches_found"] / self.stats["total_addresses_checked"]
            print(f"  Match probability: {probability:.6e}")
    
    def _print_final_stats_parallel(self, total_cpu_time: float, total_queue_wait_time: float):
        """Print final statistics for parallel execution"""
        elapsed = time.time() - self.stats["start_time"]
        
        print("\n" + "="*60)
        print("Final Statistics (Parallel Execution):")
        print(f"  Total time: {elapsed:.2f} seconds")
        print(f"  GPU time: {self.stats['total_gpu_time']:.2f} seconds")
        print(f"  CPU time: {total_cpu_time:.2f} seconds")
        print(f"  Queue wait time: {total_queue_wait_time:.2f} seconds")
        print(f"  Keys processed: {self.stats['total_keys_processed']:,}")
        print(f"  Addresses checked: {self.stats['total_addresses_checked']:,}")
        print(f"  Matches found: {self.stats['total_matches_found']:,}")
        
        if self.stats["total_gpu_time"] > 0:
            avg_throughput = self.stats["total_addresses_checked"] / self.stats["total_gpu_time"] / 1e6
            print(f"  Average throughput: {avg_throughput:.2f} MAddr/s")
            
            # Parallel efficiency metrics
            cpu_gpu_ratio = total_cpu_time / self.stats["total_gpu_time"]
            gpu_utilization = self.stats["total_gpu_time"] / elapsed * 100
            parallel_efficiency = (self.stats["total_gpu_time"] + total_cpu_time) / elapsed
            
            print(f"  CPU/GPU time ratio: {cpu_gpu_ratio:.2f}")
            print(f"  GPU utilization: {gpu_utilization:.1f}%")
            print(f"  Parallel efficiency: {parallel_efficiency:.2f}")
            print(f"  Average queue wait: {total_queue_wait_time/elapsed*1000:.1f}ms per second")
        
        if self.stats["total_matches_found"] > 0:
            probability = self.stats["total_matches_found"] / self.stats["total_addresses_checked"]
            print(f"  Match probability: {probability:.6e}")


def main():
    """Main entry point"""
    # Configuration variables
    pattern = "000000,000000"  # Can be "888" for head, ",abc" for tail, or "888,abc" for both
    batch_size = 4096*32
    steps = 512*8
    device = 0
    useWalker = True
    use_parallel = True  # Use parallel CPU-GPU processing
    # Initialize generator
    generator = VanityAddressGenerator(device_id=device)
    
    
    # Run continuous generation
    if use_parallel:
        generator.run_continuous_parallel(
            target_pattern=pattern,
            batch_size=batch_size,
            steps_per_thread=steps,
            useWalker=useWalker
        )
    else:
        generator.run_continuous(
            target_pattern=pattern,
            batch_size=batch_size,
            steps_per_thread=steps,
            useWalker=useWalker
        )


if __name__ == "__main__":
    main()