#!/usr/bin/env python3
"""
Main entry point for CUDA-based Ethereum vanity address generator
Compatible with the existing vanity address generation workflow
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vanity_cuda.cuda_vanity import CudaVanity
from vanity.privkey_gen import generate_private_keys

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
            target_pattern: Hex pattern to search for (e.g., "888" for 0x888...)
            use_walker: Use walker kernel for better performance
            
        Returns:
            Dictionary with results and statistics
        """
        # Parse target pattern
        if not all(c in "0123456789abcdef" for c in target_pattern.lower()):
            raise ValueError(f"Invalid hex pattern: {target_pattern}")
        
        target_nibble = int(target_pattern[0], 16) if target_pattern else 0x8
        nibble_count = len(target_pattern)
        
        # Generate private keys
        print(f"Generating {batch_size} private keys...")
        privkeys = generate_private_keys(batch_size)
        privkeys = [bytes.fromhex("801b58f6029d6514ac85f20db88f919b4b26fc3b72128c379cd7f7f790974c61")]
        # for i in range(len(privkeys)):
        #     print(f"privkeys[{i}] = {privkeys[i].hex()}")
        # Run GPU kernel
        
        if use_walker:
            indices, gpu_time = self.cuda_generator.generate_vanity_walker(
                privkeys,
                steps_per_thread=steps_per_thread,
                target_nibble=target_nibble,
                nibble_count=nibble_count
            )
            addresses_checked = batch_size * steps_per_thread
            all_addresses = None  # Walker kernel doesn't support address output yet
        else:
            indices, gpu_time, all_addresses = self.cuda_generator.generate_vanity_simple(
                privkeys,
                target_nibble=target_nibble,
                nibble_count=nibble_count
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
                # Calculate actual private key (base_key + step * G)
                base_key = privkeys[key_idx]
                # For simplicity, we store the base key and step
                # In production, you'd compute the actual private key
                results["matches"].append({
                    "private_key": base_key.hex(),
                    "key_index": key_idx,
                    "step": step,
                    "pattern": f"0x{target_pattern}"
                })
        
        # Print addresses only when not using walker mode
        if not use_walker and all_addresses is not None:
            print(f"\nAddresses from batch:")
            for i in range(min(10, batch_size)):  # Show first 10 addresses
                addr_bytes = all_addresses[i*20:(i+1)*20]
                addr_hex = "0x" + addr_bytes.tobytes().hex()
                privkey_hex = privkeys[i].hex()
                print(f"  privkey[{i}]: {privkey_hex}")
                print(f"  address[{i}]: {addr_hex}")
        
        return results
    
    def run_continuous(self,
                       target_pattern: str = "888",
                       batch_size: int = 10000,
                       steps_per_thread: int = 256,
                       useWalker: bool = True):
        """
        Run continuous vanity address generation
        
        Args:
            target_pattern: Hex pattern to search for
            batch_size: Number of keys per batch
            steps_per_thread: Steps per thread for walker kernel
            save_interval: Save results every N matches
        """
        print(f"Starting continuous generation for pattern 0x{target_pattern}")
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
                      f"Avg: {avg_throughput:.2f} MAddr/s")
                

                # Save results periodically
                if len(all_matches) > 0:
                    print(f"ans = {all_matches[0]}")
                    
        except KeyboardInterrupt:
            print("\n\nStopping generation...")
        
      
        
        # Print final statistics
        self._print_final_stats()
    
   
    
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


def main():
    """Main entry point"""
    # Configuration variables
    pattern = "888"
    batch_size = 1
    steps = 1
    device = 0
    benchmark = False
    useWalker = False
    # Initialize generator
    generator = VanityAddressGenerator(device_id=device)
    
    if benchmark:
        # Run benchmark
        print("Running benchmark...")
        generator.cuda_generator.benchmark(
            num_keys=batch_size,
            steps_per_thread=steps
        )
    else:
        # Run continuous generation
        generator.run_continuous(
            target_pattern=pattern,
            batch_size=batch_size,
            steps_per_thread=steps,
            useWalker=useWalker
        )


if __name__ == "__main__":
    main()