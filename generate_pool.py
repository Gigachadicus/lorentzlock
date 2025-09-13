#!/usr/bin/env python3
"""
Generate Lorenz pool bytes file for the encryption system.
Uses the existing Lorenz system to generate a large pool of chaotic data.
"""

import argparse
import numpy as np
from lorenz_system import LorenzSystem, LorenzParameters
from lorenz_cipher import quantize_lorenz_state


def generate_pool_file(output_file: str, pool_size_mb: int = 10):
    """Generate Lorenz pool bytes file"""
    print(f"Generating {pool_size_mb}MB pool file: {output_file}")
    
    # Initialize Lorenz system with standard parameters
    params = LorenzParameters(sigma=10.0, rho=28.0, beta=8/3)
    system = LorenzSystem(params)
    
    pool_size_bytes = pool_size_mb * 1024 * 1024
    bytes_per_step = 4  # 4 bytes per quantized state
    total_steps = pool_size_bytes // bytes_per_step
    steps_per_batch = 10000
    
    with open(output_file, 'wb') as f:
        bytes_written = 0
        
        for batch in range(0, total_steps, steps_per_batch):
            current_batch_size = min(steps_per_batch, total_steps - batch)
            
            # Run Lorenz system for this batch
            trajectory = system.run_steps(current_batch_size)
            
            # Quantize each state to bytes
            batch_bytes = bytearray()
            for state in trajectory:
                quantized = quantize_lorenz_state(state, bits=32)
                # Convert to 4 bytes (big-endian)
                batch_bytes.extend(quantized.to_bytes(4, 'big'))
            
            # Write batch to file
            f.write(batch_bytes)
            bytes_written += len(batch_bytes)
            
            # Progress update
            progress = (batch + current_batch_size) / total_steps * 100
            print(f"Progress: {progress:.1f}% ({bytes_written} bytes written)")
    
    print(f"Pool file generated: {output_file} ({bytes_written} bytes)")


def main():
    parser = argparse.ArgumentParser(description='Generate Lorenz pool bytes file')
    parser.add_argument('--output', '-o', default='lorenz_pool.bin', 
                       help='Output pool file path')
    parser.add_argument('--size-mb', type=int, default=10,
                       help='Pool size in MB (default: 10)')
    
    args = parser.parse_args()
    
    generate_pool_file(args.output, args.size_mb)


if __name__ == '__main__':
    main()
