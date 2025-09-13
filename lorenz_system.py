# lorenz_system.py - Optimized Lorenz Chaos Generator
import numpy as np
import hashlib
import struct
import os


class LorenzParameters:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta


class LorenzSystem:
    def __init__(self, params: LorenzParameters, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
        self.params = params
        self.dt = float(dt)
        self.state = np.array(initial_state, dtype=float)
        self.t = 0.0

    def lorenz_equations(self, t, state):
        x, y, z = state
        dx = self.params.sigma * (y - x)
        dy = x * (self.params.rho - z) - y
        dz = x * y - self.params.beta * z
        return [dx, dy, dz]

    def run_steps_fast(self, steps: int):
        """Optimized Euler integration for speed"""
        dt = self.dt
        sigma, rho, beta = self.params.sigma, self.params.rho, self.params.beta
        
        # Pre-allocate array
        trajectory = np.zeros((steps, 3))
        x, y, z = self.state
        
        # Fast Euler integration loop
        for i in range(steps):
            # Lorenz equations
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            # Update state
            x += dt * dx
            y += dt * dy
            z += dt * dz
            
            # Store
            trajectory[i] = [x, y, z]
        
        # Update internal state
        self.state = np.array([x, y, z])
        self.t += steps * dt
        
        return trajectory


def quantize_lorenz_state_fast(trajectory: np.ndarray) -> np.ndarray:
    """Fast vectorized quantization using the formula: q(v) = int(|v| * 10^6) mod 256"""
    # Apply the quantization formula to each coordinate
    x_quantized = (np.abs(trajectory[:, 0]) * 1e6).astype(np.uint64) % 256
    y_quantized = (np.abs(trajectory[:, 1]) * 1e6).astype(np.uint64) % 256
    z_quantized = (np.abs(trajectory[:, 2]) * 1e6).astype(np.uint64) % 256
    
    # Combine into single bytes (you could use x, y, z separately or combine them)
    # Here we'll return all three as separate bytes
    combined = np.column_stack([x_quantized, y_quantized, z_quantized])
    return combined.astype(np.uint8)


def generate_pool_bytes(size_mb: int = 1) -> bytes:
    """Generate chaotic pool bytes using Lorenz system"""
    print(f"Generating {size_mb}MB Lorenz chaos pool...")
    
    # Initialize Lorenz system
    params = LorenzParameters(sigma=10.0, rho=28.0, beta=8.0/3.0)
    system = LorenzSystem(params, dt=0.001, initial_state=[1.0, 1.0, 1.0])
    
    pool_size_bytes = size_mb * 1024 * 1024
    bytes_per_step = 3  # x, y, z quantized values
    total_steps = pool_size_bytes // bytes_per_step
    
    steps_per_batch = 50000
    pool_data = bytearray()
    
    print(f"Total steps: {total_steps}, batch size: {steps_per_batch}")
    
    batch_count = 0
    for batch_start in range(0, total_steps, steps_per_batch):
        current_batch_size = min(steps_per_batch, total_steps - batch_start)
        
        # Run Lorenz system
        trajectory = system.run_steps_fast(current_batch_size)
        
        # Quantize using the specified formula
        quantized_values = quantize_lorenz_state_fast(trajectory)
        
        # Flatten and convert to bytes
        batch_bytes = quantized_values.flatten().tobytes()
        pool_data.extend(batch_bytes[:current_batch_size * 3])
        
        batch_count += 1
        if batch_count % 10 == 0:
            progress = len(pool_data) / pool_size_bytes * 100
            print(f"Progress: {progress:.1f}% ({len(pool_data)} bytes)")
    
    # Ensure exact size
    result = bytes(pool_data[:pool_size_bytes])
    print(f"Generated {len(result)} bytes of Lorenz chaos pool")
    
    # Store final state for seed derivation
    final_state = system.state
    print(f"Final Lorenz state: X={final_state[0]:.6f}, Y={final_state[1]:.6f}, Z={final_state[2]:.6f}")
    
    return result


def save_pool_bytes(pool_bytes: bytes, filename: str):
    """Save pool bytes to file"""
    with open(filename, 'wb') as f:
        f.write(pool_bytes)
    print(f"Pool bytes saved to {filename}")


def load_pool_bytes(filename: str) -> bytes:
    """Load pre-generated pool bytes"""
    with open(filename, 'rb') as f:
        return f.read()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Lorenz chaos pool')
    parser.add_argument('--output', '-o', default='lorenz_pool.bin', 
                       help='Output pool file')
    parser.add_argument('--size-mb', type=int, default=1,
                       help='Pool size in MB')
    
    args = parser.parse_args()
    
    # Generate and save pool
    pool_data = generate_pool_bytes(args.size_mb)
    save_pool_bytes(pool_data, args.output)
