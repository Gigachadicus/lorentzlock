import numpy as np
from scipy.integrate import solve_ivp
from lorenz_cipher import quantize_lorenz_state


class LorenzParameters:
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta


class LorenzSystem:
    def __init__(self, params: LorenzParameters, dt=0.01, initial_state=[1.0, 1.0, 1.0]):
        self.params = params
        self.dt = float(dt)
        self.initial_state = np.array(initial_state, dtype=float)
        self.state_history = None
        self.t = 0.0

    def lorenz_equations(self, t, state):
        x, y, z = state
        dx = self.params.sigma * (y - x)
        dy = x * (self.params.rho - z) - y
        dz = x * y - self.params.beta * z
        return [dx, dy, dz]

    def run_steps(self, steps: int):
        t_span = (self.t, self.t + self.dt * steps)
        t_eval = np.linspace(*t_span, steps)

        solution = solve_ivp(
            fun=self.lorenz_equations,
            t_span=t_span,
            y0=self.initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-9,
            atol=1e-9,
        )
        self.state_history = solution.y.T
        self.initial_state = self.state_history[-1]
        self.t += (steps * self.dt)
        return self.state_history


def generate_pool_bytes(size_mb: int = 10) -> bytes:
    """Generate Lorenz pool bytes in memory"""
    print(f"Generating {size_mb}MB pool in memory...")
    
    # Initialize Lorenz system
    params = LorenzParameters(sigma=10.0, rho=28.0, beta=8/3)
    system = LorenzSystem(params)
    
    pool_size_bytes = size_mb * 1024 * 1024
    bytes_per_step = 4
    total_steps = pool_size_bytes // bytes_per_step
    steps_per_batch = 10000
    
    pool_data = bytearray()
    
    for batch in range(0, total_steps, steps_per_batch):
        current_batch_size = min(steps_per_batch, total_steps - batch)
        
        # Run Lorenz system
        trajectory = system.run_steps(current_batch_size)
        
        # Quantize states to bytes
        for state in trajectory:
            quantized = quantize_lorenz_state(state, bits=32)
            pool_data.extend(quantized.to_bytes(4, 'big'))
    
    print(f"Generated {len(pool_data)} bytes of Lorenz pool data")
    return bytes(pool_data)


def save_pool_bytes(pool_bytes: bytes, filename: str):
    """Save pool bytes to file"""
    with open(filename, 'wb') as f:
        f.write(pool_bytes)
    print(f"Pool bytes saved to {filename}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Lorenz pool bytes')
    parser.add_argument('--output', '-o', default='lorenz_pool.bin', 
                       help='Output pool file path')
    parser.add_argument('--size-mb', type=int, default=10,
                       help='Pool size in MB (default: 10)')
    
    args = parser.parse_args()
    
    # Generate and save pool bytes
    pool_data = generate_pool_bytes(args.size_mb)
    save_pool_bytes(pool_data, args.output)
