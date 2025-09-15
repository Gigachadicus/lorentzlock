import numpy as np
from scipy.integrate import solve_ivp
import pickle

class LorenzSystem:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.state_pool = None
        self.pool_index = 0
    
    def lorenz_equations(self, t, state):
        x, y, z = state
        dxdt = self.sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - self.beta * z
        return [dxdt, dydt, dzdt]
    
    def generate_states(self, num_samples=100000):
        """Generate deterministic Lorenz states"""
        initial_state = [1.0, 1.0, 1.0]  # Fixed initial condition for reproducibility
        t_span = (0, num_samples * 0.01)
        t_eval = np.linspace(0, t_span[1], num_samples)
        
        print(f"Generating {num_samples} Lorenz states...")
        sol = solve_ivp(self.lorenz_equations, t_span, initial_state, 
                       t_eval=t_eval, method='RK45', rtol=1e-8)
        
        self.state_pool = sol.y.T  # Transpose to get (time, xyz) shape
        print(f"Generated state pool shape: {self.state_pool.shape}")
        return self.state_pool
    
    def get_state_at_index(self, index):
        """Get Lorenz state at specific index (for deterministic access)"""
        if self.state_pool is None:
            raise ValueError("State pool not generated. Call generate_states() first.")
        
        return self.state_pool[index % len(self.state_pool)]
    
    def export_states(self, filename='lorenz_states.pkl'):
        """Export states to file for sharing between master/slave"""
        if self.state_pool is None:
            self.generate_states()
        
        with open(filename, 'wb') as f:
            pickle.dump(self.state_pool, f)
        print(f"Lorenz states exported to {filename}")
    
    def import_states(self, filename='lorenz_states.pkl'):
        """Import states from file"""
        with open(filename, 'rb') as f:
            self.state_pool = pickle.load(f)
        print(f"Lorenz states imported from {filename}")

if __name__ == "__main__":
    # Generate and save Lorenz states for use by master/slave
    lorenz_system = LorenzSystem()
    lorenz_system.generate_states(100000)
    lorenz_system.export_states()
    
    # Test a few states
    print("Sample states:")
    for i in [0, 1000, 10000]:
        state = lorenz_system.get_state_at_index(i)
        print(f"State {i}: x={state[0]:.6f}, y={state[1]:.6f}, z={state[2]:.6f}")
