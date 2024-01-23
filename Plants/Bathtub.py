import numpy as np

class bathtub:
    def bathtub_dynamics(water_level, inflow_rate, drain_rate, noise_level=0.1):
        # Simulate the noise in draining
        noise = np.random.normal(0, noise_level)
        
        # Calculate the new water level
        new_water_level = water_level + inflow_rate - (drain_rate + noise) * water_level
        return max(new_water_level, 0)  # Ensure water level doesn't go below 0
