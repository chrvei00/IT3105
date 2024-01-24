import random as rnd
class Bathtub:
    # Initialize the bathtub
    def __init__(self, water_level, drain_rate, noise_level):
        self.water_level = water_level
        self.drain_rate = drain_rate
        self.noise_level = noise_level

    # Simulate the dynamics of the bathtub
    def bathtub_dynamics(self, inflow_rate):
        # Simulate the noise in draining
        noise = rnd.uniform(-self.noise_level, self.noise_level)
        # Calculate the new water level
        new_water_level = self.water_level + inflow_rate - (self.drain_rate + noise) * self.water_level
        return max(new_water_level, 0)  # Ensure water level doesn't go below 0

    def output(self):
        return self.water_level
    
    def input(self, control_signal):
        self.water_level = self.bathtub_dynamics(control_signal)