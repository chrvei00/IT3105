import random as rnd
from Plants.Plant import Plant
class Bathtub(Plant):
    # Initialize the bathtub
    def __init__(self, cross_section, water_level, noise_level):
        self.drain_cross_section = cross_section/100
        self.water_level = water_level
        self.noise_level = noise_level

    # Simulate the dynamics of the bathtub
    def dynamics(self, control_signal):
        # Simulate the noise in draining
        noise = rnd.uniform(-self.noise_level, self.noise_level)
        control_signal = max(min(control_signal, 50), 0)
        # Calculate the new water level
        if control_signal < 0:
            control_signal = 0
        self.water_level += control_signal - self.drain_cross_section * ((9.81*self.water_level)**0.5) + noise
        if self.water_level < 0:
            self.water_level = 0

    def output(self):
        return self.water_level
    
    def input(self, control_signal):
        self.dynamics(control_signal)