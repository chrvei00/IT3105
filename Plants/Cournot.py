import random as rnd
from Plants.Plant import Plant

class Cournot(Plant):
    def __init__(self, q0, q1, noise_level, p_max, c_m):
        if q0 < 0 or q1 < 0 or q1 > 1 or q0 > 1:
            raise ValueError("Quantity must be between 0 and 1")
        self.q0 = q0
        self.q1 = q1
        self.noise_level = noise_level
        self.p_max = p_max
        self.c_m = c_m
        self.p0 = 0        
    
    def dynamics(self, control_signal):
        # Simulate the noise
        noise = rnd.uniform(-self.noise_level, self.noise_level)
        control_signal = max(min(control_signal, 1), -1)
        # Calculate the new quantity
        self.q0 = max(min(self.q0 + control_signal, 1), 0)
        self.q1 = max(min(self.q1 + noise, 1), 0)
        # Calculate the new profit
        self.p0 = self.q0 * ((self.p_max - self.q0 - self.q1) - self.c_m*self.p_max)


    def output(self):
        return self.p0

    def input(self, control_signal):
        self.dynamics(control_signal)
