import random as rnd
from Plants.Plant import Plant

class Population(Plant):
    def __init__(self, stock, hunter_rate,  growth_factor, noise_level):
        self.stock = stock
        self.hunter_rate = hunter_rate
        self.growth_factor = growth_factor
        self.noise_level = noise_level

    def dynamics(self, control_signal):
        # Simulate the noise
        noise = rnd.uniform(-self.noise_level, self.noise_level)
        # Maximum hunters are 100, and minimum are 0
        control_signal = max(min(control_signal, 100), 0)
        # Calculate the new stock
        self.stock = max((self.stock+self.stock*self.growth_factor-control_signal*self.stock*self.hunter_rate + noise), 0)

    def output(self):
        return self.stock

    def input(self, control_signal):
        self.dynamics(control_signal)
