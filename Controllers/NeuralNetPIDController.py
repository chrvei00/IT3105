import jax
import jax.numpy as jnp
import numpy as np
class NeuralNetPIDController():

    def __init__(self, activation_function):
        self.activation_function = activation_function

    def gen_jaxnet_params(self, layers, factor):
        sender = layers[0]; params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(-factor, factor, (sender, receiver))
            biases = np.random.uniform(-factor, factor, (1, receiver))
            sender = receiver
            params.append([weights, biases])
        return params

    def predict(self, params, features):
        for weights, biases in params:
            activations = self.activation_function(jnp.dot(features, weights) + biases)
            features = activations
        return activations
    
    def update_params(self, params, grads, learning_rate, direction):
        updated_params = []
        for (w, b), (dw, db) in zip(params, grads):
            new_w = w + direction * learning_rate * dw
            new_b = b + direction * learning_rate * db
            updated_params.append((new_w, new_b))
        return updated_params
    
    def control_action(self, params, error, integral_error, derivate_error):
        return self.predict(params, jnp.array([[error, integral_error, derivate_error]]))[0][0]

    def print_params(self, params):
        for i, (w, b) in enumerate(params):
            print(f"Layer {i + 1}:")
            print(f"Weights:\n{w}")
            print(f"Biases:\n{b}")
            print()