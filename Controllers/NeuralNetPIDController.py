import jax
import jax.numpy as jnp
import numpy as np
class NeuralNetPIDController():

    def __init__(self):
        self.batched_predict = jax.vmap(self.predict, in_axes=(None, 0))

    def gen_jaxnet_params(self, layers):
        sender = layers[0]; params = []
        for receiver in layers[1:]:
            weights = np.random.uniform(-.1, .1, (sender, receiver))
            biases = np.random.uniform(-.1, .1, (1, receiver))
            sender = receiver
            params.append([weights, biases])
        return params

    def predict(self, params, features):
        def sigmoid(x):
            return 1 / (1 + jnp.exp(-x))
        for weights, biases in params:
            activations = sigmoid(jnp.dot(features, weights) + biases)
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