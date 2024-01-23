import jax.numpy as jnp
from jax import grad, jit

class StandardPIDController:
    def __init__(self, kp, ki, kd, setpoint):
        # PID coefficients
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Inir jax
        self.grad = grad

        # Initial error and integral of error
        self.error_history = []
        self.param_history = [[] for i in range(3)]
        self.curr_MSE = 0

    def update_params(self, dt):
        # Calculate error
        self.curr_MSE = self.compute_MSE(current_value)

        # Update koeficients
        self.kp += self.grad(lambda x: self.compute_MSE(x))(self.kp) * dt
        self.ki += self.grad(lambda x: self.compute_MSE(x))(self.ki) * dt
        self.kd += self.grad(lambda x: self.compute_MSE(x))(self.kd) * dt

        # Append to history
        self.param_history[0].append(self.kp)
        self.param_history[1].append(self.ki)
        self.param_history[2].append(self.kd)

    # Calculate Output
    def calculate_output(self, error, dt):
        # Calculate error
        self.curr_MSE = self.compute_MSE(current_value)

        # Update integral of error
        self.integral_error += error * dt

        # PID output
        return (self.kp * error) + (self.ki * self.integral_error) + (self.kd * derivative_error)

    # Calculate MSE
    def compute_MSE(self, error):
        return jnp.mean(jnp.square(self.error_history))

    # Append error to history
    def append_error(self, error):
        self.error_history.append(error)

    # Output control action
    def get_control_action(self, error, dt):
        # Append error to history
        self.append_error(error)

        # Update parameters
        self.update_params(dt)

        # Calculate output
        return self.calculate_output(error, dt)