class PIDController():
    def update_params(self, params, grads, learning_rate, direction) -> tuple:
        """Update controller params using gradient descent"""
        pass

    def control_signal(self, params, error, integral_error, derivate_error) -> float:
        """Calculate control signal"""
        pass