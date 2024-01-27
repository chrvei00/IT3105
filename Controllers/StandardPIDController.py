import jax

class StandardPIDController:
    def update_params(self, params, grads, learning_rate, direction):
        g_kp = float(grads[0])
        g_ki = float(grads[1])
        g_kd = float(grads[2])
        print(f"g_kp: {g_kp}, g_ki: {g_ki}, g_kd: {g_kd}")

        kp, ki, kd = params

        kp += direction*learning_rate * g_kp
        ki += direction*learning_rate * g_ki
        kd += direction*learning_rate * g_kd
        return (kp, ki, kd)

    # Calculate Output
    def control_action(self, params, error, integral_error, derivate_error):
        kp, ki, kd = params
        return kp * error + ki * integral_error + kd * derivate_error