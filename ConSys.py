import jax
import jax.numpy as jnp
import numpy as np
import inquirer
from Assets.plots import plot_pid_mse
from config import configGlobal, configPlant, configController, update_config

class ConSys:

    def __init__(self, plant, params, direction, controller, epochs, timesteps, learning_rate):
        #Init configs
        self.plant_name = plant
        self.direction = direction
        self.controller = controller
        self.epochs = epochs
        self.timesteps = timesteps
        self.learning_rate = learning_rate
        #Init history
        self.mse_history = []
        self.param_history = [params]
        #Init gradient function
        self.grad_fn = jax.value_and_grad(self.run_epoch, argnums=0)
        #Init plant
        self.target_level, self.plant = configPlant(self.plant_name)

    def run_timestep(self, params, errors):
        #Get control action from controller
        control_action = self.controller.control_action(params, errors[-1], sum(errors), errors[-2]-errors[-1])
        #Apply control action to plant
        self.plant.input(control_action)
        #Calculate error
        return self.target_level - self.plant.output()

    def run_epoch(self, params):
        error_history = [0, 0]
        #Run timesteps
        for _ in range(int(self.timesteps) - 1):
            error_history.append(self.run_timestep(params, error_history))
        #Return average error
        return self.calculate_mse(jnp.array(error_history))

    def run_system(self):
        #Run epochs
        for _ in range(int(self.epochs)):
            print(f"Epoch {_ + 1}")
            #Update controller params (kp, ki, kd) using gradient descent
            mse, grads = self.grad_fn(self.param_history[-1])
            self.mse_history.append(np.abs(mse))
            self.param_history.append(self.controller.update_params(self.param_history[-1], grads, self.learning_rate, self.direction))
            print(f"Params: {self.param_history[-1]}")
            print(f"MSE: {mse}\n")
            #Reset plant
            self.plant = configPlant(self.plant_name)[1]
            
        #Plot mse history
        plot_pid_mse(self.mse_history, self.param_history)

    def calculate_mse(self, error_history):
        return jnp.mean(jnp.square(error_history))
    

def run_simulation():
    #Prompt config
    qSimulation = [
        inquirer.List('controller',
                    message="Which Controller do you wish to use?",
                    choices=['StandardPIDController', 'NeuralNetPIDController'],
                ),
        inquirer.List('plant',
                    message="Which Plant do you wish to use?",
                    choices=['Bathtub', 'Cournot', 'Population'],
                ),
    ]
    aSimulation = inquirer.prompt(qSimulation)
    #Initialize plant, controller and global
    epochs, timesteps, learning_rate = configGlobal()
    params, direction, controller = configController(aSimulation['controller'])
    #Run simulation
    ConSys(aSimulation['plant'], params, direction, controller, epochs, timesteps, learning_rate).run_system()    

def main():
    #Get input from user
    qSetup = [
        inquirer.List('setup',
                    message="What do you wish to do?",
                    choices=['Run simulation', 'Update config'],
                ),
    ]
    aSetup = inquirer.prompt(qSetup)
    
    if aSetup['setup'] == 'Run simulation':
        run_simulation()
    elif aSetup['setup'] == 'Update config':
        update_config()
        return
    else:
        print("Error: Invalid input")

if __name__ == "__main__":
    #Run system
    main()
