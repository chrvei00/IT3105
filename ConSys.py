import configparser
import inquirer
import jax.numpy as jnp
from Controllers.StandardPIDController import StandardPIDController
from Plants.TestPlant import TestPlant

def run_simulation(plant, controller):
    print ("Running simulation...")
    for i in range(epochs):
        #Get control action from controller
        control_action = controller.get_control_action(plant.process_value)
        #Apply control action to plant
        plant.apply_control_action(control_action)
        #Update plant state
        plant.update()

def print_res(res):
    print("Simulation results:")
    print(f"Final time: {res['time']}")
    print(f"Final process value: {res['process_value']}")
    print(f"Final control action: {res['control_action']}")

def main():
    #Get input from user
    questions = [
        inquirer.List('controller',
                    message="Which Controller do you wish to use?",
                    choices=['StandardPIDController', 'NeuralPIDController'],
                ),
        inquirer.List('plant',
                    message="Which Plant do you wish to use?",
                    choices=['Bathtub', 'Economics', 'TestPlant'],
                ),
    ]
    answers = inquirer.prompt(questions)
    #Inject config params
    config = configparser.ConfigParser()
    config.read('config.ini')
    epochs = config.getint('test', 'epochs')
    #Initialize plant and controller
    plant = TestPlant()
    controller = StandardPIDController()
    #Run simulation
    res = run_simulation(plant, controller, epochs)
    #Print results
    print_res(res)


if __name__ == '__main__':
    main()