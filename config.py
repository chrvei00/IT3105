import inquirer
from configparser import ConfigParser
from Plants.Bathtub import Bathtub
from Controllers.StandardPIDController import StandardPIDController

def configGlobal():
    config = ConfigParser()
    config.read('config.ini')
    return config.getint('Global', 'epochs'), config.getint('Global', 'time_steps'), config.getfloat('Global', 'learning_rate')

def configPlant(plant):
    config = ConfigParser()
    config.read('config.ini')
    new_plant = None
    if plant == "Bathtub":
        new_plant = Bathtub(config.getint('Bathtub', 'waterlevel'), config.getfloat('Bathtub', 'drain_rate'), config.getfloat('Bathtub', 'noise_level'))
    else:
        print("Plant not found")
    return config.getint('Bathtub', 'target_level'), new_plant

def configController(controller):
    config = ConfigParser()
    config.read('config.ini')
    if controller == "StandardPIDController":
        params = (config.getfloat('StandardPIDController', 'Kp'), config.getfloat('StandardPIDController', 'Ki'), config.getfloat('StandardPIDController', 'Kd'))
        return params, config.getint('StandardPIDController', 'direction'), StandardPIDController()

def updateGlobal():
    print("Updating global...")
    config = ConfigParser()
    config.read('config.ini')
    #Check if section exists
    if not config.has_section('Global'):
        config.add_section('Global')
    config.set('Global', 'epochs', input("Epochs: (int) "))
    config.set('Global', 'time_steps', input("Time_steps: (int) "))
    config.set('Global', 'learning_rate', input("Learning_rate: (float) "))
    with open('config.ini', 'w') as f:
        config.write(f)

def updateBathtub():
    print("Updating bathtub...")
    config = ConfigParser()
    config.read('config.ini')
    #Check if section exists
    if not config.has_section('Bathtub'):
        config.add_section('Bathtub')
    config.set('Bathtub', 'waterlevel', input("Waterlevel: (int) "))
    config.set('Bathtub', 'drain_rate', input("Drain_rate: (int) "))
    config.set('Bathtub', 'noise_level', input("Noise_level: (int) "))
    config.set('Bathtub', 'target_level', input("Target_level: (int) "))
    with open('config.ini', 'w') as f:
        config.write(f)

def updateStandardPIDController():
    print("Updating StandardPIDController...")
    config = ConfigParser()
    config.read('config.ini')
    #Check if section exists
    if not config.has_section('StandardPIDController'):
        config.add_section('StandardPIDController')
    config.set('StandardPIDController', 'Kp', input("Kp: (int) "))
    config.set('StandardPIDController', 'Ki', input("Ki: (int) "))
    config.set('StandardPIDController', 'Kd', input("Kd: (int) "))
    config.set('StandardPIDController', 'direction', input("Direction: (int) "))
    with open('config.ini', 'w') as f:
        config.write(f)

def update_config():
    questions = [
        inquirer.List('config',
                    message="Which config do you wish to update?",
                    choices=['Global', 'StandardPIDController', 'Bathtub'],
                ),
    ]
    answers = inquirer.prompt(questions)
    if answers['config'] == "Global":
        updateGlobal()
    elif answers['config'] == "StandardPIDController":
        updateStandardPIDController()
    elif answers['config'] == "Bathtub":
        updateBathtub()
    else:
        print("Config not found")    
    

if __name__ == '__main__':
    update_config()