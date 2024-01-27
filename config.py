import inquirer
from configparser import ConfigParser
from Plants.Bathtub import Bathtub
from Plants.Cournot import Cournot
from Plants.Population import Population
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
        new_plant = Bathtub(config.getfloat('Bathtub', 'cross_section'), config.getfloat('Bathtub', 'waterlevel'), config.getfloat('Bathtub', 'noise_level'))
        target_level = config.getfloat('Bathtub', 'target_level')
    elif plant == "Cournot":
        new_plant = Cournot(config.getfloat('Cournot', 'q0'), config.getfloat('Cournot', 'q1'), config.getfloat('Cournot', 'noise_level'), config.getfloat('Cournot', 'p_max'), config.getfloat('Cournot', 'c_m'))
        target_level = config.getfloat('Cournot', 'target_level')
    elif plant == "Population":
        new_plant = Population(config.getfloat('Population', 'stock'), config.getfloat('Population', 'hunter_rate'), config.getfloat('Population', 'growth_rate'), config.getfloat('Population', 'noise_level'))
        target_level = config.getfloat('Population', 'target_level')
    else:
        print("Plant not found")
    return target_level, new_plant

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
    config.set('Bathtub', 'cross_section', input("Cross_section: (int) "))
    config.set('Bathtub', 'noise_level', input("Noise_level: (int) "))
    config.set('Bathtub', 'target_level', input("Target_level: (int) "))
    with open('config.ini', 'w') as f:
        config.write(f)

def updateCournot():
    print("Updating cournot...")
    config = ConfigParser()
    config.read('config.ini')
    #Check if section exists
    if not config.has_section('Cournot'):
        config.add_section('Cournot')
    config.set('Cournot', 'q0', input("q0: (float) (0-1) "))
    config.set('Cournot', 'q1', input("q1: (float) (0-1) "))
    config.set('Cournot', 'noise_level', input("Noise_level: (float) "))
    config.set('Cournot', 'p_max', input("p_max: (float) "))
    config.set('Cournot', 'c_m', input("c_m: (float) "))
    config.set('Cournot', 'target_level', input("Target_level: (float) "))
    with open('config.ini', 'w') as f:
        config.write(f)

def updatePopulation():
    print("Updating population...")
    config = ConfigParser()
    config.read('config.ini')
    #Check if section exists
    if not config.has_section('Population'):
        config.add_section('Population')
    config.set('Population', 'stock', input("Stock: (int) "))
    config.set('Population', 'hunter_rate', input("Hunter_rate: (float) "))
    config.set('Population', 'growth_rate', input("Growth_rate: (float) "))
    config.set('Population', 'noise_level', input("Noise_level: (float) "))
    config.set('Population', 'target_level', input("Target_level: (float) "))
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
                    choices=['Global', 'StandardPIDController', 'Bathtub', 'Cournot', 'Population'],
                ),
    ]
    answers = inquirer.prompt(questions)
    if answers['config'] == "Global":
        updateGlobal()
    elif answers['config'] == "StandardPIDController":
        updateStandardPIDController()
    elif answers['config'] == "Bathtub":
        updateBathtub()
    elif answers['config'] == "Cournot":
        updateCournot()
    elif answers['config'] == "Population":
        updatePopulation()
    else:
        print("Config not found")    
    

if __name__ == '__main__':
    update_config()