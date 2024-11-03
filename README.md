# IT3105 - Poker Agent (AI) & PID Controller

## Table of Contents

- [Project Description](#project-description)
- [Installation & Usage](#installation--usage)
- [Contact](#contact)

## Project Description

This project is a part of the course IT3105 - Artificial Intelligence Programming at NTNU.
The project consists of two parts: a poker agent and a PID controller.

### PID Controller

This part consists of a system that simulates a environment, for example a cornout system, a bathtub filling with water or something similar.
The point was to implement a PID controller that could control the system. In order to do this I implemented a Neural Net from scratch that would take the error, integral of error and derivatie of error as input.
The output of the neural net would be the control signal that would be sent to the system.
Results from running simulations can be found in [this folder](./P1:%20Pid%20Controller/Project%201%20-%20Lab%20report.pdf).

A lab report can be found [here](./P1:%20Pid%20Controller/Project%201%20-%20Lab%20report.pdf).

### Poker Agent

This part consists of a poker agent that can play Texas Hold'em Poker. The agent creates a game tree and uses neural networks to evaluate leafnodes of the tree. While playing, the algorithm learns more and more about the perfect strategy for playing poker against the current opponents.

Poker is extremely complex and it is impossible to create a perfect poker agent. However, the agent should be able to play decently well against the opponents it has played against during training. The agent should also be able to adapt to new opponents and learn their strategies.

Some simplifications have been made to the game of poker in order to speed up the learning process. For example, the agent only plays heads-up (1v1), the cards in the deck are limited. Also the legal actions are limited to check, bet, fold and all-in.

The project_requirements include detailed information about how the agent works. Including how information is passed between the different parts of the agent, how the agent learns and how the agent plays.

### Graphics

#### PID Controller

Here is a graphic where the neural net is training on the system and decreasing the error over time.

![graphic](./P1:%20Pid%20Controller/plots/StandardPIDController_Bathtub_50_100_0.001_2024-02-01%2013:53:23.321856.png)

#### Poker Agent

Parts of the Agent:

![Agent parts](./P2:%20DeepStack/graphics/parts.png)

Setup screen:

![Setup screen](./P2:%20DeepStack/graphics/setup.png)

Playing screen:

![playing screen](./P2:%20DeepStack/graphics/playing_screen.png)

## Installation & Usage

### PID Controller & Poker Agent

The project can easily be run by running the ConSys.py file in the P1: PID Controller or the main.py file in P2: DeepStack folder.

## Contact

If you have any questions or want to get in touch, you can reach me at:

- [Github](https://github.com/chrvei00)
- [LinkedIn](https://www.linkedin.com/in/christian-veiby)
- [christian@veiby.com](mailto:christian@veiby.com)
