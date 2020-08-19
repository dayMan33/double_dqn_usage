
Double DQN usage example
=======
A simple usage example of the double_DQN libray

Contents
--------

* [Description](#description)
* [Instalation](#installation)
* [Usage](#usage)
* [Support](#support)

description
-----------
This is an example explaining how to use the [double DQN](https://github.com/dayMan33/double_DQN.git) library.
In this example, it is used to learn a policy in a simple environment in which each state is a  vector of length 4 with with integers in the range (0,10) inclusive. 
each action swaps 2 coordinates of the state, and the game reaches a terminal state once the state-vector is sorted or more than 60 steps have been taken. 
The environment's rewards is -1 for any (state, action) that does not lead to a terminal state, and 0 if the observed next state is terminal (sorted).
 
The code files in this project include: 
- **simple_env.py:** A simple environment inheriting from dqnENV with all the required methods for training an agent.
- **trainer.py:**  A script creating an agent and training it in the simple environment.
 
installation
--------

### clone
Clone this repository to your local machine using 'repository address goes here'
            
    git clone https://github.com/dayMan33/double_dqn_usage.git

### setup 
while in the project directory, run setup.sh to install all requirements.

    double_dqn> setup.sh

usage
-----
Check out the code in trainer, which trains an agent and evaluates it against a non trained agent. 
Feel free to play around with the parameters and try it yourself. 

support
-------
For any questions or comments, feel free to email me at danielrotem33@gmailcom.


