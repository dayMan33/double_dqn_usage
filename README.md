
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
- **simple_env.py:**
- **trainer.py:** 
 
installation
--------

### clone
Clone this repository to your local machine using 'repository address goes here'
            
    git clone 'git address goes here' 

### setup 
while in the project directory, run setup.sh to install all requirements.

    double_dqn> setup.sh

usage
-----
To start training an agent, you must implement a class of dqn_env with the required methods. Only then can you 
initialize a dqn_agent with an instance of the environment as its only argument. Once you have done that, you will need
to set the model of the agent to be a compiled tf.keras Model tailored specifically to your environment's needs. 
After setting the agent's model, you can train it by calling dqn_agent.train with the necessary arguments

```python
from double_dqn.dqn_env import DQNenv
from double_dqn.dqn_agent import DQNagent
path = 'directory_for_saving_trained_model'
num_episodes = N
env = MyEnv() # Inherits from DQNenv
agent = DQNagent(env)
model = build_model(env.get_state_shape(), env.get_action_shape())
agent.set_model(model) # A compiled tf.keral Model to use as the agent's NN.
agent.train(num_episodes, path)
```
    
The train method saves the weights and the model architecture to the specified path

For a more detailed example, check out this [repository](https://github.com/dayMan33/double_dqn_usage.git)

support
-------
For any questions or comments, feel free to email me at danielrotem33@gmailcom.


