from double_dqn.dqn_agent import DQNAgent
from dqn_usege_example.simple_env import SimpleEnv
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import model_from_json
from tensorflow.keras import Model
import json
import os

LIST_LEN = 4
UPDATE_RATE = 25
EPISODES = 500
EVAL_EPISODES = 20

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'saved_model')
ARCHITECTURE_PATH = os.path.join(MODEL_PATH, 'architecture.txt')


def build_model(input_shape: tuple, output_shape: tuple):
    """
    This method creates the actual network. The architecture is something to think about and design based on the
    specifics of your implementation.
    """
    n_layers = 2
    loss = 'mse'
    optimizer = 'adam'
    n_actions = max(output_shape)
    input_layer = Input(shape=input_shape)
    fc = Dense(60, activation='relu')(input_layer)
    for i in range(n_layers):
        fc = Dense(40, activation='relu')(fc)
    final = Dense(n_actions)(fc)
    model = Model(inputs=input_layer, outputs=final)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def train():
    # Initiate environment, agent and model
    env = SimpleEnv(LIST_LEN, 0, 10)
    agent = DQNAgent(env, UPDATE_RATE)
    state_shape, action_shape = env.get_state_shape(), env.get_action_shape()
    model = build_model(state_shape, action_shape)
    with open(ARCHITECTURE_PATH, 'w') as architecture_file:
        config = model.to_json()
        json.dump(config, architecture_file)

    agent.set_model(model)

    # Train agent
    agent.train(EPISODES, MODEL_PATH, 100, show_progress=True)


def evaluate():
    env = SimpleEnv(LIST_LEN, 0, 10)
    untrained_agent = DQNAgent(env, UPDATE_RATE, exploration_decay=0.001)
    untrained_agent.set_model(build_model(env.get_state_shape(), env.get_action_shape()))
    trained_agent = DQNAgent(env, UPDATE_RATE)

    with open(ARCHITECTURE_PATH, 'r') as json_file:
        config = json.load(json_file)
        model = model_from_json(config)
        trained_agent.set_model(model)
    trained_agent.load_weights(os.path.join(MODEL_PATH, 'final_weights'))

    untrained_num_actions, trained_num_actions = [], []
    for i in range(EVAL_EPISODES):
        # evaluate a randomly initiated DQN agent -
        state = env.reset()
        run = True
        j = 0
        while run:
            j += 1
            action = untrained_agent.get_action(state)
            next_state, reward, is_terminal = env.step(action)
            if is_terminal:
                untrained_num_actions.append(j)
                run = False
            state = next_state

        # evaluate a trained DQN agent -
        run = True
        state = env.reset()
        j = 0
        while run:
            j += 1
            action = trained_agent.get_action(state)
            next_state, reward, is_terminal = env.step(action)
            if is_terminal:
                trained_num_actions.append(j)
                run = False
            state = next_state
    untrained_average = sum(untrained_num_actions) / len(untrained_num_actions)
    trained_average = sum(trained_num_actions) / len(trained_num_actions)
    plt.bar([1, 2], [untrained_average, trained_average], width=0.7, color=['orange', 'blue'])
    plt.xticks([1, 2], ['untrained', 'trained'])

    plt.title('Average number of swaps taken to sort list')
    plt.ylabel('average # of swaps per episode')
    plt.show()


if __name__ == '__main__':
    train()
    evaluate()
