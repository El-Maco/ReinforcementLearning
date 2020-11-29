import wimblepong
import gym
import numpy as np
from matplotlib import pyplot as plt
from itertools import count
import torch
import logging
import sys
import os
import argparse
import random
from agent import newai as DQNAgent

parser = argparse.ArgumentParser()
parser.add_argument("--load", action="store_true",
                    help="Load weights from file")
parser.add_argument("--file", type=str, default=None,
                    help="Name of the file that contains the weights")
parser.add_argument("--render", action="store_true",
                    help="render")
args = parser.parse_args()

# set up logging
logging.basicConfig(level=logging.INFO, filename='dqn.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logging.info("log file for DQN agent training")

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")

TARGET_UPDATE = 30
glie_a = 5000
num_episodes = 50000
hidden = 64
gamma = 0.99
replay_buffer_size = 50000
batch_size = 32


wins = 0

# Get number of actions from gym action space
#n_actions = env.action_space.n
#state_space_dim = env.observation_space.shape


# Task 4 - DQN
agent = DQNAgent()
if args.load:
    agent.policy_net.load_state_dict(agent.load_model(args.file))

# Training loop
#FROM EXERCISE 4
cumulative_rewards = []
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    # PRE PROCESS THE STATE
    state = agent._preprocess(state)

    done = False
    if ep < 25000:
    	eps = 0.01
    else:
    	eps = 0.01
    cum_reward = 0

    i = 0
    if args.render:
        env.render()
    while not done:
        # Select and perform an action
        action = agent.get_action(state, eps)
        next_state, reward, done, _ = env.step(action)
        cum_reward += reward
        next_state = agent._preprocess(next_state)

        # Task 4: Update the DQN
        agent.store_transition(state, action, next_state, reward, done)
        agent.update_network()

        # Move to the next state
        state = next_state
        if args.render:
            env.render()
        if done:
            # EMPTY memory
            #agent.history.empty()
            if reward > 0:
                wins += 1

        i += 1
    cumulative_rewards.append(cum_reward)
    #plot_rewards(cumulative_rewards)
    logging.info("Episode lasted for %i time steps", i)


    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    if ep % 10 == 0:
        logging.info("trained for: %s episodes", ep)
        logging.info("victory rate: %r", wins/(ep+1))
        logging.info("Won total: %s",wins)

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        logging.info("saving model at ep: %s", ep)
        torch.save(agent.policy_net.state_dict(), "weights_%s_%d_new.mdl" % ("DQN", ep))

print('Complete')
