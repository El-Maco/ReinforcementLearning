
####################################################################
############ ELEC-E8125 - Reinforcement learning D #################
### This project is created by Robin Nyman and Marcus Myllyviita ###
####################################################################


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
from agent import Agent as DQNAgent

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
env_name = "WimblepongVisualSimpleAI-v0"
# env_name = "CartPole-v0"
env = gym.make(env_name)

if "CartPole" in env_name:
    TARGET_UPDATE = 50
    glie_a = 500
    num_episodes = 2000
    hidden = 12
    gamma = 0.95
    replay_buffer_size = 500000
    batch_size = 256
    frame_stacks = 2
    height = 400
    width = 600
elif "WimblepongVisualSimpleAI" in env_name:
    TARGET_UPDATE = 25
    glie_a = 5700
    num_episodes = 100000
    hidden = 64
    gamma = 0.99
    replay_buffer_size = 50000
    batch_size = 32
    frame_stacks = 3
    height = 100
    width = 100
else:
    raise ValueError("Please provide hyperparameters for %s" % env_name)


wins = 0

# Get number of actions from gym action space
n_actions = env.action_space.n
state_space_dim = env.observation_space.shape


# Task 4 - DQN
agent = DQNAgent(n_actions, frame_stacks, gamma, batch_size, replay_buffer_size)
if args.load:
    agent.policy_net.load_state_dict(agent.load_model(args.file))

# Training loop
#FROM EXERCISE 4
cumulative_rewards = []
total_frames=0
for ep in range(num_episodes):
    # Initialize the environment and state
    state = env.reset()

    # PRE PROCESS THE STATE
    if "CartPole" in env_name:
        state = env.render(mode='rgb_array')
        env.close()
    state = agent._preprocess(state)

    done = False
    eps = glie_a/(glie_a+ep)
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
    total_frames+=i
    cumulative_rewards.append(cum_reward)
    #plot_rewards(cumulative_rewards)
    logging.info("Episode lasted for %i time steps - %s", i, "W" if cum_reward > 0 else "L")


    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    if ep % 10 == 0:
        logging.info("Mean frame rate: %r",total_frames/(ep+1))
        logging.info("trained for: %s episodes", ep)
        logging.info("victory rate: %r", wins/(ep+1))
        logging.info("Won total: %s",wins)

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        logging.info("saving model at ep: %s", ep)
        torch.save(agent.policy_net.state_dict(), "weights_%s_%d_new.mdl" % ("DQN", ep))

print('Complete')
