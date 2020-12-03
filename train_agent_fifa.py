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

import time
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
env = gym.make("WimblepongVisualMultiplayer-v0")

TARGET_UPDATE = 25
glie_a = 5700
num_episodes = 50000
hidden = 64
gamma = 0.99
replay_buffer_size = 50000
batch_size = 32
frame_stacks = 3
warmup = 1000


wins = 0

# Get number of actions from gym action space
n_actions = env.action_space.n
#state_space_dim = env.observation_space.shape

player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id,bpe = 4)

# Introduces corruption to the game as the organization they are named after
# This takes effect for the x amount first episodes specified as the variable
fifa_episodes = 4000

# Task 4 - DQN
agent = DQNAgent(frame_stacks, n_actions, gamma, batch_size, replay_buffer_size)
if args.load:
    agent.policy_net.load_state_dict(agent.load_model(args.file))

env.set_names(agent.get_name(), opponent.get_name())
collision_threshold = 1
# Training loop
#FROM EXERCISE 4
cumulative_rewards = []
total_frames = 0
for ep in range(num_episodes):
    # Initialize the environment and state
    state,state1 = env.reset()
    # PRE PROCESS THE STATE
    state = agent._preprocess(state)
    done = False
    eps = glie_a/(glie_a+ep)
    cum_reward = 0
    collision_counter = 0
    collision_timer = 0
    i = 0
    if args.render:
        env.render()
    while not done:
        # Select and perform an action
        if ep < fifa_episodes and env.ball.last_touch == 2:
            collision_timer = i
            collision_counter = 1
        action = agent.get_action(state, eps)
        action2 = opponent.get_action()
        if ep < fifa_episodes and collision_counter == collision_threshold and i - collision_timer > 15:
            action2 = 0
        (ob1, _), (rew1, _), done, info = env.step((action,action2))
        cum_reward += rew1
        next_state = agent._preprocess(ob1)
        # Task 4: Update the DQN
        agent.store_transition(state, action, next_state, rew1, done)
        agent.update_network()

        # Move to the next state
        state = next_state
        if args.render:
            env.render()
        if done:
            # EMPTY memory
            #agent.history.empty()
            if rew1 > 9:
                wins += 1
        i += 1
    total_frames += i
    cumulative_rewards.append(cum_reward)
    #plot_rewards(cumulative_rewards)
    logging.info("Episode lasted for %i time steps", i)


    # Update the target network, copying all weights and biases in DQN
    # Uncomment for Task 4
    if ep % TARGET_UPDATE == 0:
        agent.update_target_network()

    if ep % 10 == 0:
        logging.info("Mean frame rate: %r",total_frames/(ep+1))
        logging.info("trained for: %s episodes", ep)
        logging.info("victory rate: %r", wins/(ep+1))

    # Save the policy
    # Uncomment for Task 4
    if ep % 1000 == 0:
        logging.info("saving model at ep: %s", ep)
        torch.save(agent.policy_net.state_dict(), "weights_%s_%d.mdl" % ("DQN", ep))

print('Complete')
