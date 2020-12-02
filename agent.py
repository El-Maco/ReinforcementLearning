import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import random
from utilis import Transition, ReplayMemory
from matplotlib import pyplot as plt

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html This was bad
# This is good: https://ai.stackexchange.com/questions/10203/dqn-stuck-at-suboptimal-policy-in-atari-pong-task
class DQN(nn.Module):

    def __init__(self, frame_stacks, outputs):
        super(DQN, self).__init__()
        self.action_space = 3
        self.hidden = 512

        self.conv1 = torch.nn.Conv2d(in_channels=frame_stacks,
                                     out_channels=32,
                                     kernel_size=8,
                                     stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.reshaped_size = 64 * 1 * 1
        self.fc1 = torch.nn.Linear(self.reshaped_size, self.hidden)
        self.head = torch.nn.Linear(self.hidden, outputs)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.shape[0], self.reshaped_size)
        x = F.relu(self.fc1(x))
        return self.head(x)

#Agent from exercise 4
class Agent(object):
    def __init__(self, frame_stacks=3, n_actions=3, gamma=0.98, batch_size=32, replay_buffer_size=50000):
        self.train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("train_device is:",self.train_device)
        self.policy_net = DQN(frame_stacks, n_actions).to(device=self.train_device)
        self.target_net = DQN(frame_stacks, n_actions).to(device=self.train_device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.gamma = gamma
        self.states = []
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.prev_obs = None
        self.n_actions = n_actions
        self.frame_stacks = frame_stacks
        self.agent_name = "TODO"

    def update_network(self, updates=1):
        for _ in range(updates):
            self._do_network_update()

    def _do_network_update(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8)
        non_final_mask = non_final_mask.type(torch.bool)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                         batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states).to(self.train_device)
        state_batch = torch.stack(batch.state).to(self.train_device)
        action_batch = torch.cat(batch.action).to(self.train_device)
        reward_batch = torch.cat(batch.reward).to(self.train_device)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size).to(device=self.train_device)

        # Double DQN, Vanilla did converge slow..
        # https://github.com/Shivanshu-Gupta/Pytorch-Double-DQN/blob/master/agent.py
        _, next_state_actions = self.policy_net(non_final_next_states).max(1, keepdim=True)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1,
                                                                                          next_state_actions).squeeze(1)

        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, state, epsilon=0.05):
        sample = random.random()
        if sample > epsilon:
            with torch.no_grad():
                #Have to convert to a batch of one state!
                state = torch.from_numpy(state).float().unsqueeze(0).to(device=self.train_device)
                q_values = self.policy_net(state)
                return torch.argmax(q_values).item()
        else:
            return random.randrange(self.n_actions)

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, next_state, reward, done):
        action = torch.Tensor([[action]]).long()
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state = torch.from_numpy(next_state).float()
        state = torch.from_numpy(state).float()
        self.memory.push(state, action, next_state, reward, done)

    def load_model(self,fpath="model.mdl"):
        #Pass filename as argument to load desired model
        return torch.load(fpath)

    def reset(self):
        return

    def get_name(self):
        return self.agent_name

    #https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
    def _preprocess(self, observation):

        observation = np.mean(observation, axis=2).astype(np.uint8)  # convert to greyscale
        observation = observation[::5, ::5] # Ball is 5 pixels -> quaranteed 1 pixel
        observation = observation[:, 2:-2] # Remove first 2 columns and Right 3 columns
        # np.set_printoptions(threshold=np.inf)
        # print(observation.shape)
        # plt.imshow(observation)
        # plt.show()
        observation = np.expand_dims(observation, axis=0)

        if self.prev_obs is None:
            self.prev_obs = observation

        stack_ob = np.concatenate((self.prev_obs, observation), axis=0)

        while stack_ob.shape[0] < self.frame_stacks:
            stack_ob = self._stack_frames(stack_ob, observation)
        self.prev_obs = stack_ob[1:self.frame_stacks, :, :]
        return stack_ob

    def _stack_frames(self, stack_ob, obs):
        return np.concatenate((stack_ob, obs), axis=0)

