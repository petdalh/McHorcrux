import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self,
                 alpha,                   # actor learning rate
                 input_dims,               # e.g. env.observation_space.shape
                 fc1_dims,               # first hidden layer size
                 fc2_dims,               # second hidden layer size
                 chkpt_dir,
                 n_actions,                # env.action_space.shape[0]
                 name
                 ):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

        # Input layer
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        fan_in1 = self.fc1.weight.data.size(1)
        f1 = 1./np.sqrt(fan_in1)
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # Hidden layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        fan_in2 = self.fc2.weight.data.size(1)
        f2 = 1./np.sqrt(fan_in2)
        # f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Output layers
        self.mu = nn.Linear(fc2_dims, n_actions)
        f3 = 0.003
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = self.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = self.relu(state_value)

        action = self.mu(state_value)
        action = self.tanh(action)

        return action

    def test(self):
        print(self.fc1.weight.data.size(1))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, chkpt_dir, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, f"{name}_ddpg")

        # Input layer
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        fan_in1 = self.fc1.weight.data.size(1)
        f1 = 1./np.sqrt(fan_in1)
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        # Hidden layer
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        fan_in2 = self.fc2.weight.data.size(1)
        f2 = 1./np.sqrt(fan_in2)
        # f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # Output layer (Q-value)
        self.actions = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)
        f3 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = self.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.relu(self.actions(action))
        state_action_value = self.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
