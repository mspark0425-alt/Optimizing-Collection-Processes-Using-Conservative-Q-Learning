import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Q-network definition
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Remove Base_PD (index 0) and Tone_ever (index 11) from state
        state = state[:, [i for i in range(state.shape[1]) if i not in [0, 11]]]
        x = torch.cat([state, action], dim=1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        q_value = self.q_out(x)
        
        return q_value