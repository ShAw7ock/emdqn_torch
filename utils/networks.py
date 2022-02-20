import torch.nn as nn
import torch.nn.functional as f


class MLPNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dims=128):
        super(MLPNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, num_actions)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
