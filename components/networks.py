import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def weight_init(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        # extra parameter for the bias and register buffer for the bias parameter
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        # reset parameter as initialization of the layer
        self.reset_parameter()

    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, inputs):
        # sample random noise in sigma weight buffer and bias buffer
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(inputs, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes, layer_type="ff"):
        super(QNetwork, self).__init__()
        self.input_shape = state_size
        self.action_size = action_size
        self.state_dim = len(state_size)
        if self.state_dim == 3:
            self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
            self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            weight_init([self.cnn_1, self.cnn_2, self.cnn_3])

            if layer_type == "noisy":
                self.ff_1 = NoisyLinear(self.calc_input_layer(), hidden_sizes)
                self.ff_2 = NoisyLinear(hidden_sizes, action_size)
            else:
                self.ff_1 = nn.Linear(self.calc_input_layer(), hidden_sizes)
                self.ff_2 = nn.Linear(hidden_sizes, action_size)
                weight_init([self.ff_1])
        elif self.state_dim == 1:
            if layer_type == "noisy":
                self.head_1 = nn.Linear(self.input_shape[0], hidden_sizes)
                self.ff_1 = NoisyLinear(hidden_sizes, hidden_sizes)
                self.ff_2 = NoisyLinear(hidden_sizes, action_size)
            else:
                self.head_1 = nn.Linear(self.input_shape[0], hidden_sizes)
                self.ff_1 = nn.Linear(hidden_sizes, hidden_sizes)
                self.ff_2 = nn.Linear(hidden_sizes, action_size)
                weight_init([self.head_1, self.ff_1])
        else:
            print("Unknown input dimension!")

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]

    def forward(self, inputs):
        if self.state_dim == 3:
            x = torch.relu(self.cnn_1(inputs))
            x = torch.relu(self.cnn_2(x))
            x = torch.relu(self.cnn_3(x))
            x = x.view(inputs.size(0), -1)
        else:
            x = torch.relu(self.head_1(inputs))

        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)

        return out


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_sizes, layer_type="ff"):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingQNetwork, self).__init__()
        self.input_shape = state_size
        self.state_dim = len(self.input_shape)
        self.action_size = action_size
        if self.state_dim == 3:
            self.cnn_1 = nn.Conv2d(4, out_channels=32, kernel_size=8, stride=4)
            self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
            self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
            weight_init([self.cnn_1, self.cnn_2, self.cnn_3])
            if layer_type == "noisy":
                self.ff_1_A = NoisyLinear(self.calc_input_layer(), hidden_sizes)
                self.ff_1_V = NoisyLinear(self.calc_input_layer(), hidden_sizes)
                self.advantage = NoisyLinear(hidden_sizes, action_size)
                self.value = NoisyLinear(hidden_sizes, 1)
                weight_init([self.ff_1_A, self.ff_1_V])
            else:
                self.ff_1_A = nn.Linear(self.calc_input_layer(), hidden_sizes)
                self.ff_1_V = nn.Linear(self.calc_input_layer(), hidden_sizes)
                self.advantage = nn.Linear(hidden_sizes, action_size)
                self.value = nn.Linear(hidden_sizes, 1)
                weight_init([self.ff_1_A, self.ff_1_V])
        elif self.state_dim == 1:
            if layer_type == "noisy":
                self.head_1 = nn.Linear(self.input_shape[0], hidden_sizes)
                self.ff_1_A = NoisyLinear(hidden_sizes, hidden_sizes)
                self.ff_1_V = NoisyLinear(hidden_sizes, hidden_sizes)
                self.advantage = NoisyLinear(hidden_sizes, action_size)
                self.value = NoisyLinear(hidden_sizes, 1)
                weight_init([self.head_1, self.ff_1_A, self.ff_1_V])
            else:
                self.head_1 = nn.Linear(self.input_shape[0], hidden_sizes)
                self.ff_1_A = nn.Linear(hidden_sizes, hidden_sizes)
                self.ff_1_V = nn.Linear(hidden_sizes, hidden_sizes)
                self.advantage = nn.Linear(hidden_sizes, action_size)
                self.value = nn.Linear(hidden_sizes, 1)
                weight_init([self.head_1, self.ff_1_A, self.ff_1_V])
        else:
            print("Unknown input dimension!")

    def calc_input_layer(self):
        x = torch.zeros(self.input_shape).unsqueeze(0)
        x = self.cnn_1(x)
        x = self.cnn_2(x)
        x = self.cnn_3(x)
        return x.flatten().shape[0]

    def forward(self, inputs):
        """
        """
        if self.state_dim == 3:
            x = torch.relu(self.cnn_1(inputs))
            x = torch.relu(self.cnn_2(x))
            x = torch.relu(self.cnn_3(x))
            x = x.view(inputs.size(0), -1)
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))
        else:
            x = torch.relu(self.head_1(inputs))
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))

        value = self.value(x_V)
        value = value.expand(inputs.size(0), self.action_size)
        advantage = self.advantage(x_A)
        Q = value + advantage - advantage.mean()
        return Q

