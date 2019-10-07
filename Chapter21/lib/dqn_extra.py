import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

# replay buffer params
BETA_START = 0.4
BETA_FRAMES = 100000

# distributional DQN params
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(
            in_features, out_features, bias=bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        if not self.training:
            return super(NoisyLinear, self).forward(input)
        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * \
                   self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + \
            self.weight
        return F.linear(input, v, bias)

    def sample_noise(self):
        self.epsilon_weight.normal_()
        if self.bias is not None:
            self.epsilon_bias.normal_()


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features,
                 sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(
            in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(1, in_features)
        self.register_buffer("epsilon_input", z1)
        z2 = torch.zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        if bias:
            w = torch.full((out_features,), sigma_init)
            self.sigma_bias = nn.Parameter(w)

    def forward(self, input):
        if not self.training:
            return super(NoisyFactorizedLinear, self).forward(input)

        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * \
                         torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        v = self.weight + self.sigma_weight * noise_v
        return F.linear(input, v, bias)


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 256),
            NoisyLinear(256, n_actions)
        ]
        self.fc_adv = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


class BaselineDQN(nn.Module):
    """
    Dueling net
    """
    def __init__(self, input_shape, n_actions):
        super(BaselineDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        adv, val = self.adv_val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def adv_val(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc_adv(conv_out), self.fc_val(conv_out)


class MountainCarBaseDQN(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarBaseDQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class MountainCarNoisyNetDQN(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarNoisyNetDQN, self).__init__()

        self.noisy_layers = [
            NoisyLinear(hid_size, n_actions),
        ]

        self.net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            self.noisy_layers[0]
        )

    def forward(self, x):
        return self.net(x)

    def sample_noise(self):
        for l in self.noisy_layers:
            l.sample_noise()
