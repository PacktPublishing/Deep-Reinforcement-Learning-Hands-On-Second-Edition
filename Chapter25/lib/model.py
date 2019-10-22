import ptan
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, List


class DQNModel(nn.Module):
    def __init__(self, view_shape, feats_shape, n_actions):
        super(DQNModel, self).__init__()

        self.view_conv = nn.Sequential(
            nn.Conv2d(view_shape[0], 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, padding=1),        # padding was added for deer model
            nn.ReLU(),
        )
        view_out_size = self._get_conv_out(view_shape)
        self.fc = nn.Sequential(
            nn.Linear(view_out_size + feats_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        view_batch, feats_batch = x
        batch_size = view_batch.size()[0]
        conv_out = self.view_conv(view_batch).view(batch_size, -1)
        fc_input = torch.cat((conv_out, feats_batch), dim=1)
        return self.fc(fc_input)


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


class DQNNoisyModel(nn.Module):
    def __init__(self, view_shape, feats_shape, n_actions):
        super(DQNNoisyModel, self).__init__()

        self.view_conv = nn.Sequential(
            nn.Conv2d(view_shape[0], 32, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=2, padding=1),        # padding was added for deer model
            nn.ReLU(),
        )
        view_out_size = self._get_conv_out(view_shape)
        self.fc = nn.Sequential(
            nn.Linear(view_out_size + feats_shape[0], 128),
            nn.ReLU(),
            NoisyLinear(128, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.view_conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        view_batch, feats_batch = x
        batch_size = view_batch.size()[0]
        conv_out = self.view_conv(view_batch).view(batch_size, -1)
        fc_input = torch.cat((conv_out, feats_batch), dim=1)
        return self.fc(fc_input)


class MAgentPreprocessor:
    """
    Transform the batch of observations from data.MAgentEnv into tuple of tensors, suitable for model call
    """
    def __init__(self, device: Union[torch.device, str] = "cpu"):
        self.device = device

    def __call__(self, batch: List[Tuple[np.ndarray, np.ndarray]]) \
            -> Tuple[torch.tensor, torch.tensor]:
        """
        Preprocess batch of observations from MAgentEnv
        :param batch: list of tuples with view numpy array and features numpy array
        :return: tuple of tensors with the same arrays
        """
        view_arrays, feat_arrays = zip(*batch)
        view_t = torch.tensor(view_arrays, dtype=torch.float32).to(self.device)
        feat_t = torch.tensor(feat_arrays, dtype=torch.float32).to(self.device)
        return view_t, feat_t


def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = exp.state  # the result will be masked anyway
        else:
            lstate = exp.last_state
        last_states.append(lstate)
    return states, np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           last_states


def calc_loss_dqn(batch, net, tgt_net, preprocessor, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = \
        unpack_batch(batch)

    states = preprocessor(states)
    next_states = preprocessor(next_states)

    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_vals = net(states).gather(1, actions_v)
    state_action_vals = state_action_vals.squeeze(-1)
    with torch.no_grad():
        next_state_vals = tgt_net(next_states).max(1)[0]
        next_state_vals[done_mask] = 0.0

    bellman_vals = next_state_vals.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_vals, bellman_vals)


class GroupDQNAgent(ptan.agent.BaseAgent):
    """
    Similar to DQNAgent, but works with several models. Observations are tuples
    """
    def __init__(self, dqn_models: List[DQNModel],
                 action_selector, device="cpu",
                 preprocessor=ptan.agent.default_states_preprocessor):
        self.dqn_models = dqn_models
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        result = []
        for states_batch, model in zip(states, self.dqn_models):
            if self.preprocessor is not None:
                states_batch = self.preprocessor(states_batch)
            q_v = model(states_batch)
            q = q_v.data.cpu().numpy()
            actions = self.action_selector(q)
            result.append(actions)
        return result, agent_states
