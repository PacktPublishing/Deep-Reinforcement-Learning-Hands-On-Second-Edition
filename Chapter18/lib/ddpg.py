import ptan
import numpy as np
import torch
import torch.nn as nn
from typing import Iterable, Optional


def fc_from_hidden_sizes(input_size: int, hidden_sizes: Iterable[int], output_size: int,
                         mid_activation: nn.Module = nn.ReLU,
                         out_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers = []
    prev_size = input_size
    for size in hidden_sizes:
        layers.append(nn.Linear(prev_size, size))
        layers.append(mid_activation())
        prev_size = size
    layers.append(nn.Linear(prev_size, output_size))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


class DDPGActor(nn.Module):
    def __init__(self, obs_size: int, act_size: int, hidden_sizes: Iterable[int] = (20, )):
        super(DDPGActor, self).__init__()

        self.net = fc_from_hidden_sizes(obs_size, hidden_sizes, act_size,
                                        mid_activation=nn.ReLU, out_activation=nn.Tanh)

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size: int, act_size: int,
                 hidden_obs_sizes: Iterable[int] = (100, ),
                 hidden_out_sizes: Iterable[int] = (100, 50)):
        super(DDPGCritic, self).__init__()

        self.obs_net = fc_from_hidden_sizes(obs_size, hidden_obs_sizes[:-1], hidden_obs_sizes[-1],
                                            mid_activation=nn.ReLU, out_activation=nn.ReLU)
        self.out_net = fc_from_hidden_sizes(hidden_obs_sizes[-1] + act_size, hidden_out_sizes, 1,
                                            mid_activation=nn.ReLU, out_activation=None)

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


class AgentDDPG(ptan.agent.BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)

                action += self.ou_epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states


def unpack_batch_ddpg(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = ptan.agent.float32_preprocessor(actions).to(device)
    rewards_v = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v
