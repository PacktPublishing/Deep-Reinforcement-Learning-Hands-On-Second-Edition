import ptan
import random
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import preproc


class DQNModel(nn.Module):
    def __init__(self, obs_size: int, cmd_size: int,
                 hid_size: int = 256):
        super(DQNModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size + cmd_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1)
        )

    def forward(self, obs, cmd):
        x = torch.cat((obs, cmd), dim=1)
        return self.net(x)

    def q_values(self, obs_t, commands_t):
        """
        Calculate q-values for observation and tensor of commands
        :param obs_t: preprocessed observation, need to be of [1, obs_size] shape
        :param commands_t: commands to be evaluated, shape is [N, cmd_size]
        :return: list of q-values for commands
        """
        result = []
        for cmd_t in commands_t:
            qval = self(obs_t, cmd_t.unsqueeze(0))[0].cpu().item()
            result.append(qval)
        return result


class DQNAgent(ptan.agent.BaseAgent):
    def __init__(self, net: DQNModel,
                 preprocessor: preproc.Preprocessor,
                 epsilon: float = 0.0, device="cpu"):
        self.net = net
        self._prepr = preprocessor
        self._epsilon = epsilon
        self.device = device

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float):
        if 0.0 <= value <= 1.0:
            self._epsilon = value

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)

        # for every state in the batch, calculate
        actions = []
        for state in states:
            commands = state['admissible_commands']
            if random.random() <= self.epsilon:
                actions.append(random.randrange(len(commands)))
            else:
                obs_t = self._prepr.encode_sequences(
                    [state['obs']]).to(self.device)
                commands_t = self._prepr.encode_commands(commands)
                commands_t = commands_t.to(self.device)
                q_vals = self.net.q_values(obs_t, commands_t)
                actions.append(np.argmax(q_vals))
        return actions, agent_states


@torch.no_grad()
def unpack_batch(batch: List[ptan.experience.Experience],
                 preprocessor: preproc.Preprocessor,
                 net: DQNModel, device="cpu"):
    """
    Convert batch to data needed for Bellman step
    :param batch: list of ptan.Experience objects
    :param preprocessor: emb.Preprocessor instance
    :param net: network to be used for next state approximation
    :param device: torch device
    :return: tuple (list of observations, list of taken commands,
                    list of rewards, list of best Qs for the next s)
    """
    # calculate Qs for next states
    observations, taken_commands, rewards, best_q = [], [], [], []
    last_obs, last_commands, last_offsets = [], [], []
    for exp in batch:
        observations.append(exp.state['obs'])
        taken_commands.append(exp.state['admissible_commands'][exp.action])
        rewards.append(exp.reward)

        # calculate best Q value for the next state
        if exp.last_state is None:
            # final state in the episode, Q=0
            last_offsets.append(len(last_commands))
        else:
            last_obs.append(exp.last_state['obs'])
            last_commands.extend(exp.last_state['admissible_commands'])
            last_offsets.append(len(last_commands))

    obs_t = preprocessor.encode_sequences(last_obs).to(device)
    commands_t = preprocessor.encode_commands(last_commands).to(device)

    prev_ofs = 0
    obs_ofs = 0
    for ofs in last_offsets:
        if prev_ofs == ofs:
            best_q.append(0.0)
        else:
            q_vals = net.q_values(obs_t[obs_ofs:obs_ofs+1], commands_t[prev_ofs:ofs])
            best_q.append(max(q_vals))
            obs_ofs += 1
        prev_ofs = ofs
    return observations, taken_commands, rewards, best_q


def calc_loss_dqn(batch, preprocessor, tgt_preprocessor, net,
                  tgt_net, gamma, device="cpu"):
    observations, taken_commands, rewards, next_best_qs = \
        unpack_batch(batch, tgt_preprocessor, tgt_net, device)

    obs_t = preprocessor.encode_sequences(observations).to(device)
    cmds_t = preprocessor.encode_commands(taken_commands).to(device)
    q_values_t = net(obs_t, cmds_t)
    tgt_q_t = torch.tensor(rewards) + gamma * torch.tensor(next_best_qs)
    tgt_q_t = tgt_q_t.to(device)
    return F.mse_loss(q_values_t.squeeze(-1), tgt_q_t)
