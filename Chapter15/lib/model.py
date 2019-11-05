import gym
import ptan
import random
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as t_distr

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

    @torch.no_grad()
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

    @torch.no_grad()
    def q_values_cmd(self, obs_t, commands_t):
        x = torch.cat(torch.broadcast_tensors(obs_t.unsqueeze(0), commands_t), dim=1)
        q_vals = self.net(x)
        return q_vals.cpu().numpy()[:, 0].tolist()


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
def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast],
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


class CommandModel(nn.Module):
    def __init__(self, obs_size: int, dict_size: int,
                 embeddings: nn.Embedding, max_tokens: int,
                 max_commands: int, start_token: int,
                 sep_token: int):
        super(CommandModel, self).__init__()

        self.emb = embeddings
        self.max_tokens = max_tokens
        self.max_commands = max_commands
        self.start_token = start_token
        self.sep_token = sep_token

        self.rnn = nn.LSTM(
            input_size=embeddings.embedding_dim,
            hidden_size=obs_size, batch_first=True)
        self.out = nn.Linear(in_features=obs_size,
                             out_features=dict_size)

    def forward(self, input_seq, obs_t):
        hid_t = obs_t.unsqueeze(0)
        output, _ = self.rnn(input_seq, (hid_t, hid_t))
        return self.out(output)

    def commands(self, obs_batch):
        """
        Generate commands from batch of encoded observations
        :param obs_batch: tensor of (batch, obs_size)
        :return: batch of commands lists
        """
        batch_size = obs_batch.size(0)
        # list of finalized commands and logits for every observation in batch
        commands = [[] for _ in range(batch_size)]

        # currently being constructed list
        cur_commands = [[] for _ in range(batch_size)]

        # preprare input tensor with start token embeddings
        inp_t = torch.full((batch_size, ), self.start_token,
                           dtype=torch.long)
        inp_t = inp_t.to(obs_batch.device)
        inp_t = self.emb(inp_t)
        # adding time dimension (dim=1, as batch_first=True)
        inp_t = inp_t.unsqueeze(1)
        # hidden state is inserted on first dim
        p_hid_t = obs_batch.unsqueeze(0)
        hid = (p_hid_t, p_hid_t)

        while True:
            out, hid = self.rnn(inp_t, hid)
            out = out.squeeze(1)
            # output logits for batch at current time step
            out_t = self.out(out)

            cat = t_distr.Categorical(logits=out_t)
            tokens = cat.sample()

            for idx, token in enumerate(tokens):
                token = token.item()
                cur_commands[idx].append(token)
                if token == self.sep_token or \
                        len(cur_commands[idx]) >= self.max_tokens:
                    if cur_commands[idx]:
                        l = len(commands[idx])
                        if l < self.max_commands:
                            commands[idx].append(cur_commands[idx])
                        cur_commands[idx] = []
                    if token != self.sep_token:
                        tokens[idx] = self.sep_token
            if min(map(len, commands)) == self.max_commands:
                break
            # convert tokens into input tensor
            inp_t = self.emb(tokens)
            inp_t = inp_t.unsqueeze(1)
        return commands


class CmdAgent(ptan.agent.BaseAgent):
    def __init__(self, env, cmd: CommandModel,
                 preprocessor: preproc.Preprocessor,
                 device = "cpu"):
        self.env = env
        self.cmd = cmd
        self.prepr = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)

        actions = []
        for state in states:
            obs_t = self.prepr.encode_sequences(
                [state['obs']]).to(self.device)
            commands = self.cmd.commands(obs_t)[0]
            cmd = random.choice(commands)
            tokens = [
                self.env.action_space.id2w[t]
                for t in cmd
                if t not in {self.cmd.sep_token,
                             self.cmd.start_token}
            ]
            action = " ".join(tokens)
            actions.append(action)
        return actions, agent_states


def pretrain_loss(cmd: CommandModel, commands: List,
                  observations_t: torch.Tensor):
    commands_batch = []
    target_batch = []
    min_length = None

    for cmds in commands:
        inp = [cmd.start_token]
        for c in cmds:
            inp.extend(c[1:])
        commands_batch.append(inp[:-1])
        target_batch.append(inp[1:])
        if min_length is None or len(inp) < min_length:
            min_length = len(inp)

    commands_batch = [c[:min_length-1]
                      for c in commands_batch]
    target_batch = [c[:min_length-1]
                    for c in target_batch]

    commands_t = torch.tensor(commands_batch, dtype=torch.long)
    commands_t = commands_t.to(observations_t.device)
    target_t = torch.tensor(target_batch, dtype=torch.long)
    target_t = target_t.to(observations_t.device)
    input_t = cmd.emb(commands_t)
    logits_t = cmd(input_t, observations_t)
    logits_t = logits_t.view(-1, logits_t.size()[-1])
    target_t = target_t.view(-1)
    return F.cross_entropy(logits_t, target_t)


class CmdDQNAgent(ptan.agent.BaseAgent):
    def __init__(self, env, net: DQNModel, cmd: CommandModel,
                 cmd_encoder: preproc.Encoder,
                 preprocessor: preproc.Preprocessor,
                 epsilon: float,
                 device = "cpu"):
        self.env = env
        self.net = net
        self.cmd = cmd
        self.cmd_encoder = cmd_encoder
        self.prepr = preprocessor
        self.epsilon = epsilon
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)

        actions = []
        for state in states:
            obs_t = self.prepr.encode_sequences([state['obs']]).to(self.device)
            commands = self.cmd.commands(obs_t)[0]
            if random.random() <= self.epsilon:
                act_index = random.randrange(len(commands))
            else:
                cmd_enc_t = self.prepr._apply_encoder(commands, self.cmd_encoder)
                q_vals = self.net.q_values_cmd(obs_t[0], cmd_enc_t)
                act_index = np.argmax(q_vals)

            cmd = commands[act_index]
            tokens = [
                self.env.action_space.id2w[t]
                for t in cmd
                if t not in {self.cmd.sep_token, self.cmd.start_token}
            ]
            action = " ".join(tokens)
            actions.append(action)
        return actions, agent_states


@torch.no_grad()
def unpack_batch_dqncmd(batch, prep: preproc.Preprocessor,
                        cmd: CommandModel,
                        cmd_encoder: preproc.Encoder,
                        net: DQNModel, env: gym.Env):
    observations, taken_actions, rewards = [], [], []
    not_done_indices, next_observations = [], []

    for idx, exp in enumerate(batch):
        observations.append(exp.state['obs'])
        taken_actions.append(env.action_space.tokenize(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_indices.append(idx)
            next_observations.append(exp.last_state['obs'])

    observations_t = prep.encode_sequences(observations)
    next_q_vals = [0.0] * len(batch)
    if next_observations:
        next_observations_t = \
            prep.encode_sequences(next_observations)
        next_commands = cmd.commands(next_observations_t)

        for idx, next_obs_t, next_cmds in \
                zip(not_done_indices,
                    next_observations_t,
                    next_commands):
            next_embs_t = prep._apply_encoder(
                next_cmds, cmd_encoder)
            q_vals = net.q_values_cmd(next_obs_t, next_embs_t)
            next_q_vals[idx] = max(q_vals)

    return observations_t, taken_actions, rewards, next_q_vals


def calc_loss_dqncmd(batch, preprocessor, cmd,
                     cmd_encoder, tgt_cmd_encoder,
                     net, tgt_net, gamma, env, device="cpu"):
    obs_t, commands, rewards, next_best_qs = unpack_batch_dqncmd(
        batch, preprocessor, cmd, tgt_cmd_encoder, tgt_net, env)
    cmds_t = preprocessor._apply_encoder(commands, cmd_encoder)
    q_values_t = net(obs_t, cmds_t)
    tgt_q_t = torch.tensor(rewards) + \
              gamma * torch.tensor(next_best_qs)
    tgt_q_t = tgt_q_t.to(device)
    return F.mse_loss(q_values_t.squeeze(-1), tgt_q_t)
