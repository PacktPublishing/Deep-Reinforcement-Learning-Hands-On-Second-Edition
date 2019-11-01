import numpy as np
import torch
import torch.nn as nn
from types import SimpleNamespace

SEED = 123

HYPERPARAMS = {
    'pong': SimpleNamespace(**{
        'env_name':         "PongNoFrameskip-v4",
        'stop_reward':      18.0,
        'run_name':         'pong',
        'replay_size':      100000,
        'replay_initial':   10000,
        'target_net_sync':  1000,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'breakout-small': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout-small',
        'replay_size':      3*10 ** 5,
        'replay_initial':   20000,
        'target_net_sync':  1000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       64
    }),
    'breakout': SimpleNamespace(**{
        'env_name':         "BreakoutNoFrameskip-v4",
        'stop_reward':      500.0,
        'run_name':         'breakout',
        'replay_size':      10 ** 6,
        'replay_initial':   50000,
        'target_net_sync':  10000,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.1,
        'learning_rate':    0.00025,
        'gamma':            0.99,
        'batch_size':       32
    }),
    'invaders': SimpleNamespace(**{
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    }),
}


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params.epsilon_start
        self.epsilon_final = params.epsilon_final
        self.epsilon_frames = params.epsilon_frames
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
