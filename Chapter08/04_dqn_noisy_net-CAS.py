#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import dqn_extra, common


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.noisy_layers = [
            dqn_model.NoisyLinear(input_shape[0], 128),
            dqn_model.NoisyLinear(128, 128),
            dqn_model.NoisyLinear(128, 128)
            #dqn_model.NoisyLinear(128, 128),
            #dqn_model.NoisyLinear(128, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
            nn.ReLU(),
            self.noisy_layers[2],
            nn.ReLU(),
            nn.Linear(128, n_actions)
            #self.noisy_layers[3],
            #nn.ReLU(),
            #self.noisy_layers[4],
            #nn.ReLU()
            #nn.Linear(128, 64),
            #nn.ReLU(),
            #nn.Linear(64, n_actions)
        )

    def forward(self, x):
        fx = x.float() / 256
        return self.fc(fx)

    def noisy_layers_sigma_snr(self):
        return [
            ((layer.weight ** 2).mean().sqrt() / (layer.sigma_weight ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]


if __name__ == "__main__":
    params = common.HYPERPARAMS['cartpole']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)

    writer = SummaryWriter(comment="-" + params.run_name + "-noisy-net")
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if frame_idx % 500 == 0:
                for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                    writer.add_scalar("sigma_snr_layer_%d" % (layer_idx+1),
                                      sigma_l2, frame_idx)
