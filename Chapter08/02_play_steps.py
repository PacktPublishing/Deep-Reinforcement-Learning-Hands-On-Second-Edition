#!/usr/bin/env python3
import gym
import ptan
import argparse

import numpy as np
import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-s", "--steps", type=int, default=1, help="Play steps to use, default=1")
    parser.add_argument("--seed", type=int, help="Random seed to use")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    params['batch_size'] *= args.steps

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)
    if args.seed is not None:
        env.seed(args.seed)

    suffix = "" if args.seed is None else "_seed=%s" % args.seed
    writer = SummaryWriter(comment="-" + params['run_name'] + "-02_play_steps=%d%s" % (args.steps, suffix))
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += args.steps
            buffer.populate(args.steps)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] < args.steps:
                tgt_net.sync()
