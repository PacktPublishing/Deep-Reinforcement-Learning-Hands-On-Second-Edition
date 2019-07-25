#!/usr/bin/env python3
import ptan
import argparse

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from tensorboardX import SummaryWriter

from lib import dqn_model, common, atari_wrappers

PLAY_STEPS = 4


def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True, pytorch_img=True)
    return env


def play_func(params, net, cuda, exp_queue):
    device = torch.device("cuda" if cuda else "cpu")
    env = make_env(params)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-05_new_wrappers")
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    exp_source_iter = iter(exp_source)

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

    exp_queue.put(None)


if __name__ == "__main__":
    mp.set_start_method('spawn')
    params = common.HYPERPARAMS['pong']
    params['batch_size'] *= PLAY_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = make_env(params)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=(params, net, args.cuda, exp_queue))
    play_proc.start()

    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += PLAY_STEPS
        for _ in range(PLAY_STEPS):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params['replay_initial']:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params['batch_size'])
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'],
                                      cuda=args.cuda, cuda_async=True)
        loss_v.backward()
        optimizer.step()

        if frame_idx % params['target_net_sync'] < PLAY_STEPS:
            tgt_net.sync()
