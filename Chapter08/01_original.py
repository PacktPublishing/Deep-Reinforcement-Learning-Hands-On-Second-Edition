#!/usr/bin/env python3
import gym
import ptan
from datetime import datetime
import argparse

import torch
import torch.optim as optim

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger

from lib import dqn_model, common


def batch_generator(buffer: ptan.experience.ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model,
                                      gamma=params.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    logdir = f"runs/{datetime.now().isoformat()}-{params.run_name}-01_original"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")
    handler = tb_logger.OutputHandler(tag="train", metric_names=['avg_loss'],
                                      output_transform=lambda a: a)
    tb.attach(engine, log_handler=handler, event_name=Events.ITERATION_COMPLETED)
    engine.run(batch_generator(buffer, params.replay_initial, params.batch_size))
    #
    # frame_idx = 0
    #
    # with common.RewardTracker(writer, params.stop_reward) as reward_tracker:
    #     while True:
    #         frame_idx += 1
    #         buffer.populate(1)
    #         epsilon_tracker.frame(frame_idx)
    #
    #         new_rewards = exp_source.pop_total_rewards()
    #         if new_rewards:
    #             if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
    #                 break
    #
    #         if len(buffer) < params['replay_initial']:
    #             continue
    #
    #         optimizer.zero_grad()
    #         batch = buffer.sample(params['batch_size'])
    #         loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda)
    #         loss_v.backward()
    #         optimizer.step()
    #
    #         if frame_idx % params['target_net_sync'] == 0:
    #             tgt_net.sync()
