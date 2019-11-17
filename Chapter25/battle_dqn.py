#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "MAgent/python"))

import gym
from gym.wrappers.time_limit import TimeLimit
import ptan
import torch
import argparse
import magent
from typing import Tuple, List
import ptan.ignite as ptan_ignite

from torch import optim
from types import SimpleNamespace
from lib import data, model, common
from ignite.engine import Engine


MAP_SIZE = 16
COUNT_AGENTS_1 = 20
COUNT_AGENTS_2 = 20
WALLS_DENSITY = 0.04
MAX_EPISODE = 300


PARAMS = SimpleNamespace(**{
    'run_name':         'battle',
    'stop_reward':      None,
    'replay_size':      2000000,
    'replay_initial':   100,
    'target_net_sync':  2000,
    'epsilon_frames':   10**5,
    'epsilon_start':    1.0,
    'epsilon_final':    0.02,
    'learning_rate':    1e-4,
    'gamma':            0.95,
    'batch_size':       128
})


def test_model(net: model.DQNModel,
               device: torch.device, gw_config) -> Tuple[float, float, float, float]:
    test_env = magent.GridWorld(gw_config, map_size=MAP_SIZE)
    group_a, group_b = test_env.get_handles()

    def reset_env():
        test_env.reset()
        test_env.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * WALLS_DENSITY)
        test_env.add_agents(group_a, method="random", n=COUNT_AGENTS_1)
        test_env.add_agents(group_b, method="random", n=COUNT_AGENTS_2)

    env_a = data.MAgentEnv(test_env, group_a, reset_env_func=reset_env, is_slave=True)
    env_b = data.MAgentEnv(test_env, group_b, reset_env_func=reset_env, steps_limit=MAX_EPISODE)
    preproc = model.MAgentPreprocessor(device)
    agent_a = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device, preprocessor=preproc)
    agent_b = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device, preprocessor=preproc)

    a_obs = env_a.reset()
    b_obs = env_b.reset()
    a_steps = 0
    a_rewards = 0.0
    b_steps = 0
    b_rewards = 0.0

    while True:
        a_actions = agent_a(a_obs)[0]
        b_actions = agent_b(b_obs)[0]
        a_obs, a_r, a_dones, _ = env_a.step(a_actions)
        b_obs, b_r, b_dones, _ = env_b.step(b_actions)
        a_steps += len(a_obs)
        a_rewards += sum(a_r)
        if a_dones[0]:
            break
        b_steps += len(b_obs)
        b_rewards += sum(b_r)
        if b_dones[0]:
            break

    return a_rewards / COUNT_AGENTS_1, b_steps / COUNT_AGENTS_1, \
           b_rewards / COUNT_AGENTS_2, b_steps / COUNT_AGENTS_2


def batch_generator(a_exp: ptan.experience.ExperienceSource,
                    b_exp: ptan.experience.ExperienceSource,
                    buffer: ptan.experience.ExperienceReplayBuffer,
                    replay_initial: int, batch_size: int):
    for a_e, b_e in zip(a_exp, b_exp):
        buffer._add(a_e)
        buffer._add(b_e)
        if len(buffer) < replay_initial:
            continue
        yield buffer.sample(batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA computations")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    args = parser.parse_args()

    config = "battle"

    device = torch.device("cuda" if args.cuda else "cpu")
    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)

    m_env = magent.GridWorld(config, map_size=MAP_SIZE)

    a_handle, b_handle = m_env.get_handles()

    def reset_env():
        m_env.reset()
        m_env.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * WALLS_DENSITY)
        m_env.add_agents(a_handle, method="random", n=COUNT_AGENTS_1)
        m_env.add_agents(b_handle, method="random", n=COUNT_AGENTS_2)

    a_env = data.MAgentEnv(m_env, a_handle, reset_env_func=lambda: None, is_slave=True)
    b_env = data.MAgentEnv(m_env, b_handle, reset_env_func=reset_env, is_slave=False, steps_limit=MAX_EPISODE)

    obs = data.MAgentEnv.handle_obs_space(m_env, a_handle)

    net = model.DQNModel(
        obs.spaces[0].shape, obs.spaces[1].shape,
        m_env.get_action_space(a_handle)[0]).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)

    action_selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=PARAMS.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(action_selector, PARAMS)
    preproc = model.MAgentPreprocessor(device)

    agent = ptan.agent.DQNAgent(net, action_selector, device, preprocessor=preproc)
    a_exp_source = ptan.experience.ExperienceSourceFirstLast(
        a_env, agent, PARAMS.gamma, vectorized=True)
    b_exp_source = ptan.experience.ExperienceSourceFirstLast(
        b_env, agent, PARAMS.gamma, vectorized=True)
    buffer = ptan.experience.ExperienceReplayBuffer(None, PARAMS.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=PARAMS.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = model.calc_loss_dqn(
            batch, net, tgt_net.target_model, preproc,
            gamma=PARAMS.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        if engine.state.iteration % PARAMS.target_net_sync == 0:
            tgt_net.sync()

        epsilon_tracker.frame(engine.state.iteration)
        return {
            'epsilon': action_selector.epsilon,
            'loss': loss_v.item()
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, PARAMS, b_exp_source, args.name,
                        extra_metrics=('test_reward_a', 'test_steps_a', 'test_reward_b', 'test_steps_b'))
    best_test_reward = None

    @engine.on(ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def test_network(engine):
        net.train(False)
        a_reward, a_steps, b_reward, b_steps = test_model(net, device, config)
        net.train(True)
        engine.state.metrics['test_reward_a'] = a_reward
        engine.state.metrics['test_steps_a'] = a_steps
        engine.state.metrics['test_reward_b'] = b_reward
        engine.state.metrics['test_steps_b'] = b_steps
        print("Test done: A got %.3f reward after %.2f steps, B %.3f reward after %.2f steps" % (
            a_reward, a_steps, b_reward, b_steps
        ))

        global best_test_reward
        reward = max(a_reward, b_reward)

        if best_test_reward is None:
            best_test_reward = reward
        elif best_test_reward < reward:
            print("Best test reward updated %.3f <- %.3f, save model" % (
                best_test_reward, reward
            ))
            best_test_reward = reward
            torch.save(net.state_dict(), os.path.join(saves_path, "best_%.3f.dat" % reward))

    engine.run(batch_generator(
        a_exp_source, b_exp_source, buffer,
        PARAMS.replay_initial, PARAMS.batch_size))
