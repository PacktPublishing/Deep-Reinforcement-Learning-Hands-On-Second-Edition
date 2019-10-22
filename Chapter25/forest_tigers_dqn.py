#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "MAgent/python"))

import ptan
import torch
import argparse
import magent
from typing import Tuple
import ptan.ignite as ptan_ignite

from torch import optim
from types import SimpleNamespace
from lib import data, model, common
from ignite.engine import Engine


MAP_SIZE = 64
COUNT_TIGERS = 10
COUNT_DEERS = 50
WALLS_DENSITY = 0.04


PARAMS = SimpleNamespace(**{
    'run_name':         'tigers',
    'stop_reward':      None,
    'replay_size':      1000000,
    'replay_initial':   100,
    'target_net_sync':  1000,
    'epsilon_frames':   5*10**5,
    'epsilon_start':    1.0,
    'epsilon_final':    0.02,
    'learning_rate':    1e-4,
    'gamma':            0.99,
    'batch_size':       32
})


def test_model(net: model.DQNModel, device: torch.device, gw_config) -> Tuple[float, float]:
    test_env = magent.GridWorld(gw_config, map_size=MAP_SIZE)
    deer_handle, tiger_handle = test_env.get_handles()

    def reset_env():
        test_env.reset()
        test_env.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * WALLS_DENSITY)
        test_env.add_agents(deer_handle, method="random", n=COUNT_DEERS)
        test_env.add_agents(tiger_handle, method="random", n=COUNT_TIGERS)

    env = data.MAgentEnv(test_env, tiger_handle, reset_env_func=reset_env)
    preproc = model.MAgentPreprocessor(device)
    agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device, preprocessor=preproc)

    obs = env.reset()
    steps = 0
    rewards = 0.0

    while True:
        actions = agent(obs)[0]
        obs, r, dones, _ = env.step(actions)
        steps += len(obs)
        rewards += sum(r)
        if dones[0]:
            break

    return rewards / COUNT_TIGERS, steps / COUNT_TIGERS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA computations")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("--mode", default='forest', choices=['forest', 'double_attack', 'double_attack_nn'],
                        help="GridWorld mode, could be 'forest', 'double_attack' or 'double_attck_nn', default='forest'")
    args = parser.parse_args()

    config = args.mode
    # tweak count of agents in this mode to simplify exploration
    if args.mode == 'double_attack':
        COUNT_TIGERS = 20
        COUNT_DEERS = 1024
        # tweaked double_attack
        config = data.config_double_attack(MAP_SIZE)
    elif args.mode == 'double_attack_nn':
        COUNT_TIGERS = 20
        COUNT_DEERS = 1024
        # original double_attack setting
        config = 'double_attack'

    device = torch.device("cuda" if args.cuda else "cpu")
    saves_path = os.path.join("saves", args.name)
    os.makedirs(saves_path, exist_ok=True)

    m_env = magent.GridWorld(config, map_size=MAP_SIZE)

    # two groups of animal
    deer_handle, tiger_handle = m_env.get_handles()

    def reset_env():
        m_env.reset()
        m_env.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * WALLS_DENSITY)
        m_env.add_agents(deer_handle, method="random", n=COUNT_DEERS)
        m_env.add_agents(tiger_handle, method="random", n=COUNT_TIGERS)

    env = data.MAgentEnv(m_env, tiger_handle, reset_env_func=reset_env)

    if args.mode == 'double_attack_nn':
        net = model.DQNNoisyModel(env.single_observation_space.spaces[0].shape,
                                  env.single_observation_space.spaces[1].shape,
                                  m_env.get_action_space(tiger_handle)[0]).to(device)
    else:
        net = model.DQNModel(env.single_observation_space.spaces[0].shape,
                             env.single_observation_space.spaces[1].shape,
                             m_env.get_action_space(tiger_handle)[0]).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)

    if args.mode == 'double_attack':
        action_selector = ptan.actions.ArgmaxActionSelector()
        epsilon_tracker = None
    else:
        action_selector = ptan.actions.EpsilonGreedyActionSelector(
            epsilon=PARAMS.epsilon_start)
        epsilon_tracker = common.EpsilonTracker(action_selector, PARAMS)
    preproc = model.MAgentPreprocessor(device)
    agent = ptan.agent.DQNAgent(net, action_selector, device, preprocessor=preproc)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, PARAMS.gamma, vectorized=True)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, PARAMS.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=PARAMS.learning_rate)

    def process_batch(engine, batch):
        res = {}
        optimizer.zero_grad()
        loss_v = model.calc_loss_dqn(
            batch, net, tgt_net.target_model, preproc,
            gamma=PARAMS.gamma, device=device)
        loss_v.backward()
        optimizer.step()
        if epsilon_tracker is not None:
            epsilon_tracker.frame(engine.state.iteration)
            res['epsilon'] = action_selector.epsilon
        if engine.state.iteration % PARAMS.target_net_sync == 0:
            tgt_net.sync()
        res['loss'] = loss_v.item()
        return res

    engine = Engine(process_batch)
    common.setup_ignite(engine, PARAMS, exp_source, args.name,
                        extra_metrics=('test_reward', 'test_steps'))
    best_test_reward = None

    @engine.on(ptan_ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def test_network(engine):
        net.train(False)
        reward, steps = test_model(net, device, config)
        net.train(True)
        engine.state.metrics['test_reward'] = reward
        engine.state.metrics['test_steps'] = steps
        print("Test done: got %.3f reward after %.2f steps" % (
            reward, steps
        ))

        global best_test_reward
        if best_test_reward is None:
            best_test_reward = reward
        elif best_test_reward < reward:
            print("Best test reward updated %.3f <- %.3f, save model" % (
                best_test_reward, reward
            ))
            best_test_reward = reward
            torch.save(net.state_dict(), os.path.join(saves_path, "best_%.3f.dat" % reward))

    engine.run(common.batch_generator(buffer, PARAMS.replay_initial,
                                      PARAMS.batch_size))
