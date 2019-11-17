#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "MAgent/python"))

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


MAP_SIZE = 64
COUNT_TIGERS = 10
COUNT_DEERS = 50
WALLS_DENSITY = 0.04


PARAMS = SimpleNamespace(**{
    'run_name':         'tigers-deers',
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


def test_model(net_deer: model.DQNModel, net_tiger: model.DQNModel,
               device: torch.device, gw_config) -> Tuple[float, float, float, float]:
    test_env = magent.GridWorld(gw_config, map_size=MAP_SIZE)
    deer_handle, tiger_handle = test_env.get_handles()

    def reset_env():
        test_env.reset()
        test_env.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * WALLS_DENSITY)
        test_env.add_agents(deer_handle, method="random", n=COUNT_DEERS)
        test_env.add_agents(tiger_handle, method="random", n=COUNT_TIGERS)

    deer_env = data.MAgentEnv(test_env, deer_handle, reset_env_func=reset_env, is_slave=True)
    tiger_env = data.MAgentEnv(test_env, tiger_handle, reset_env_func=reset_env)
    preproc = model.MAgentPreprocessor(device)
    deer_agent = ptan.agent.DQNAgent(net_deer, ptan.actions.ArgmaxActionSelector(), device, preprocessor=preproc)
    tiger_agent = ptan.agent.DQNAgent(net_tiger, ptan.actions.ArgmaxActionSelector(), device, preprocessor=preproc)

    t_obs = tiger_env.reset()
    d_obs = deer_env.reset()
    deer_steps = 0
    deer_rewards = 0.0
    tiger_steps = 0
    tiger_rewards = 0.0

    while True:
        d_actions = deer_agent(d_obs)[0]
        t_actions = tiger_agent(t_obs)[0]
        d_obs, d_r, d_dones, _ = deer_env.step(d_actions)
        t_obs, t_r, t_dones, _ = tiger_env.step(t_actions)
        tiger_steps += len(t_obs)
        tiger_rewards += sum(t_r)
        if t_dones[0]:
            break
        deer_steps += len(d_obs)
        deer_rewards += sum(d_r)
        if d_dones[0]:
            break

    return deer_rewards / COUNT_DEERS, deer_steps / COUNT_DEERS, \
           tiger_rewards / COUNT_TIGERS, tiger_steps / COUNT_TIGERS


def batches_generator(buffers: List[ptan.experience.ExperienceReplayBuffer],
                      replay_initial: int, batch_size: int):
    while True:
        for buf in buffers:
            buf.populate(1)
        if len(buffers[0]) < replay_initial:
            continue
        batches = [
            buf.sample(batch_size)
            for buf in buffers
        ]
        yield batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA computations")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("--mode", default='forest', choices=['forest', 'double_attack'],
                        help="GridWorld mode, could be 'forest' or 'double_attack', default='forest'")
    args = parser.parse_args()

    # tweak count of agents in this mode to simplify exploration
    if args.mode == 'double_attack':
        COUNT_TIGERS = 20
        COUNT_DEERS = 512
        config = data.config_double_attack(MAP_SIZE)
    else:
        config = data.config_forest(MAP_SIZE)

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

    deer_env = data.MAgentEnv(m_env, deer_handle, reset_env_func=lambda: None, is_slave=True)
    tiger_env = data.MAgentEnv(m_env, tiger_handle, reset_env_func=reset_env, is_slave=False)

    deer_obs = data.MAgentEnv.handle_obs_space(m_env, deer_handle)
    tiger_obs = data.MAgentEnv.handle_obs_space(m_env, tiger_handle)

    net_deer = model.DQNModel(
        deer_obs.spaces[0].shape, deer_obs.spaces[1].shape,
        m_env.get_action_space(deer_handle)[0]).to(device)
    tgt_net_deer = ptan.agent.TargetNet(net_deer)
    print(net_deer)

    net_tiger = model.DQNModel(
        tiger_obs.spaces[0].shape, tiger_obs.spaces[1].shape,
        m_env.get_action_space(tiger_handle)[0]).to(device)
    tgt_net_tiger = ptan.agent.TargetNet(net_tiger)
    print(net_tiger)

    action_selector = ptan.actions.EpsilonGreedyActionSelector(
        epsilon=PARAMS.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(action_selector, PARAMS)
    preproc = model.MAgentPreprocessor(device)

    deer_agent = ptan.agent.DQNAgent(net_deer, action_selector, device, preprocessor=preproc)
    tiger_agent = ptan.agent.DQNAgent(net_tiger, action_selector, device, preprocessor=preproc)
    deer_exp_source = ptan.experience.ExperienceSourceFirstLast(
        deer_env, deer_agent, PARAMS.gamma, vectorized=True)
    tiger_exp_source = ptan.experience.ExperienceSourceFirstLast(
        tiger_env, tiger_agent, PARAMS.gamma, vectorized=True)
    deer_buffer = ptan.experience.ExperienceReplayBuffer(deer_exp_source, PARAMS.replay_size)
    tiger_buffer = ptan.experience.ExperienceReplayBuffer(tiger_exp_source, PARAMS.replay_size)
    deer_optimizer = optim.Adam(net_deer.parameters(), lr=PARAMS.learning_rate)
    tiger_optimizer = optim.Adam(net_tiger.parameters(), lr=PARAMS.learning_rate)

    def process_batches(engine, batches):
        res = {}
        loss = 0.0
        for name, batch, opt, net, tgt_net in zip(
                ["deer", "tiger"],
                batches, [deer_optimizer, tiger_optimizer],
                [net_deer, net_tiger],
                [tgt_net_deer, tgt_net_tiger]):
            opt.zero_grad()
            loss_v = model.calc_loss_dqn(
                batch, net, tgt_net.target_model, preproc,
                gamma=PARAMS.gamma, device=device)
            loss_v.backward()
            opt.step()
            res[name + "_loss"] = loss_v.item()
            loss += loss_v.item()
            if engine.state.iteration % PARAMS.target_net_sync == 0:
                tgt_net.sync()

        epsilon_tracker.frame(engine.state.iteration)
        res['epsilon'] = action_selector.epsilon
        res['loss'] = loss
        return res

    engine = Engine(process_batches)
    common.setup_ignite(engine, PARAMS, tiger_exp_source, args.name,
                        extra_metrics=('test_reward_deer', 'test_steps_deer', 'test_reward_tiger', 'test_steps_tiger'))
    best_test_reward_deer = None
    best_test_reward_tiger = None

    @engine.on(ptan_ignite.PeriodEvents.ITERS_10000_COMPLETED)
    def test_network(engine):
        net_deer.train(False)
        net_tiger.train(False)
        deer_reward, deer_steps, tiger_reward, tiger_steps = test_model(net_deer, net_tiger, device, config)
        net_deer.train(True)
        net_tiger.train(True)
        engine.state.metrics['test_reward_deer'] = deer_reward
        engine.state.metrics['test_steps_deer'] = deer_steps
        engine.state.metrics['test_reward_tiger'] = tiger_reward
        engine.state.metrics['test_steps_tiger'] = tiger_steps
        print("Test done: Deers got %.3f reward after %.2f steps, tigers %.3f reward after %.2f steps" % (
            deer_reward, deer_steps, tiger_reward, tiger_steps
        ))

        global best_test_reward_deer, best_test_reward_tiger

        if best_test_reward_deer is None:
            best_test_reward_deer = deer_reward
        elif best_test_reward_deer < deer_reward:
            print("Best test deer reward updated %.3f <- %.3f, save model" % (
                best_test_reward_deer, deer_reward
            ))
            best_test_reward_deer = deer_reward
            torch.save(net_deer.state_dict(), os.path.join(saves_path, "deer_best_%.3f.dat" % deer_reward))

        if best_test_reward_tiger is None:
            best_test_reward_tiger = tiger_reward
        elif best_test_reward_tiger < tiger_reward:
            print("Best test tiger reward updated %.3f <- %.3f, save model" % (
                best_test_reward_tiger, tiger_reward
            ))
            best_test_reward_tiger = tiger_reward
            torch.save(net_tiger.state_dict(), os.path.join(saves_path, "tiger_best_%.3f.dat" % tiger_reward))

    engine.run(batches_generator(
        [deer_buffer, tiger_buffer],
        PARAMS.replay_initial, PARAMS.batch_size))
