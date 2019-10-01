#!/usr/bin/env python3
import ptan
import ptan.ignite as ptan_ignite
import gym
import argparse
import random
import torch
import torch.optim as optim
from types import SimpleNamespace

from ignite.engine import Engine

from lib import common, dqn_extra


HYPERPARAMS = {
    'egreedy': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      -120.0,
        'run_name':         'egreedy',
        'replay_size':      100000,
        'replay_initial':   100,
        'target_net_sync':  100,
        'epsilon_frames':   10**5,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': False,
    }),
    'egreedy-long': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      -120.0,
        'run_name':         'egreedy-long',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  100,
        'epsilon_frames':   10 ** 6,
        'epsilon_start':    1.0,
        'epsilon_final':    0.02,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': True,
    }),
    'noisynet': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      -120.0,
        'run_name':         'noisynet',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  1000,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'eps_decay_trigger': False,
    }),
    'counts': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      -120.0,
        'run_name':         'counts',
        'replay_size':      100000,
        'replay_initial':   1000,
        'target_net_sync':  1000,
        'learning_rate':    0.0001,
        'gamma':            0.99,
        'batch_size':       32,
        'counts_reward_scale': 0.5,
        'eps_decay_trigger': False,
    }),
}

N_STEPS = 4


def counts_hash(obs):
    r = obs.tolist()
    return tuple(map(lambda v: round(v, 3), r))


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-p", "--params", default='egreedy', choices=list(HYPERPARAMS.keys()),
                        help="Parameters, default=egreedy")
    args = parser.parse_args()

    params = HYPERPARAMS[args.params]

    env = gym.make(params.env_name)
    if args.params == 'counts':
        env = common.PseudoCountRewardWrapper(env, reward_scale=params.counts_reward_scale, hash_function=counts_hash)
    env.seed(common.SEED)
    if args.params.startswith("egreedy") or args.params == 'counts':
        net = dqn_extra.MountainCarBaseDQN(env.observation_space.shape[0], env.action_space.n)
    elif args.params == 'noisynet':
        net = dqn_extra.MountainCarNoisyNetDQN(env.observation_space.shape[0], env.action_space.n)
    tgt_net = ptan.agent.TargetNet(net)
    print(net)

    if args.params.startswith('egreedy'):
        selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
        epsilon_tracker = common.EpsilonTracker(selector, params)
        training_enabled = not params.eps_decay_trigger
        epsilon_tracker_frame = 0
    else:
        selector = ptan.actions.ArgmaxActionSelector()
        training_enabled = True

    agent = ptan.agent.DQNAgent(net, selector, preprocessor=ptan.agent.float32_preprocessor)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=N_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        if not training_enabled:
            return {
                "loss": 0.0,
                "epsilon": selector.epsilon
            }

        optimizer.zero_grad()
        loss_v = common.calc_loss_double_dqn(batch, net, tgt_net.target_model,
                                             gamma=params.gamma**N_STEPS)
        loss_v.backward()
        optimizer.step()
        res = {
            "loss": loss_v.item(),
            "epsilon": 0.0,
        }
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()

        if args.params.startswith("egreedy"):
            epsilon_tracker.frame(engine.state.iteration - epsilon_tracker_frame)
            res['epsilon'] = selector.epsilon
        return res

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, args.name)

    @engine.on(ptan_ignite.EpisodeEvents.EPISODE_COMPLETED)
    def check_reward_trigger(trainer: Engine):
        global training_enabled, epsilon_tracker_frame
        if training_enabled:
            return
        # check trigger condition to enable epsilon decay
        if trainer.state.episode_reward > -200:
            training_enabled = True
            epsilon_tracker_frame = trainer.state.iteration
            print("Epsilon decay triggered!")

    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
