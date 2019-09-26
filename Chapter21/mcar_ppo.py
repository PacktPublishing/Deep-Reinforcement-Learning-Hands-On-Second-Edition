#!/usr/bin/env python3
import ptan
import gym
import argparse
import random
import torch
import torch.optim as optim

from ignite.engine import Engine
from types import SimpleNamespace
from lib import common, ppo


HYPERPARAMS = {
    'ppo': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      -120.0,
        'run_name':         'ppo',
        'learning_rate':    1e-4,
        'gamma':            0.99,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'batch_size':       32,
        'gae_lambda':       0.95,
    }),
}


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-p", "--params", default='ppo', help="Parameters, default=ppo")
    args = parser.parse_args()
    params = HYPERPARAMS[args.params]

    env = gym.make(params.env_name)
    env.seed(common.SEED)

    net = ppo.MountainCarBasePPO(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net.actor, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        print(batch)
        assert False
        return {
            "loss": 0.0
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, args.name)
    engine.run(ppo.batch_generator(exp_source, net.critic, net.actor, params.ppo_trajectory,
                                   params.ppo_epoches, params.batch_size,
                                   params.gamma, params.gae_lambda))
