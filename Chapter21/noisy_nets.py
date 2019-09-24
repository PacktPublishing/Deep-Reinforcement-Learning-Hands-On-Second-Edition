#!/usr/bin/env python3
import gym
import ptan
import argparse
import random
import numpy as np

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import common, dqn_extra

NAME = "noisy_nets"
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 10000
NOISY_SNR_EVERY_ITERS = 10000
N_STEPS = 4


@torch.no_grad()
def evaluate_states(states, net, device, engine):
    s_v = torch.tensor(states).to(device)
    adv, val = net.adv_val(s_v)
    engine.state.metrics['adv'] = adv.mean().item()
    engine.state.metrics['val'] = val.mean().item()


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['seaquest']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)

    net = dqn_extra.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=N_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_double_dqn(batch, net, tgt_net.target_model,
                                             gamma=params.gamma**N_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        if engine.state.iteration % EVAL_EVERY_FRAME == 0:
            eval_states = getattr(engine.state, "eval_states", None)
            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)
                engine.state.eval_states = eval_states
            evaluate_states(eval_states, net, device, engine)
        if engine.state.iteration % NOISY_SNR_EVERY_ITERS == 0:
            for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                engine.state.metrics[f'snr_{layer_idx+1}'] = sigma_l2

        return {
            "loss": loss_v.item(),
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME + "_" + args.name,
                        extra_metrics=('adv', 'val', 'snr_1', 'snr_2'))
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
