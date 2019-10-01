#!/usr/bin/env python3
import ptan
import argparse
import random
import numpy as np

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import common, dqn_extra, atari_wrappers

NAME = "baseline"
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 10000
N_STEPS = 4
N_ENVS = 3


@torch.no_grad()
def evaluate_states(states, net, device, engine):
    s_v = torch.tensor(states).to(device)
    adv, val = net.adv_val(s_v)
    engine.state.metrics['adv'] = adv.mean().item()
    engine.state.metrics['val'] = val.mean().item()


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS_DQN['seaquest']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = []
    for _ in range(N_ENVS):
        env = atari_wrappers.make_atari(params.env_name, skip_noop=True, skip_maxskip=True)
        env = atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True, frame_stack_count=2)
        env.seed(common.SEED)
        envs.append(env)

    net = dqn_extra.BaselineDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params.epsilon_start)
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        envs, agent, gamma=params.gamma, steps_count=N_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=params.replay_size)
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss_double_dqn(batch, net, tgt_net.target_model,
                                             gamma=params.gamma**N_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        epsilon_tracker.frame(engine.state.iteration)
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
        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME + "_" + args.name, extra_metrics=('adv', 'val'))
    engine.run(common.batch_generator(buffer, params.replay_initial, params.batch_size))
