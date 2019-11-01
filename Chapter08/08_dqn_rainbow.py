#!/usr/bin/env python3
import gym
import ptan
import argparse
import random

import torch
import torch.optim as optim

from ignite.engine import Engine

from lib import common, dqn_extra

NAME = "08_rainbow"
N_STEPS = 4
PRIO_REPLAY_ALPHA = 0.6


def calc_loss_rainbow(batch, batch_weights, net, tgt_net, gamma,
                      device="cpu", double=True):
    states, actions, rewards, dones, next_states = \
        common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_values = net(states_v).gather(1, actions_v)
    state_action_values = state_action_values.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        if double:
            next_state_actions = net(next_states_v).max(1)[1]
            next_state_actions = next_state_actions.unsqueeze(-1)
            next_state_values = tgt_net(next_states_v).gather(
                1, next_state_actions).squeeze(-1)
        else:
            next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = \
            next_state_values.detach() * gamma + rewards_v
    losses_v = (state_action_values -
                expected_state_action_values) ** 2
    losses_v *= batch_weights_v
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()


def calc_loss_prio(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()



if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params.env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    env.seed(common.SEED)

    net = dqn_extra.RainbowDQN(env.observation_space.shape,
                        env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.ArgmaxActionSelector()
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.gamma, steps_count=N_STEPS)
    buffer = dqn_extra.PrioReplayBuffer(
        exp_source, params.replay_size, PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(),
                           lr=params.learning_rate)

    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss_v, sample_prios = calc_loss_prio(
            batch, batch_weights, net, tgt_net.target_model,
            gamma=params.gamma**N_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "beta": buffer.update_beta(engine.state.iteration),
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME)
    engine.run(common.batch_generator(buffer, params.replay_initial,
                                      params.batch_size))
