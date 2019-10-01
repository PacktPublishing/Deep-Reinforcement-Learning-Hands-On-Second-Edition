#!/usr/bin/env python3
import ptan
import ptan.ignite as ptan_ignite
import gym
import argparse
import random
import warnings
import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine
from types import SimpleNamespace
from lib import common, ppo, dqn_extra


def counts_hash(obs):
    r = obs.tolist()
    return tuple(map(lambda v: round(v, 3), r))


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-p", "--params", default='ppo', help="Parameters, default=ppo")
    args = parser.parse_args()
    params = common.HYPERPARAMS_PPO[args.params]

    env = gym.make(params.env_name)
    test_env = gym.make(params.env_name)
    if args.params == 'counts':
        env = dqn_extra.PseudoCountRewardWrapper(env, reward_scale=params.counts_reward_scale,
                                                 hash_function=counts_hash)
    env.seed(common.SEED)

    if args.params == 'noisynets':
        net = ppo.MountainCarNoisyNetsPPO(env.observation_space.shape[0], env.action_space.n)
    else:
        net = ppo.MountainCarBasePPO(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net.actor, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)
    opt_actor = optim.Adam(net.actor.parameters(), lr=params.actor_lr)
    opt_critic = optim.Adam(net.critic.parameters(), lr=params.critic_lr)

    def process_batch(engine, batch):
        states_t, actions_t, adv_t, ref_t, old_logprob_t = batch

        opt_critic.zero_grad()
        value_t = net.critic(states_t)
        loss_value_t = F.mse_loss(value_t.squeeze(-1), ref_t)
        loss_value_t.backward()
        opt_critic.step()

        opt_actor.zero_grad()
        policy_t = net.actor(states_t)
        logpolicy_t = F.log_softmax(policy_t, dim=1)

        prob_t = F.softmax(policy_t, dim=1)
        loss_entropy_t = (prob_t * logpolicy_t).sum(dim=1).mean()

        logprob_t = logpolicy_t.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        ratio_t = torch.exp(logprob_t - old_logprob_t)
        surr_obj_t = adv_t * ratio_t
        clipped_surr_t = adv_t * torch.clamp(ratio_t, 1.0 - params.ppo_eps, 1.0 + params.ppo_eps)
        loss_policy_t = -torch.min(surr_obj_t, clipped_surr_t).mean()
        loss_polent_t = params.entropy_beta * loss_entropy_t + loss_policy_t
        loss_polent_t.backward()
        opt_actor.step()

        res = {
            "loss": loss_value_t.item() + loss_polent_t.item(),
            "loss_value": loss_value_t.item(),
            "loss_policy": loss_policy_t.item(),
            "adv": adv_t.mean().item(),
            "ref": ref_t.mean().item(),
            "loss_entropy": loss_entropy_t.item(),
        }
        return res


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, args.name, extra_metrics=('test_reward', 'test_steps'))

    @engine.on(ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def test_network(engine):
        obs = test_env.reset()
        reward = 0.0
        steps = 0

        while True:
            acts, _ = agent([obs])
            obs, r, is_done, _ = test_env.step(acts[0])
            reward += r
            steps += 1
            if is_done:
                break
        print("Test done: got %.3f reward after %d steps" % (
            reward, steps
        ))
        engine.state.metrics['test_reward'] = reward
        engine.state.metrics['test_steps'] = steps

    engine.run(ppo.batch_generator(exp_source, net.critic, net.actor, params.ppo_trajectory,
                                   params.ppo_epoches, params.batch_size,
                                   params.gamma, params.gae_lambda))
