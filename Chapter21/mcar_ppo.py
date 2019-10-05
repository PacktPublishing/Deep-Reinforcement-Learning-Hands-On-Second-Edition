#!/usr/bin/env python3
import ptan
import ptan.ignite as ptan_ignite
import gym
import argparse
import random
import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine
from types import SimpleNamespace
from lib import common, ppo


HYPERPARAMS = {
    'debug': SimpleNamespace(**{
        'env_name':         "CartPole-v0",
        'stop_reward':      None,
        'stop_test_reward': 190.0,
        'run_name':         'debug',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.9,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
    'ppo': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'ppo',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.99,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
    'noisynet': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'noisynet',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.99,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
    'counts': SimpleNamespace(**{
        'env_name':         "MountainCar-v0",
        'stop_reward':      None,
        'stop_test_reward': -130.0,
        'run_name':         'counts',
        'actor_lr':         1e-4,
        'critic_lr':        1e-4,
        'gamma':            0.99,
        'ppo_trajectory':   2049,
        'ppo_epoches':      10,
        'ppo_eps':          0.2,
        'batch_size':       32,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
        'counts_reward_scale': 0.5,
    }),

    'distill': SimpleNamespace(**{
        'env_name': "MountainCar-v0",
        'stop_reward': None,
        'stop_test_reward': -130.0,
        'run_name': 'distill',
        'actor_lr': 1e-4,
        'critic_lr': 1e-4,
        'gamma': 0.99,
        'ppo_trajectory': 2049,
        'ppo_epoches': 10,
        'ppo_eps': 0.2,
        'batch_size': 32,
        'gae_lambda': 0.95,
        'entropy_beta': 0.1,
        'reward_scale': 100.0,
        'distill_lr': 1e-5,
    }),
}


def counts_hash(obs):
    r = obs.tolist()
    return tuple(map(lambda v: round(v, 3), r))


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-p", "--params", default='ppo', choices=list(HYPERPARAMS.keys()),
                        help="Parameters, default=ppo")
    args = parser.parse_args()
    params = HYPERPARAMS[args.params]

    env = gym.make(params.env_name)
    test_env = gym.make(params.env_name)
    if args.params == 'counts':
        env = common.PseudoCountRewardWrapper(env, reward_scale=params.counts_reward_scale,
                                              hash_function=counts_hash)
    net_distill = None
    if args.params == 'distill':
        net_distill = ppo.MountainCarNetDistillery(env.observation_space.shape[0])
        env = common.NetworkDistillationRewardWrapper(env, net_distill.extra_reward, reward_scale=params.reward_scale)

    env.seed(common.SEED)

    if args.params == 'noisynet':
        net = ppo.MountainCarNoisyNetsPPO(env.observation_space.shape[0], env.action_space.n)
    else:
        net = ppo.MountainCarBasePPO(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net.actor, apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)
    opt_actor = optim.Adam(net.actor.parameters(), lr=params.actor_lr)
    opt_critic = optim.Adam(net.critic.parameters(), lr=params.critic_lr)
    if net_distill is not None:
        opt_distill = optim.Adam(net_distill.trn_net.parameters(), lr=params.distill_lr)

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

        if net_distill is not None:
            opt_distill.zero_grad()
            loss_distill_t = net_distill.loss(states_t)
            loss_distill_t.backward()
            opt_distill.step()
            res['loss_distill'] = loss_distill_t.item()

        return res


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, args.name, extra_metrics=(
        'test_reward', 'avg_test_reward', 'test_steps'))

    @engine.on(ptan_ignite.PeriodEvents.ITERS_1000_COMPLETED)
    def test_network(engine):
        net.actor.train(False)
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
        test_reward_avg = getattr(engine.state, "test_reward_avg", None)
        if test_reward_avg is None:
            test_reward_avg = reward
        else:
            test_reward_avg = test_reward_avg * 0.95 + 0.05 * reward
        engine.state.test_reward_avg = test_reward_avg
        print("Test done: got %.3f reward after %d steps, avg reward %.3f" % (
            reward, steps, test_reward_avg
        ))
        engine.state.metrics['test_reward'] = reward
        engine.state.metrics['avg_test_reward'] = test_reward_avg
        engine.state.metrics['test_steps'] = steps

        if test_reward_avg > params.stop_test_reward:
            print("Reward boundary has crossed, stopping training. Contgrats!")
            engine.should_terminate = True
        net.actor.train(True)

    def new_ppo_batch():
        # In noisy networks we need to reset the noise
        if args.params == 'noisynet':
            net.sample_noise()

    engine.run(ppo.batch_generator(exp_source, net, params.ppo_trajectory,
                                   params.ppo_epoches, params.batch_size,
                                   params.gamma, params.gae_lambda, new_batch_callable=new_ppo_batch))
