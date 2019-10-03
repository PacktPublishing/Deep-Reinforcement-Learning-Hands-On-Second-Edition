#!/usr/bin/env python3
import ptan
import ptan.ignite as ptan_ignite
import gym
import argparse
import random
import itertools
import torch
import torch.optim as optim
import torch.nn.functional as F

from ignite.engine import Engine
from types import SimpleNamespace
from lib import common, ppo, atari_wrappers

N_STEPS = 4
N_ENVS = 3
NAME = "atari"

HYPERPARAMS = {
    'ppo': SimpleNamespace(**{
        'env_name':         "SeaquestNoFrameskip-v4",
        'stop_reward':      None,
        'stop_test_reward': 10000,
        'run_name':         'ppo',
        'lr':               5e-5,
        'gamma':            0.99,
        'ppo_trajectory':   256,
        'ppo_epoches':      4,
        'ppo_eps':          0.2,
        'batch_size':       64,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
    'noisynet': SimpleNamespace(**{
        'env_name':         "SeaquestNoFrameskip-v4",
        'stop_reward':      None,
        'stop_test_reward': 10000,
        'run_name':         'noisynet',
        'lr':               5e-5,
        'gamma':            0.99,
        'ppo_trajectory':   256,
        'ppo_epoches':      4,
        'ppo_eps':          0.2,
        'batch_size':       64,
        'gae_lambda':       0.95,
        'entropy_beta':     0.1,
    }),
}


if __name__ == "__main__":
    random.seed(common.SEED)
    torch.manual_seed(common.SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Run name")
    parser.add_argument("-p", "--params", default='ppo', choices=list(HYPERPARAMS.keys()),
                        help="Parameters, default=ppo")
    args = parser.parse_args()
    params = HYPERPARAMS[args.params]
    device = torch.device("cuda" if args.cuda else "cpu")

    envs = []
    for _ in range(N_ENVS):
        env = atari_wrappers.make_atari(params.env_name, skip_noop=True, skip_maxskip=True)
        env = atari_wrappers.wrap_deepmind(env, pytorch_img=True, frame_stack=True)
        envs.append(env)

    test_env = atari_wrappers.make_atari(params.env_name, skip_noop=True, skip_maxskip=True)
    test_env = atari_wrappers.wrap_deepmind(test_env, pytorch_img=True, frame_stack=True)

    if args.params == 'noisynet':
        net = ppo.AtariNoisyNetsPPO(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = ppo.AtariBasePPO(env.observation_space.shape, env.action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, preprocessor=ptan.agent.float32_preprocessor,
                                   device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)
    optimizer = optim.Adam(net.parameters(), lr=params.lr)

    def process_batch(engine, batch):
        states_t, actions_t, adv_t, ref_t, old_logprob_t = batch

        optimizer.zero_grad()
        policy_t, value_t = net(states_t)
        loss_value_t = F.mse_loss(value_t.squeeze(-1), ref_t)

        logpolicy_t = F.log_softmax(policy_t, dim=1)

        prob_t = F.softmax(policy_t, dim=1)
        loss_entropy_t = (prob_t * logpolicy_t).sum(dim=1).mean()

        logprob_t = logpolicy_t.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        ratio_t = torch.exp(logprob_t - old_logprob_t)
        surr_obj_t = adv_t * ratio_t
        clipped_surr_t = adv_t * torch.clamp(ratio_t, 1.0 - params.ppo_eps, 1.0 + params.ppo_eps)
        loss_policy_t = -torch.min(surr_obj_t, clipped_surr_t).mean()

        loss_t = params.entropy_beta * loss_entropy_t + loss_policy_t + loss_value_t
        loss_t.backward()
        optimizer.step()

        res = {
            "loss": loss_t.item(),
            "loss_value": loss_value_t.item(),
            "loss_policy": loss_policy_t.item(),
            "adv": adv_t.mean().item(),
            "ref": ref_t.mean().item(),
            "loss_entropy": loss_entropy_t.item(),
        }

        return res


    engine = Engine(process_batch)
    common.setup_ignite(engine, params, exp_source, NAME + "_" + args.name, extra_metrics=(
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
                                   params.gamma, params.gae_lambda, device=device,
                                   trim_trajectory=False, new_batch_callable=new_ppo_batch))
