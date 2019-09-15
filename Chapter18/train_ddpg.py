#!/usr/bin/env python3
import gym
import gym.wrappers
import ptan
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import os

from lib import microtaur, ddpg


TIME_LIMIT = 1000
REPLAY_SIZE = 100000
OBS_HISTORY_STEPS = 4
LEARNING_RATE = 1e-4
GAMMA = 0.9
BATCH_SIZE = 64
REPLAY_INITIAL = 10000
TEST_ITERS = 1000


def make_env(reward_scheme: microtaur.RewardScheme, zero_yaw: bool = False):
    env = gym.make(microtaur.ENV_ID, reward_scheme=reward_scheme, zero_yaw = zero_yaw)
    assert isinstance(env, gym.wrappers.TimeLimit)
    env._max_episode_steps = TIME_LIMIT
    if OBS_HISTORY_STEPS > 1:
        env = ptan.common.wrappers_simple.FrameStack1D(env, OBS_HISTORY_STEPS)
    return env


def test_net(net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


if __name__ == "__main__":
    microtaur.register()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action='store_true', help="Use cuda for training")
    reward_schemes = [r.name for r in microtaur.RewardScheme]
    parser.add_argument("--reward", default='Height', choices=reward_schemes,
                        help="Reward scheme to use, one of: %s" % reward_schemes)
    parser.add_argument("--zero-yaw", default=False, action='store_true', help="Pass zero yaw to observation")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "ddpg-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = make_env(microtaur.RewardScheme[args.reward], zero_yaw=args.zero_yaw)
    test_env = make_env(microtaur.RewardScheme[args.reward], zero_yaw=args.zero_yaw)
    print("Env: %s, obs=%s, act=%s" % (env, env.observation_space, env.action_space))

    act_net = ddpg.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = ddpg.DDPGCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    print("Actor: %s" % act_net)
    print("Critic: %s" % crt_net)

    writer = SummaryWriter(comment="-ddpg_" + args.name)
    agent = ddpg.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LEARNING_RATE)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                frame_idx += 1
                buffer.populate(1)
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frame_idx)
                    tracker.reward(rewards[0], frame_idx)

                if len(buffer) < REPLAY_INITIAL:
                    continue

                batch = buffer.sample(BATCH_SIZE)
                states_v, actions_v, rewards_v, dones_mask, last_states_v = ddpg.unpack_batch_ddpg(batch, device)

                # train critic
                crt_opt.zero_grad()
                q_v = crt_net(states_v, actions_v)
                last_act_v = tgt_act_net.target_model(last_states_v)
                q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
                tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)

                # train actor
                act_opt.zero_grad()
                cur_actions_v = act_net(states_v)
                actor_loss_v = -crt_net(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                if frame_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(act_net, test_env, device=device)
                    print("Test done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, frame_idx)
                    writer.add_scalar("test_steps", steps, frame_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, frame_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(act_net.state_dict(), fname)
                        best_reward = rewards
