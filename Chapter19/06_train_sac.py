#!/usr/bin/env python3
import os
import ptan
import gym
import time
from tensorboardX import SummaryWriter

from lib import model, common, test_net, make_parser, parse_args

import torch
import torch.optim as optim
import torch.nn.functional as F

GAMMA = 0.99
BATCH_SIZE = 64
LR_ACTS = 1e-4
LR_VALS = 1e-4
REPLAY_SIZE = 100000
REPLAY_INITIAL = 10000
SAC_ENTROPY_ALPHA = 0.1

TEST_ITERS = 10000

if __name__ == "__main__":

    parser = make_parser()

    args, device, save_path, test_env, maxeps = parse_args(parser)

    env = gym.make(args.env)

    act_net = model.ModelActor( env.observation_space.shape[0], env.action_space.shape[0], args.hid).to(device)
    crt_net = model.ModelCritic( env.observation_space.shape[0], args.hid).to(device)
    twinq_net = model.ModelSACTwinQ( env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    print(twinq_net)

    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    writer = SummaryWriter(comment="-sac_" + args.name)
    agent = model.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=REPLAY_SIZE)
    act_opt = optim.Adam(act_net.parameters(), lr=LR_ACTS)
    crt_opt = optim.Adam(crt_net.parameters(), lr=LR_VALS)
    twinq_opt = optim.Adam(twinq_net.parameters(), lr=LR_VALS)

    frame_idx = 0
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(
                writer, batch_size=10) as tb_tracker:
            while True:

                if len(tracker.total_rewards) >= maxeps:
                    break

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
                states_v, actions_v, ref_vals_v, ref_q_v = \
                    common.unpack_batch_sac(
                        batch, tgt_crt_net.target_model,
                        twinq_net, act_net, GAMMA,
                        SAC_ENTROPY_ALPHA, device)

                tb_tracker.track("ref_v", ref_vals_v.mean(), frame_idx)
                tb_tracker.track("ref_q", ref_q_v.mean(), frame_idx)

                # train TwinQ
                twinq_opt.zero_grad()
                q1_v, q2_v = twinq_net(states_v, actions_v)
                q1_loss_v = F.mse_loss(q1_v.squeeze(),
                                       ref_q_v.detach())
                q2_loss_v = F.mse_loss(q2_v.squeeze(),
                                       ref_q_v.detach())
                q_loss_v = q1_loss_v + q2_loss_v
                q_loss_v.backward()
                twinq_opt.step()
                tb_tracker.track("loss_q1", q1_loss_v, frame_idx)
                tb_tracker.track("loss_q2", q2_loss_v, frame_idx)

                # Critic
                crt_opt.zero_grad()
                val_v = crt_net(states_v)
                v_loss_v = F.mse_loss(val_v.squeeze(),
                                      ref_vals_v.detach())
                v_loss_v.backward()
                crt_opt.step()
                tb_tracker.track("loss_v", v_loss_v, frame_idx)

                # Actor
                act_opt.zero_grad()
                acts_v = act_net(states_v)
                q_out_v, _ = twinq_net(states_v, acts_v)
                act_loss = -q_out_v.mean()
                act_loss.backward()
                act_opt.step()
                tb_tracker.track("loss_act", act_loss, frame_idx)

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

    pass
