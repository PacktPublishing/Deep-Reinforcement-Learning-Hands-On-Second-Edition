#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "MAgent/python"))
import magent

import argparse
import torch
import numpy as np
from lib import model, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ma", "--model_a", required=True,
                        help="Model file to load in the first group")
    parser.add_argument("-mb", "--model_b", required=True,
                        help="Model file to load in the second group")
    parser.add_argument("--map-size", type=int, default=16,
                        help="Size of the map, default=64")
    parser.add_argument("--render", default="render",
                        help="Directory to store renders, default=render")
    parser.add_argument("--walls-density", type=float, default=0.04,
                        help="Density of walls, default=0.04")
    parser.add_argument("--count_a", type=int, default=20,
                        help="Size of the first group, default=100")
    parser.add_argument("--count_b", type=int, default=20,
                        help="Size of the second group, default=100")
    parser.add_argument("--max-steps", type=int, help="Set limit of steps")

    args = parser.parse_args()

    env = magent.GridWorld('battle', map_size=args.map_size)
    env.set_render_dir(args.render)
    a_handle, b_handle = env.get_handles()

    env.reset()
    env.add_walls(method="random", n=args.map_size *
                                     args.map_size *
                                     args.walls_density)
    env.add_agents(a_handle, method="random", n=args.count_a)
    env.add_agents(b_handle, method="random", n=args.count_b)

    v = env.get_view_space(a_handle)
    v = (v[-1], ) + v[:2]
    net_a = model.DQNModel(v, env.get_feature_space(
        a_handle), env.get_action_space(a_handle)[0])
    net_a.load_state_dict(torch.load(args.model_a))
    print(net_a)

    v = env.get_view_space(b_handle)
    v = (v[-1], ) + v[:2]
    net_b = model.DQNModel(v, env.get_feature_space(
        b_handle), env.get_action_space(b_handle)[0])
    net_b.load_state_dict(torch.load(args.model_b))
    print(net_b)

    a_total_reward = b_total_reward = 0.0
    total_steps = 0

    while True:
        # A actions
        view_obs, feats_obs = env.get_observation(a_handle)
        view_obs = np.array(view_obs)
        feats_obs = np.array(feats_obs)
        view_obs = np.moveaxis(view_obs, 3, 1)
        view_t = torch.tensor(view_obs, dtype=torch.float32)
        feats_t = torch.tensor(feats_obs, dtype=torch.float32)
        qvals = net_a((view_t, feats_t))
        actions = torch.max(qvals, dim=1)[1].cpu().numpy()
        actions = actions.astype(np.int32)
        env.set_action(a_handle, actions)

        view_obs, feats_obs = env.get_observation(b_handle)
        view_obs = np.array(view_obs)
        feats_obs = np.array(feats_obs)
        view_obs = np.moveaxis(view_obs, 3, 1)
        view_t = torch.tensor(view_obs, dtype=torch.float32)
        feats_t = torch.tensor(feats_obs, dtype=torch.float32)
        qvals = net_b((view_t, feats_t))
        actions = torch.max(qvals, dim=1)[1].cpu().numpy()
        actions = actions.astype(np.int32)
        env.set_action(b_handle, actions)

        done = env.step()
        if done:
            break
        env.render()
        env.clear_dead()
        a_total_reward += env.get_reward(a_handle).sum()
        b_total_reward += env.get_reward(b_handle).sum()
        if args.max_steps is not None and args.max_steps <= total_steps:
            break
        total_steps += 1

    print("Average reward: A %.3f, B %.3f" % (
            a_total_reward / args.count_a,
            b_total_reward / args.count_b
    ))
