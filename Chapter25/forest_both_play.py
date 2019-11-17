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
    parser.add_argument("-md", "--model_deer", required=True,
                        help="Model file to load in deer agent")
    parser.add_argument("-mt", "--model_tiger", required=True,
                        help="Model file to load in tiger agent")
    parser.add_argument("--map-size", type=int, default=64,
                        help="Size of the map, default=64")
    parser.add_argument("--render", default="render",
                        help="Directory to store renders, default=render")
    parser.add_argument("--walls-density", type=float, default=0.04,
                        help="Density of walls, default=0.04")
    parser.add_argument("--tigers", type=int, default=10,
                        help="Count of tigers, default=10")
    parser.add_argument("--deers", type=int, default=50,
                        help="Count of deers, default=50")
    parser.add_argument("--mode", default='forest', choices=['forest', 'double_attack'],
                        help="GridWorld mode, could be 'forest' or 'double_attack', default='forest'")

    args = parser.parse_args()

    if args.mode == 'forest':
        config = data.config_forest(args.map_size)
    elif args.mode == 'double_attack':
        config = data.config_double_attack(args.map_size)
    else:
        config = None

    env = magent.GridWorld(config, map_size=args.map_size)
    env.set_render_dir(args.render)
    deer_handle, tiger_handle = env.get_handles()

    env.reset()
    env.add_walls(method="random", n=args.map_size *
                                     args.map_size *
                                     args.walls_density)
    env.add_agents(deer_handle, method="random", n=args.deers)
    env.add_agents(tiger_handle, method="random", n=args.tigers)

    v = env.get_view_space(tiger_handle)
    v = (v[-1], ) + v[:2]
    net_tiger = model.DQNModel(v, env.get_feature_space(
        tiger_handle), env.get_action_space(tiger_handle)[0])
    net_tiger.load_state_dict(torch.load(args.model_tiger))
    print(net_tiger)

    v = env.get_view_space(deer_handle)
    v = (v[-1], ) + v[:2]
    net_deer = model.DQNModel(v, env.get_feature_space(
        deer_handle), env.get_action_space(deer_handle)[0])
    net_deer.load_state_dict(torch.load(args.model_deer))
    print(net_deer)

    deer_total_reward = tiger_total_reward = 0.0

    while True:
        # tiger actions
        view_obs, feats_obs = env.get_observation(tiger_handle)
        view_obs = np.array(view_obs)
        feats_obs = np.array(feats_obs)
        view_obs = np.moveaxis(view_obs, 3, 1)
        view_t = torch.tensor(view_obs, dtype=torch.float32)
        feats_t = torch.tensor(feats_obs, dtype=torch.float32)
        qvals = net_tiger((view_t, feats_t))
        actions = torch.max(qvals, dim=1)[1].cpu().numpy()
        actions = actions.astype(np.int32)
        env.set_action(tiger_handle, actions)

        view_obs, feats_obs = env.get_observation(deer_handle)
        view_obs = np.array(view_obs)
        feats_obs = np.array(feats_obs)
        view_obs = np.moveaxis(view_obs, 3, 1)
        view_t = torch.tensor(view_obs, dtype=torch.float32)
        feats_t = torch.tensor(feats_obs, dtype=torch.float32)
        qvals = net_deer((view_t, feats_t))
        actions = torch.max(qvals, dim=1)[1].cpu().numpy()
        actions = actions.astype(np.int32)
        env.set_action(deer_handle, actions)

        done = env.step()
        if done:
            break
        env.render()
        env.clear_dead()
        tiger_total_reward += env.get_reward(tiger_handle).sum()
        deer_total_reward += env.get_reward(deer_handle).sum()

    print("Average reward: tigers %.3f, deers %.3f" % (
            tiger_total_reward / args.tigers,
            deer_total_reward / args.deers
    ))
