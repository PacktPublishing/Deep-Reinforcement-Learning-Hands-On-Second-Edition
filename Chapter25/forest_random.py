#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "MAgent/python"))

import magent
from magent.builtin.rule_model import RandomActor

MAP_SIZE = 64


if __name__ == "__main__":
    env = magent.GridWorld("forest", map_size=MAP_SIZE)
    env.set_render_dir("render")

    # two groups of animal
    deer_handle, tiger_handle = env.get_handles()

    # init two models
    models = [
        RandomActor(env, deer_handle, tiger_handle),
        RandomActor(env, tiger_handle, deer_handle),
    ]

    env.reset()
    env.add_walls(method="random", n=MAP_SIZE * MAP_SIZE * 0.04)
    env.add_agents(deer_handle, method="random", n=5)
    env.add_agents(tiger_handle, method="random", n=2)

    v = env.get_view_space(tiger_handle)
    r = env.get_feature_space(tiger_handle)
    vv = env.get_view_space(deer_handle)
    rr = env.get_feature_space(deer_handle)

    env.clear_dead()

    done = False
    step_idx = 0
    while not done:
        deer_obs = env.get_observation(deer_handle)
        tiger_obs = env.get_observation(tiger_handle)
        deer_act = models[0].infer_action(deer_obs)
        tiger_act = models[1].infer_action(tiger_obs)
        print("%d: HP deers:  %s" % (step_idx, deer_obs[0][:,1,1,2]))
        print("%d: HP tigers: %s" % (step_idx, tiger_obs[0][:,4,4,2]))

        env.set_action(deer_handle, deer_act)
        env.set_action(tiger_handle, tiger_act)
        env.render()
        done = env.step()
        t_reward = env.get_reward(tiger_handle)
        d_reward = env.get_reward(deer_handle)
        print("Rewards: deer %s, tiger %s" % (d_reward, t_reward))
        step_idx += 1
