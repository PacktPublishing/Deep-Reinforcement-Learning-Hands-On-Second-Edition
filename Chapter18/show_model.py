#!/usr/bin/env python3
import gym
import ptan
import time
import argparse
import torch

from lib import microtaur, ddpg

OBS_HISTORY_STEPS = 4


@torch.no_grad()
def infer(net, obs):
    obs_t = torch.tensor([obs], dtype=torch.float32)
    act_t = net(obs_t)
    return act_t[0].numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rotate", type=int, help="If given, rotate given leg 0..3 back and forth")
    parser.add_argument("-m", "--model", help="Optional model file to load")
    parser.add_argument("--zero-yaw", default=False, action='store_true', help="Pass zero yaw to observation")
    parser.add_argument("-v", "--value", default=0.0, type=float, help="Value to be assigned as action on all legs, default=0.0")
    args = parser.parse_args()

    microtaur.register()
    env = gym.make(microtaur.ENV_ID, render=True, zero_yaw=args.zero_yaw)
    if OBS_HISTORY_STEPS > 1:
        env = ptan.common.wrappers_simple.FrameStack1D(env, OBS_HISTORY_STEPS)

    net = None
    if args.model is not None:
        net = ddpg.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
        print(net)
        net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    obs = env.reset()
    actions = [args.value, args.value, args.value, args.value]
    if net is not None:
        actions = infer(net, obs)
    positions = list(microtaur.generate_positions(0, 10))
    idx = 0
    try:
        while True:
            # get the height from the model
            h = env.unwrapped.robot.get_link_pos()[-1]
            obs, r, *_ = env.step(actions)
            if args.rotate is not None:
                actions[args.rotate] = positions[idx]
            elif net is not None:
                actions = infer(net, obs)
            idx += 1
            idx %= len(positions)
            time.sleep(0.1)
            print(f"r={r}, h={h}, obs[-3:]={obs[-3:]}")
    finally:
        env.close()
