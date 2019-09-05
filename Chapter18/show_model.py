#!/usr/bin/env python3
import gym
import ptan
import time
import json
import pathlib
import argparse
import pybullet as p
import numpy as np

from lib import microtaur


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rotate", type=int, help="If given, rotate given leg 0..3 back and forth")
    args = parser.parse_args()

    microtaur.register()
    env = gym.make(microtaur.ENV_ID, render=True)
    obs = env.reset()
    actions = [0, 0, 0, 0]
    positions = list(microtaur.generate_positions(0, 10))
    idx = 0
    while True:
        obs, r, *_ = env.step(actions)
        if args.rotate is not None:
            actions[args.rotate] = positions[idx]
        idx += 1
        idx %= len(positions)
        time.sleep(0.1)
        print(obs[-3:])
    env.close()
