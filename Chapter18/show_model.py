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
    microtaur.register()
    env = gym.make(microtaur.ENV_ID, render=True)
    print(env)
    obs = env.reset()
    while True:
        env.unwrapped.scene.global_step()
        time.sleep(0.1)
    env.close()
