import random
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch


class InputWrapper(gym.ObservationWrapper):
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(
            observation, (64, 64))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)

envs = [InputWrapper(gym.make(name)) for name in ('Breakout-v4', 'AirRaid-v4', 'Pong-v4')]

def iterate_batches(envs, batch_size=3):
    batch = [e.reset() for e in envs]
    print(batch)
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        obs, reward, is_done, truncated, info = e.step(e.action_space.sample())
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()
        if is_done:
            e.reset()
            break

batch = next(iterate_batches(envs))
print(batch.shape)  # Assuming you want to print the shape of the batch
# Perform further processing or analysis on the batch here
