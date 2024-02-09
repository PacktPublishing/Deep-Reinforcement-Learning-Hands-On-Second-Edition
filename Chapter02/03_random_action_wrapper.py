import gymnasium as gym
from typing import TypeVar
import random

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


if __name__ == "__main__":
    env = RandomActionWrapper(gym.make("CartPole-v1"))

    observation, info = env.reset()
    total_reward = 0.0

    while True:
        observation, reward, done, truncated, info = env.step(0)
        total_reward += reward
        if done:
            break

    print("Reward got: %.2f" % total_reward)
