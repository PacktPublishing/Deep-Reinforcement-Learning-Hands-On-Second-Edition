import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    obs = env.reset()
    print(env.observation_space.shape)