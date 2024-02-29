import gymnasium  as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
    obs = obs[0]

    print(obs)

