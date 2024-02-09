import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
