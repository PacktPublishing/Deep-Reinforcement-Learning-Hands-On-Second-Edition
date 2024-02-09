import gymnasium as gym
from gymnasium.wrappers import RecordVideo

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    # env = gym.wrappers.Monitor(env, "recording")
    env = RecordVideo(env, 'video')

    total_reward = 0.0
    total_steps = 0
    observation, info = env.reset()
    env.start_video_recorder()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        env.render()
        if done:
            break

    print("Episode done in %d steps, total reward %.2f" % (
        total_steps, total_reward))
    ####
    # Don't forget to close the video recorder before the env!
    env.close_video_recorder()
    env.close()
