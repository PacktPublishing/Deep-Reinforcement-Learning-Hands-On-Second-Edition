import gym
from gymnasium.wrappers import RecordVideo
from collections import namedtuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import pdb

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 70


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    obs = obs[0]
    sm = nn.Softmax(dim=0)

    while True:
        obs_v = torch.FloatTensor(obs)
        out = net(obs_v)
        act_probs_v = sm(out)

        # extracting the probabilities from the net output tensor 
        act_probs = act_probs_v.data.numpy()
        # choose random between the probs and select as action
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, truncated, info = env.step(action)
        episode_reward += reward
        step = EpisodeStep(observation=obs, action=action)

        episode_steps.append(step)

        if is_done:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            next_obs = next_obs[0]
            if len(batch) == batch_size:
                breakpoint()
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    
    for reward, steps in batch:
        if reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, steps))
        train_act.extend(map(lambda step: step.action, steps))

    train_obs_v = torch.FloatTensor(train_obs)
    train_act_v = torch.LongTensor(train_act)

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, "records")
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    # obs size is 4
    obs_size = env.observation_space.shape[0]
    # number of actions is 2 -> Discrete(2)
    n_actions = env.action_space.n

    #create network
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="-cartpole")

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        breakpoint()

        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)

        # calculate the gradients
        loss_v.backward()
        # update the parameters
        optimizer.step()

        print(f"{iter_no}: loss={loss_v.item()}, reward_mean={reward_m}, rw_bound={reward_b}")
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 199:
            print("Solved!")
            break

    writer.close()