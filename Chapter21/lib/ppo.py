import ptan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from . import dqn_extra


class MountainCarBasePPO(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 64):
        super(MountainCarBasePPO, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


class MountainCarNoisyNetsPPO(nn.Module):
    def __init__(self, obs_size, n_actions, hid_size: int = 128):
        super(MountainCarNoisyNetsPPO, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            dqn_extra.NoisyLinear(hid_size, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)


def calc_adv_ref(values, dones, rewards, gamma, gae_lambda):
    last_gae = 0.0
    adv, ref = [], []

    for val, next_val, done, reward in zip(reversed(values[:-1]), reversed(values[1:]),
                                           reversed(dones[:-1]), reversed(rewards[:-1])):
        if done:
            delta = reward - val
            last_gae = delta
        else:
            delta = reward + gamma * next_val - val
            last_gae = delta + gamma * gae_lambda * last_gae
        adv.append(last_gae)
        ref.append(last_gae + val)
    adv = list(reversed(adv))
    ref = list(reversed(ref))
    return torch.FloatTensor(adv), torch.FloatTensor(ref)


def batch_generator(exp_source: ptan.experience.ExperienceSource,
                    critic_net: nn.Module, actor_net: nn.Module,
                    trajectory_size: int, ppo_epoches: int,
                    batch_size: int, gamma: float, gae_lambda: float,
                    device: Union[torch.device, str] = "cpu"):
    trj_states = []
    trj_actions = []
    trj_rewards = []
    trj_dones = []
    last_done_index = None
    for (exp,) in exp_source:
        trj_states.append(exp.state)
        trj_actions.append(exp.action)
        trj_rewards.append(exp.reward)
        trj_dones.append(exp.done)
        if exp.done:
            last_done_index = len(trj_states)-1
        if len(trj_states) < trajectory_size:
            continue
        # ensure that we have at least one full episode in the trajectory
        if last_done_index is None or last_done_index == len(trj_states)-1:
            continue

        # trim the trajectory till the last done plus one step (which will be discarded).
        # This increases convergence speed and stability
        trj_states = trj_states[:last_done_index+2]
        trj_actions = trj_actions[:last_done_index + 2]
        trj_rewards = trj_rewards[:last_done_index + 2]
        trj_dones = trj_dones[:last_done_index + 2]

        trj_states_t = torch.FloatTensor(trj_states).to(device)
        trj_actions_t = torch.tensor(trj_actions).to(device)
        trj_values_t = critic_net(trj_states_t).squeeze()

        adv_t, ref_t = calc_adv_ref(trj_values_t.data.cpu().numpy(),
                                    trj_dones, trj_rewards, gamma, gae_lambda)
        adv_t = adv_t.to(device)
        ref_t = ref_t.to(device)

        policy_t = actor_net(trj_states_t)
        logpolicy_t = F.log_softmax(policy_t, dim=1)
        old_logprob_t = logpolicy_t.gather(1, trj_actions_t.unsqueeze(-1)).squeeze(-1)
        adv_t = (adv_t - torch.mean(adv_t)) / torch.std(adv_t)
        old_logprob_t = old_logprob_t.detach()

        # make our trajectory splittable on even batch chunks
        trj_len = len(trj_states) - 1
        trj_len -= trj_len % batch_size
        trj_len += 1
        indices = np.arange(0, trj_len-1)

        # generate needed amount of batches
        for _ in range(ppo_epoches):
            np.random.shuffle(indices)
            for batch_indices in np.split(indices, trj_len // batch_size):
                yield (
                    trj_states_t[batch_indices],
                    trj_actions_t[batch_indices],
                    adv_t[batch_indices],
                    ref_t[batch_indices],
                    old_logprob_t[batch_indices],
                )

        trj_states.clear()
        trj_actions.clear()
        trj_rewards.clear()
        trj_dones.clear()


class MountainCarNetDistillery(nn.Module):
    def __init__(self, obs_size: int, hid_size: int = 128):
        super(MountainCarNetDistillery, self).__init__()

        self.ref_net = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )
        self.ref_net.train(False)

        self.trn_net = nn.Sequential(
            nn.Linear(obs_size, 1),
        )

    def forward(self, x):
        return self.ref_net(x), self.trn_net(x)

    def extra_reward(self, obs):
        r1, r2 = self.forward(torch.FloatTensor([obs]))
        return (r1 - r2).abs().detach().numpy()[0][0]

    def loss(self, obs_t):
        r1_t, r2_t = self.forward(obs_t)
        return F.mse_loss(r2_t, r1_t).mean()


