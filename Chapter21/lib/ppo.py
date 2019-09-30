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
    final_states = []
    for (exp,) in exp_source:
        trj_states.append(exp.state)
        trj_actions.append(exp.action)
        trj_rewards.append(exp.reward)
        trj_dones.append(exp.done)
        if exp.done:
            final_states.append(np.array(exp.state, copy=True))
        if len(trj_states) < trajectory_size:
            continue
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
        # drop last entry in prob and trajectory
        old_logprob_t = old_logprob_t[:-1].detach()

        # generate needed amount of batches
        for _ in range(ppo_epoches):
            for batch_ofs in range(1, len(trj_states)-batch_size, batch_size):
                yield (
                    trj_states_t[batch_ofs:batch_ofs + batch_size],
                    trj_actions_t[batch_ofs:batch_ofs + batch_size],
                    adv_t[batch_ofs:batch_ofs + batch_size],
                    ref_t[batch_ofs:batch_ofs + batch_size],
                    old_logprob_t[batch_ofs:batch_ofs + batch_size],
                    final_states,
                )

        trj_states.clear()
        trj_actions.clear()
        trj_rewards.clear()
        trj_dones.clear()
        final_states.clear()
