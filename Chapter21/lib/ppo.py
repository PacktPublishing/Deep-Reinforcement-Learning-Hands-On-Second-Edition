import ptan
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Optional

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

        self.noisy_layers = [
            dqn_extra.NoisyLinear(hid_size, n_actions)
        ]

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            self.noisy_layers[0],
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def sample_noise(self):
        for l in self.noisy_layers:
            l.sample_noise()


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
                    net: nn.Module,
                    trajectory_size: int, ppo_epoches: int,
                    batch_size: int, gamma: float, gae_lambda: float,
                    device: Union[torch.device, str] = "cpu", trim_trajectory: bool = True,
                    new_batch_callable: Optional[Callable] = None):
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

        if new_batch_callable is not None:
            new_batch_callable()

        # trim the trajectory till the last done plus one step (which will be discarded).
        # This increases convergence speed and stability
        if trim_trajectory:
            trj_states = trj_states[:last_done_index+2]
            trj_actions = trj_actions[:last_done_index + 2]
            trj_rewards = trj_rewards[:last_done_index + 2]
            trj_dones = trj_dones[:last_done_index + 2]

        trj_states_t = torch.FloatTensor(trj_states).to(device)
        trj_actions_t = torch.tensor(trj_actions).to(device)
        policy_t, trj_values_t = net(trj_states_t)
        trj_values_t = trj_values_t.squeeze()

        adv_t, ref_t = calc_adv_ref(trj_values_t.data.cpu().numpy(),
                                    trj_dones, trj_rewards, gamma, gae_lambda)
        adv_t = adv_t.to(device)
        ref_t = ref_t.to(device)

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


def batch_generator_distill(exp_source: ptan.experience.ExperienceSource,
                            net: nn.Module, trajectory_size: int, ppo_epoches: int,
                            batch_size: int, gamma: float, gae_lambda: float,
                            device: Union[torch.device, str] = "cpu", trim_trajectory: bool = True,
                            new_batch_callable: Optional[Callable] = None):
    """
    Same logic as batch_generator, but with distillery networks
    """
    trj_states = []
    trj_actions = []
    trj_rewards = []
    trj_rewards_ext = []
    trj_rewards_int = []
    trj_dones = []
    last_done_index = None
    trj_time = time.time()
    for (exp,) in exp_source:
        trj_states.append(exp.state)
        trj_actions.append(exp.action)
        trj_rewards_ext.append(exp.reward[0])
        trj_rewards_int.append(exp.reward[1])
        trj_rewards.append(exp.reward.sum())
        trj_dones.append(exp.done)
        if exp.done:
            last_done_index = len(trj_states)-1
        if len(trj_states) < trajectory_size:
            continue
        # ensure that we have at least one full episode in the trajectory
        if last_done_index is None or last_done_index == len(trj_states)-1:
            continue

        trj_dt = time.time() - trj_time

        if new_batch_callable is not None:
            new_batch_callable()

        prep_ts = time.time()
        # trim the trajectory till the last done plus one step (which will be discarded).
        # This increases convergence speed and stability
        if trim_trajectory:
            trj_states = trj_states[:last_done_index+2]
            trj_actions = trj_actions[:last_done_index + 2]
            trj_rewards_ext = trj_rewards_ext[:last_done_index + 2]
            trj_rewards_int = trj_rewards_int[:last_done_index + 2]
            trj_rewards = trj_rewards[:last_done_index + 2]
            trj_dones = trj_dones[:last_done_index + 2]

        trj_states_t = torch.FloatTensor(trj_states).to(device)
        trj_actions_t = torch.tensor(trj_actions).to(device)
        policy_t, trj_values_ext_t, trj_values_int_t = net(trj_states_t)
        trj_values_ext_t = trj_values_ext_t.squeeze()
        trj_values_int_t = trj_values_int_t.squeeze()

        # calculate combined rewards advantage
        adv_t, _ = calc_adv_ref((trj_values_ext_t + trj_values_int_t).data.cpu().numpy(),
                                trj_dones, trj_rewards, gamma, gae_lambda)
        adv_t = adv_t.to(device)

        # intrinistic and extrinistic reference values
        _, ref_ext_t = calc_adv_ref(trj_values_ext_t.data.cpu().numpy(),
                                    trj_dones, trj_rewards_ext, gamma, gae_lambda)
        ref_ext_t = ref_ext_t.to(device)

        _, ref_int_t = calc_adv_ref(trj_values_int_t.data.cpu().numpy(),
                                    trj_dones, trj_rewards_int, gamma, gae_lambda)
        ref_int_t = ref_int_t.to(device)

        logpolicy_t = F.log_softmax(policy_t, dim=1)
        old_logprob_t = logpolicy_t.gather(1, trj_actions_t.unsqueeze(-1)).squeeze(-1)
        adv_t = (adv_t - torch.mean(adv_t)) / torch.std(adv_t)
        old_logprob_t = old_logprob_t.detach()

        # make our trajectory splittable on even batch chunks
        trj_len = len(trj_states) - 1
        trj_len -= trj_len % batch_size
        trj_len += 1
        indices = np.arange(0, trj_len-1)
        prep_dt = time.time() - prep_ts

        # generate needed amount of batches
        for _ in range(ppo_epoches):
            np.random.shuffle(indices)
            for batch_indices in np.split(indices, trj_len // batch_size):
                yield (
                    trj_states_t[batch_indices],
                    trj_actions_t[batch_indices],
                    adv_t[batch_indices],
                    ref_ext_t[batch_indices],
                    ref_int_t[batch_indices],
                    old_logprob_t[batch_indices],
                    trj_dt,
                    prep_dt,
                )

        trj_states.clear()
        trj_actions.clear()
        trj_rewards.clear()
        trj_rewards_ext.clear()
        trj_rewards_int.clear()
        trj_dones.clear()
        trj_time = time.time()


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


class AtariBasePPO(nn.Module):
    """
    Dueling net
    """
    def __init__(self, input_shape, n_actions):
        super(AtariBasePPO, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.actor(conv_out), self.critic(conv_out)


class AtariNoisyNetsPPO(nn.Module):
    """
    Dueling net
    """
    def __init__(self, input_shape, n_actions):
        super(AtariNoisyNetsPPO, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            dqn_extra.NoisyLinear(conv_out_size, 256),
            dqn_extra.NoisyLinear(256, n_actions),
        ]

        self.actor = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.actor(conv_out), self.critic(conv_out)

    def sample_noise(self):
        for l in self.noisy_layers:
            l.sample_noise()


class AtariDistill(nn.Module):
    """
    Network to be distilled
    """
    def __init__(self, input_shape):
        super(AtariDistill, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.ff = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.ff(conv_out)


class AtariDistillPPO(nn.Module):
    """
    Dueling net
    """
    def __init__(self, input_shape, n_actions):
        super(AtariDistillPPO, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.critic_ext = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.critic_int = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.actor(conv_out), self.critic_ext(conv_out), self.critic_int(conv_out)


