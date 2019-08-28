import gym
import ptan
import numpy as np
from typing import List
from textworld.gym import register_games
from textworld.envs.wrappers.filter import EnvInfos

from lib import preproc, model, common

import torch

GAMMA = 0.9
BATCH_SIZE = 16


EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "last_command": True,
}


def unpack_batch(batch: List[ptan.experience.ExperienceFirstLast], prep: preproc.Preprocessor, net: model.A2CModel):
    states = []
    rewards = []
    not_done_idx = []
    next_states = []

    for idx, exp in enumerate(batch):
        states.append(exp.state['obs'])
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            next_states.append(exp.last_state['obs'])
    obs_t = prep.encode_sequences(states)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        obs_next_t = prep.encode_sequences(next_states)
        next_vals_t = net(obs_next_t)
        next_vals_np = next_vals_t.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA * next_vals_np

    ref_vals_t = torch.FloatTensor(rewards_np).to(obs_t.device)
    return obs_t, ref_vals_t




def run(device = "cpu"):
    env_id = register_games(["games/simple1.ulx"], request_infos=EnvInfos(**EXTRA_GAME_INFO))
    env = gym.make(env_id)
    env = preproc.TextWorldPreproc(env, use_admissible_commands=False, reward_wrong_last_command=-1)
    params = common.PARAMS['small']

    prep = preproc.Preprocessor(
        dict_size=env.observation_space.vocab_size,
        emb_size=params.embeddings, num_sequences=env.num_fields,
        enc_output_size=params.encoder_size)

    cmd = model.CommandModel(prep.obs_enc_size, env.observation_space.vocab_size, prep.emb,
                             max_tokens=env.action_space.max_length,
                             max_commands=5, start_token=env.action_space.BOS_id,
                             sep_token=env.action_space.SEP_id)
    net = model.A2CModel(obs_size=env.num_fields * params.encoder_size)
    agent = model.CmdAgent(env, cmd, prep, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)

    batch = []
    for exp in exp_source:
        batch.append(exp)
        if len(batch) < BATCH_SIZE:
            continue

        obs_t, vals_ref_t = unpack_batch(batch, prep, net)
        break

    s = env.reset()
    obs_t = prep.encode_sequences([s['obs']])
    print(obs_t)
    tokens, logits = cmd(obs_t)

    return env, prep, cmd


if __name__ == "__main__":
    run()
