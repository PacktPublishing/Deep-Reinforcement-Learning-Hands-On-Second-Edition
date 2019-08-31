#!/usr/bin/env python3
import gym
import ptan
import random
import pathlib
import argparse
import itertools
import numpy as np
from typing import List
import warnings
from textworld.gym import register_games
from textworld.envs.wrappers.filter import EnvInfos

from lib import preproc, model, common

import torch
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine


GAMMA = 0.9
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LM_PRETRAIN_START_ANNEAL = 10000
LM_PRETRAIN_STEPS = 10000
LM_PRETRAIN_FINAL = 0.4
POLICY_BETA = 0.1

# have to be less or equal to env.action_space.max_length
LM_MAX_TOKENS = 2


EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "admissible_commands": True,
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


def batch_generator(exp_source: ptan.experience.ExperienceSourceFirstLast,
                    batch_size: int):
    batch = []
    for exp in exp_source:
        batch.append(exp)
        if len(batch) == batch_size:
            yield batch
            batch.clear()


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default="simple",
                        help="Game prefix to be used during training, default=simple")
    parser.add_argument("--params", choices=list(common.PARAMS.keys()),
                        help="Training params, could be one of %s" % (list(common.PARAMS.keys())))
    parser.add_argument("-s", "--suffices", type=int, default=1,
                        help="Count of game indices to use during training, default=1")
    parser.add_argument("-v", "--validation", default='-val',
                        help="Suffix for game used for validation, default=-val")
    parser.add_argument("--cuda", default=False, action='store_true',
                        help="Use cuda for training")
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    params = common.PARAMS[args.params]

    game_files = ["games/%s%s.ulx" % (args.game, s) for s in range(1, args.suffices+1)]
    if not all(map(lambda p: pathlib.Path(p).exists(), game_files)):
        raise RuntimeError(f"Some game files from {game_files} not found! Probably you need to run make_games.sh")
    env_id = register_games(game_files, request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Registered env %s for game files %s" % (env_id, game_files))
    val_game_file = "games/%s%s.ulx" % (args.game, args.validation)
    val_env_id = register_games([val_game_file], request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Game %s, with file %s will be used for validation" % (val_env_id, val_game_file))

    env = gym.make(env_id)
    env = preproc.TextWorldPreproc(env, use_admissible_commands=False,
                                   keep_admissible_commands=True,
                                   reward_wrong_last_command=-0.1)
    val_env = gym.make(val_env_id)
    val_env = preproc.TextWorldPreproc(val_env)

    prep = preproc.Preprocessor(
        dict_size=env.observation_space.vocab_size,
        emb_size=params.embeddings, num_sequences=env.num_fields,
        enc_output_size=params.encoder_size).to(device)

    cmd = model.CommandModel(prep.obs_enc_size, env.observation_space.vocab_size, prep.emb,
                             max_tokens=LM_MAX_TOKENS,
                             start_token=env.action_space.BOS_id,
                             sep_token=env.action_space.EOS_id).to(device)
    net = model.A2CModel(obs_size=env.num_fields * params.encoder_size)
    net = net.to(device)
    agent = model.CmdAgent(env, cmd, prep, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)

    optimizer = optim.RMSprop(itertools.chain(prep.parameters(),
                                              cmd.parameters(),
                                              net.parameters()),
                              lr=LEARNING_RATE, eps=1e-5)
    lm_pretrain_prob = 1.0

    def process_batch(engine, batch):
        global lm_pretrain_prob
        optimizer.zero_grad()
        obs_t, vals_ref_t = unpack_batch(batch, prep, net)
        vals_t = net(obs_t).squeeze(-1)
        value_loss_t = F.mse_loss(vals_t, vals_ref_t)

        res_dict = {"loss_value": value_loss_t.item()}

        # lm pretraining: done in random instead of A2C policy loss
        if random.random() < lm_pretrain_prob:
            commands = []

            for s in batch:
                cmds = []
                for c in s.state['admissible_commands']:
                    t = env.action_space.tokenize(c)
                    if len(t)-2 <= LM_MAX_TOKENS:
                        cmds.append(t)
                commands.append(cmds)

            pretrain_loss_t = model.pretrain_policy_loss(cmd, commands, obs_t)
            res_dict['loss_pretrain'] = pretrain_loss_t.item()
            loss_t = value_loss_t + pretrain_loss_t
        else:
            adv_t = (vals_ref_t - vals_t).detach()
            _, logits_batch = cmd.commands(obs_t)
            policy_loss_t = None
            for logits, adv_val_t in zip(logits_batch, adv_t):
                if not logits[0]:
                    continue
                logits_t = torch.stack(logits[0])
                log_prob_t = adv_val_t * F.log_softmax(logits_t, dim=1)
                loss_p_t = -log_prob_t.mean()
                if policy_loss_t is None:
                    policy_loss_t = loss_p_t
                else:
                    policy_loss_t += loss_p_t
            policy_loss_t = policy_loss_t * POLICY_BETA
            res_dict['loss_policy'] = policy_loss_t.item()
            loss_t = value_loss_t + policy_loss_t
        res_dict['loss'] = loss_t.item()
        loss_t.backward()
        optimizer.step()

        if not hasattr(engine.state, "episode"):
            episode = 0
        else:
            episode = engine.state.episode
        if episode > LM_PRETRAIN_START_ANNEAL:
            lm_p = 1 - (episode - LM_PRETRAIN_START_ANNEAL) / LM_PRETRAIN_STEPS
            lm_pretrain_prob = max(lm_p, LM_PRETRAIN_FINAL)
        res_dict['pretrain_prob'] = lm_pretrain_prob
        return res_dict

    engine = Engine(process_batch)
    run_name = f"lm-{args.params}_{args.run}"
    save_path = pathlib.Path("saves") / run_name
    save_path.mkdir(parents=True, exist_ok=True)

    common.setup_ignite(engine, exp_source, run_name,
                        extra_metrics=('val_reward', 'val_steps'))

    @engine.on(ptan.ignite.EpisodeEvents.BEST_REWARD_REACHED)
    def best_reward_updated(trainer: Engine):
        reward = trainer.state.metrics['avg_reward']
        if reward > 0:
            save_prep_name = save_path / ("best_train_%.3f_p.dat" % reward)
            save_net_name = save_path / ("best_train_%.3f_n.dat" % reward)
            torch.save(prep.state_dict(), save_prep_name)
            torch.save(net.state_dict(), save_net_name)
            print("%d: best avg training reward: %.3f, saved" % (
                trainer.state.iteration, reward))


    engine.run(batch_generator(exp_source, BATCH_SIZE))
