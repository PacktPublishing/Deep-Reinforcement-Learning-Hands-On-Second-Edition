#!/usr/bin/env python3
import gym
import ptan
import time
import pathlib
import argparse
import itertools
import datetime
import numpy as np

from textworld.gym import register_games
from textworld.envs.wrappers.filter import EnvInfos

from lib import preproc, model

import torch
import torch.optim as optim


EXTRA_GAME_INFO = {
    "inventory": True,
    "description": True,
    "intermediate_reward": True,
    "admissible_commands": True,
    "policy_commands": True,
}

# length of encoder's output
ENC_SIZE = 20
# length of embeddings to be learned
EMB_SIZE = 20

REPLAY_SIZE = 10000
REPLAY_INITIAL = 100
GAMMA = 0.9
LEARNING_RATE = 5e-5
SYNC_NETS = 100
BATCH_SIZE = 64

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.2
STEPS_EPSILON = 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default="simple",
                        help="Game prefix to be used during training, default=simple")
    parser.add_argument("-s", "--suffices", action='append',
                        help="Game suffices to be appended to game prefix. Might be given "
                             "several times to train on multiple games, default=1")
    parser.add_argument("-v", "--validation", default='-val',
                        help="Suffix for game used for validation, default=-val")
    parser.add_argument("--cuda", default=False, action='store_true',
                        help="Use cuda for training")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    suffices = ['1'] if args.suffices is None else args.suffices

    game_files = ["games/%s%s.ulx" % (args.game, s) for s in suffices]
    if not all(map(lambda p: pathlib.Path(p).exists(), game_files)):
        raise RuntimeError(f"Some game files from {game_files} not found! Probably you need to run make_games.sh")
    env_id = register_games(game_files, request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Registered env %s for game files %s" % (env_id, game_files))
    val_game_file = "games/%s%s.ulx" % (args.game, args.validation)
    val_env_id = register_games([val_game_file], request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Game %s, with file %s will be used for validation" % (val_env_id, val_game_file))

    env = gym.make(env_id)
    env = preproc.TextWorldPreproc(env)

    prep = preproc.Preprocessor(
        dict_size=env.observation_space.vocab_size,
        emb_size=EMB_SIZE, num_sequences=env.num_fields,
        enc_output_size=ENC_SIZE).to(device)
    tgt_prep = ptan.agent.TargetNet(prep)

    net = model.DQNModel(obs_size=env.num_fields * ENC_SIZE,
                         cmd_size=ENC_SIZE)
    net = net.to(device)
    tgt_net = ptan.agent.TargetNet(net)

    agent = model.DQNAgent(net, prep, epsilon=INITIAL_EPSILON, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=GAMMA, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, REPLAY_SIZE)

    optimizer = optim.RMSprop(itertools.chain(net.parameters(), prep.parameters()),
                              lr=LEARNING_RATE, eps=1e-5)

    steps_done = 0
    episodes_done = 0
    losses = []
    rewards = []
    prev_steps = 0
    start_ts = prev_ts = time.time()

    for _ in range(10000):
        steps_done += 1
        buffer.populate(1)
        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            speed = (steps_done - prev_steps) / (time.time() - prev_ts)
            prev_steps = steps_done
            prev_ts = time.time()
            for rw, steps in rewards_steps:
                episodes_done += 1
                print("%d: Done %d episodes: reward = %.2f, steps = %d, speed = %.2f steps/sec, epsilon = %.2f" % (
                    steps_done, episodes_done, rw, steps, speed, agent.epsilon))
                rewards.append(rw)
            if rewards and np.mean(rewards[-10:]) == 6.0:
                print(
                    "Environment has been solved in %s, congrats!" % datetime.timedelta(seconds=time.time() - start_ts))
                break
        if len(buffer) < REPLAY_INITIAL:
            continue

        batch = buffer.sample(BATCH_SIZE)
        optimizer.zero_grad()
        loss_t = model.calc_loss_dqn(batch, prep, tgt_prep.target_model,
                                     net, tgt_net.target_model, GAMMA, device=device)
        loss_t.backward()
        optimizer.step()
        losses.append(loss_t.item())

        if steps_done % SYNC_NETS == 0:
            tgt_prep.sync()
            tgt_net.sync()
            print("%d: sync nets" % steps_done)

        agent.epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - steps_done / STEPS_EPSILON)
