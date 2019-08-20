#!/usr/bin/env python3
import gym
import ptan
import pathlib
import argparse
import itertools
import numpy as np
import warnings

from textworld.gym import register_games
from textworld.envs.wrappers.filter import EnvInfos

from lib import preproc, model, common

import torch
import torch.optim as optim
from ignite.engine import Engine


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
REPLAY_INITIAL = 1000
GAMMA = 0.9
LEARNING_RATE = 5e-5
SYNC_NETS = 100
BATCH_SIZE = 64

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.2
STEPS_EPSILON = 1000


if __name__ == "__main__":
    warnings.simplefilter("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--game", default="simple",
                        help="Game prefix to be used during training, default=simple")
    # parser.add_argument("-s", "--suffices", action='append',
    #                     help="Game suffices to be appended to game prefix. Might be given "
    #                          "several times to train on multiple games, default=1")
    parser.add_argument("-s", "--suffices", type=int, default=1, help="Count of indices to use in games")
    parser.add_argument("-v", "--validation", default='-val',
                        help="Suffix for game used for validation, default=-val")
    parser.add_argument("--cuda", default=False, action='store_true',
                        help="Use cuda for training")
    parser.add_argument("-r", "--run", required=True, help="Run name")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    game_files = ["games/%s%s.ulx" % (args.game, s) for s in range(1, args.siffices+1)]
    if not all(map(lambda p: pathlib.Path(p).exists(), game_files)):
        raise RuntimeError(f"Some game files from {game_files} not found! Probably you need to run make_games.sh")
    env_id = register_games(game_files, request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Registered env %s for game files %s" % (env_id, game_files))
    val_game_file = "games/%s%s.ulx" % (args.game, args.validation)
    val_env_id = register_games([val_game_file], request_infos=EnvInfos(**EXTRA_GAME_INFO), name=args.game)
    print("Game %s, with file %s will be used for validation" % (val_env_id, val_game_file))

    env = gym.make(env_id)
    env = preproc.TextWorldPreproc(env)

    val_env = gym.make(val_env_id)
    val_env = preproc.TextWorldPreproc(val_env)

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

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_t = model.calc_loss_dqn(batch, prep, tgt_prep.target_model,
                                     net, tgt_net.target_model, GAMMA, device=device)
        loss_t.backward()
        optimizer.step()
        eps = INITIAL_EPSILON - engine.state.iteration / STEPS_EPSILON
        agent.epsilon = max(FINAL_EPSILON, eps)
        if engine.state.iteration % SYNC_NETS == 0:
            tgt_net.sync()
            tgt_prep.sync()
        return {
            "loss": loss_t.item(),
            "epsilon": agent.epsilon,
        }

    engine = Engine(process_batch)
    common.setup_ignite(engine, exp_source, f"basic_{args.run}",
                        extra_metrics=('val_reward', 'val_steps'))

    @engine.on(ptan.ignite.PeriodEvents.ITERS_100_COMPLETED)
    def validate(engine):
        reward = 0.0
        steps = 0

        obs = val_env.reset()

        while True:
            obs_t = prep.encode_sequences([obs['obs']]).to(device)
            cmd_t = prep.encode_commands(obs['admissible_commands']).to(device)
            q_vals = net.q_values(obs_t, cmd_t)
            act = np.argmax(q_vals)

            obs, r, is_done, _ = val_env.step(act)
            steps += 1
            reward += r
            if is_done:
                break
        engine.state.metrics['val_reward'] = reward
        engine.state.metrics['val_steps'] = steps
        print("Validation got %.3f reward in %d steps" % (reward, steps))

    engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))
