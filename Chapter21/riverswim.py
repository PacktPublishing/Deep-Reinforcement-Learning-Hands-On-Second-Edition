#!/usr/bin/env python3
import random
import argparse
import collections

SEED = 2


def get_action(state: int, total_states: int) -> int:
    """
    Return action from the given state. Actions are selected randomly
    :param state: state we're currently in
    :return: 0 means left, 1 is right
    """
    if state == 1:
        return 1
    if state == total_states:
        return 0
    return random.choice([0, 1])


def do_action(state: int, action: int) -> int:
    """
    Simulate the action from the given state
    """
    # left action always succeeds and brings us to the left
    if action == 0:
        return state-1

    if state == 1:
        return random.choices([1, 2], weights=[0.4, 0.6])[0]
    # the rest of states are the same
    delta = random.choices([-1, 0, 1], weights=[0.05, 0.6, 0.35])[0]
    return state + delta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--steps", type=int, default=100, help="Amount of steps to simulate, default=100")
    parser.add_argument("--episode-length", type=int, default=10, help="Limit of one episode, default=10")
    parser.add_argument("--seed", type=int, default=SEED, help="Seed to use, default=%d" % SEED)
    parser.add_argument("--env-len", type=int, default=6, help="Amount of states in the environment, default=6")
    args = parser.parse_args()
    random.seed(args.seed)

    states_count = collections.Counter()
    state = 1
    episode_step = 0

    for _ in range(args.steps):
        action = get_action(state, args.env_len)
        state = do_action(state, action)
        states_count[state] += 1
        episode_step += 1
        if episode_step == args.episode_length:
            state = 1
            episode_step = 0

    for state in range(1, args.env_len+1):
        print("%d:\t%d" % (state, states_count[state]))
