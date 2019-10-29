#!/usr/bin/env python3
"""
Tool to generate test set for solver
"""
import argparse
import random

from libcube import cubes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-n", "--number", type=int, default=10, help="Amount of scramble rounds, default=10")
    parser.add_argument("-d", "--depth", type=int, default=100, help="Scramble depth, default=10")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use, if zero, no seed used. default=42")
    parser.add_argument("-o", "--output", required=True, help="Output file to produce")
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    cube_env = cubes.get(args.env)
    assert isinstance(cube_env, cubes.CubeEnv)

    with open(args.output, "w+t", encoding="utf-8") as fd_out:
        for _ in range(args.number):
            s = cube_env.initial_state
            path = []
            prev_a = None
            for _ in range(args.depth):
                a = cube_env.sample_action(prev_action=prev_a)
                path.append(a.value)
                s = cube_env.transform(s, a)
                prev_a = a
            fd_out.write(",".join(map(str, path)) + "\n")
