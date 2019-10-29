#!/usr/bin/env python3
"""
Solver using MCTS and trained model
"""
import time
import argparse
import random
import logging
import datetime
import collections
import csv

from tqdm import tqdm
import seaborn as sns
import matplotlib.pylab as plt
import torch

from libcube import cubes
from libcube import model
from libcube import mcts

log = logging.getLogger("solver")


DataPoint = collections.namedtuple("DataPoint", field_names=(
    'start_dt', 'stop_dt', 'duration', 'depth', 'scramble', 'is_solved', 'solve_steps', 'sol_len_naive', 'sol_len_bfs',
    'depth_max', 'depth_mean'
))


DEFAULT_MAX_SECONDS = 60
PLOT_MAX_DEPTHS = 50
PLOT_TASKS = 20


def generate_task(env, depth):
    res = []
    prev_a = None
    for _ in range(depth):
        a = env.sample_action(prev_action=prev_a)
        res.append(a.value)
        prev_a = a
    return res


def gather_data(cube_env, net, max_seconds, max_steps, max_depth, samples_per_depth, batch_size, device):
    """
    Try to solve lots of cubes to get data
    :param cube_env: CubeEnv
    :param net: model to be used
    :param max_seconds: time limit per cube in seconds
    :param max_steps: limit of steps, if not None it superseeds max_seconds
    :param max_depth: maximum depth of scramble
    :param samples_per_depth: how many cubes of every depth to generate
    :param device: torch.device
    :return: list DataPoint entries
    """
    result = []
    try:
        for depth in range(1, max_depth+1):
            solved_count = 0
            for task_idx in tqdm(range(samples_per_depth)):
                start_dt = datetime.datetime.utcnow()
                task = generate_task(cube_env, depth)
                tree, solution = solve_task(cube_env, task, net, cube_idx=task_idx, max_seconds=max_seconds,
                                            max_steps=max_steps, device=device, quiet=True, batch_size=batch_size)
                is_solved = solution is not None
                stop_dt = datetime.datetime.utcnow()
                duration = (stop_dt - start_dt).total_seconds()
                scramble = " ".join(map(str, task))
                tree_depth_stats = tree.get_depth_stats()
                sol_len_naive, sol_len_bfs = -1, -1
                if is_solved:
                    sol_len_naive = len(solution)
                    sol_len_bfs = len(tree.find_solution())
                data_point = DataPoint(start_dt=start_dt, stop_dt=stop_dt, duration=duration, depth=depth,
                                       scramble=scramble, is_solved=is_solved, solve_steps=len(tree),
                                       sol_len_naive=sol_len_naive, sol_len_bfs=sol_len_bfs,
                                       depth_max=tree_depth_stats['max'], depth_mean=tree_depth_stats['mean'])
                result.append(data_point)
                if is_solved:
                    solved_count += 1
            log.info("Depth %d processed, solved %d/%d (%.2f%%)", depth, solved_count, samples_per_depth,
                     100.0*solved_count/samples_per_depth)
    except KeyboardInterrupt:
        log.info("Interrupt received, got %d data samples, use them", len(result))
    return result


def save_output(data, output_file):
    with open(output_file, "wt", encoding='utf-8') as fd:
        writer = csv.writer(fd)
        writer.writerow(['start_dt', 'stop_dt', 'duration', 'depth', 'scramble', 'is_solved', 'solve_steps',
                         'sol_len_naive', 'sol_len_bfs', 'tree_depth_max', 'tree_depth_mean'])
        for dp in data:
            writer.writerow([
                dp.start_dt.isoformat(),
                dp.stop_dt.isoformat(),
                dp.duration,
                dp.depth,
                dp.scramble,
                int(dp.is_solved),
                dp.solve_steps,
                dp.sol_len_naive,
                dp.sol_len_bfs,
                dp.depth_max,
                dp.depth_mean
            ])


def solve_task(env, task, net, cube_idx=None, max_seconds=DEFAULT_MAX_SECONDS, max_steps=None,
               device=torch.device("cpu"), quiet=False, batch_size=1):
    if not quiet:
        log_prefix = "" if cube_idx is None else "cube %d: " % cube_idx
        log.info("%sGot task %s, solving...", log_prefix, task)
    cube_state = env.scramble(map(env.action_enum, task))
    tree = mcts.MCTS(env, cube_state, net, device=device)
    step_no = 0
    ts = time.time()

    while True:
        if batch_size > 1:
            solution = tree.search_batch(batch_size)
        else:
            solution = tree.search()
        if solution:
            if not quiet:
                log.info("On step %d we found goal state, unroll. Speed %.2f searches/s",
                         step_no, (step_no*batch_size) / (time.time() - ts))
                log.info("Tree depths: %s", tree.get_depth_stats())
                bfs_solution = tree.find_solution()
                log.info("Solutions: naive %d, bfs %d", len(solution), len(bfs_solution))
                log.info("BFS: %s", bfs_solution)
                log.info("Naive: %s", solution)
#                tree.dump_solution(solution)
#                tree.dump_solution(bfs_solution)
#                tree.dump_root()
#                log.info("Tree: %s", tree)
            return tree, solution
        step_no += 1
        if max_steps is not None:
            if step_no > max_steps:
                if not quiet:
                    log.info("Maximum amount of steps has reached, cube wasn't solved. "
                             "Did %d searches, speed %.2f searches/s",
                             step_no, (step_no*batch_size) / (time.time() - ts))
                    log.info("Tree depths: %s", tree.get_depth_stats())
                return tree, None
        elif time.time() - ts > max_seconds:
            if not quiet:
                log.info("Time is up, cube wasn't solved. Did %d searches, speed %.2f searches/s..",
                         step_no, (step_no*batch_size) / (time.time() - ts))
                log.info("Tree depths: %s", tree.get_depth_stats())
            return tree, None


def produce_plots(data, prefix, max_seconds, max_steps):
    data_solved = [(dp.depth, int(dp.is_solved)) for dp in data]
    data_steps = [(dp.depth, dp.solve_steps) for dp in data if dp.is_solved]

    if max_steps is not None:
        suffix = "(steps limit %d)" % max_steps
    else:
        suffix = "(time limit %d secs)" % max_seconds

    sns.set()
    d, v = zip(*data_solved)
    plot = sns.lineplot(d, v)
    plot.set_title("Solve ratio per depth %s" % suffix)
    plot.get_figure().savefig(prefix + "-solve_vs_depth.png")

    plt.clf()
    d, v = zip(*data_steps)
    plot = sns.lineplot(d, v)
    plot.set_title("Steps to solve per depth %s" % suffix)
    plot.get_figure().savefig(prefix + "-steps_vs_depth.png")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", required=True, help="Type of env to train, supported types=%s" % cubes.names())
    parser.add_argument("-m", "--model", required=True, help="Model file to load, has to match env type")
    parser.add_argument("--max-time", type=int, default=DEFAULT_MAX_SECONDS,
                        help="Limit in seconds for each task, default=%s" % DEFAULT_MAX_SECONDS)
    parser.add_argument("--max-steps", type=int, help="Limit amount of MCTS searches to be done. "
                                                      "If specified, superseeds --max-time")
    parser.add_argument("--max-depth", type=int, default=PLOT_MAX_DEPTHS,
                        help="Maximum depth for plots and data, default=%s" % PLOT_MAX_DEPTHS)
    parser.add_argument("--samples", type=int, default=PLOT_TASKS,
                        help="Count of tests of each depth, default=%s" % PLOT_TASKS)
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size to use during the search, default=1")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use, if zero, no seed used. default=42")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Text file with permutations to read cubes to solve, "
                                             "possibly produced by gen_cubes.py")
    group.add_argument("-p", "--perm", help="Permutation in form of actions list separated by comma")
    group.add_argument("-r", "--random", metavar="DEPTH", type=int, help="Generate random scramble of given depth")
    group.add_argument("--plot", metavar="PREFIX", help="Produce plots of model solve accuracy")
    group.add_argument("-o", "--output", help="Write test result into csv file with given name")
    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")

    cube_env = cubes.get(args.env)
    log.info("Using environment %s", cube_env)
    assert isinstance(cube_env, cubes.CubeEnv)              # just to help pycharm understand type

    net = model.Net(cube_env.encoded_shape, len(cube_env.action_enum)).to(device)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    net.eval()
    log.info("Network loaded from %s", args.model)

    if args.random is not None:
        task = generate_task(cube_env, args.random)
        solve_task(cube_env, task, net, max_seconds=args.max_time, max_steps=args.max_steps, device=device,
                   batch_size=args.batch)
    elif args.perm is not None:
        task = list(map(int, args.perm.split(',')))
        solve_task(cube_env, task, net, max_seconds=args.max_time, max_steps=args.max_steps, device=device,
                   batch_size=args.batch)
    elif args.input is not None:
        log.info("Processing scrambles from %s", args.input)
        count = 0
        solved = 0
        with open(args.input, 'rt', encoding='utf-8') as fd:
            for idx, l in enumerate(fd):
                task = list(map(int, l.strip().split(',')))
                _, solution  = solve_task(cube_env, task, net, cube_idx=idx, max_seconds=args.max_time,
                                          max_steps=args.max_steps, device=device, batch_size=args.batch)
                if solution is not None:
                    solved += 1
                count += 1
        log.info("Solved %d out of %d cubes, which is %.2f%% success ratio", solved, count, 100*solved / count)
    elif args.plot is not None:
        log.info("Produce plots with prefix %s", args.plot)
        data = gather_data(cube_env, net, args.max_time, args.max_steps, args.max_depth, args.samples,
                           args.batch, device)
        produce_plots(data, args.plot, args.max_time)
    elif args.output is not None:
        data = gather_data(cube_env, net, args.max_time, args.max_steps, args.max_depth, args.samples,
                           args.batch, device)
        save_output(data, args.output)
        pass
