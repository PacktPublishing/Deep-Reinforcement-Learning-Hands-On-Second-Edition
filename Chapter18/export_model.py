#!/usr/bin/env python3
"""
Utility exports pytorch model into python module to be used with ulinalg (https://github.com/jalawson/ulinalg)
"""
import pathlib
import argparse
import torch
import torch.nn as nn
from lib import ddpg

DEFAULT_INPUT_DIM = 28
ACTIONS_DIM = 4


def write_prefix(fd):
    fd.write("""
import math as m
from lib    hw.ulinalg import ulinalg, umatrix


def relu(x):
    return x.apply(lambda v: 0.0 if v < 0.0 else v)


def tanh(x):
    return x.apply(m.tanh)


def linear(x, w_pair):
    w, b = w_pair
    y = ulinalg.dot(w, x) + b
    return y


""")


def write_weights(fd, weights):
    fd.write("WEIGHTS = [\n")

    for w, b in weights:
        fd.write("(umatrix.matrix(%s), umatrix.matrix([%s]).T),\n" % (
            w.tolist(),  b.tolist()
        ))

    fd.write("]\n")


def write_forward_pass(fd, forward_pass):
    fd.write("""

def forward(vals):
    x = umatrix.matrix(vals, cstride=1, rstride=len(vals), dtype=float).T
""")

    for f in forward_pass:
        fd.write("    %s\n" % f)

    fd.write("    return x\n")


def write_suffix(fd, input_dim):
    fd.write("""

if __name__ == "__main__":
    x = [0.0] * %d
    y = forward(x)
    print(y.shape)
    print(y)
""" % input_dim)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Model data file to be exported")
    parser.add_argument("-o", "--output", required=True, help="Name of output python file to be created")
    parser.add_argument("--input-dim", type=int, default=DEFAULT_INPUT_DIM,
                        help="Dimension of the input, default=%s" % DEFAULT_INPUT_DIM)
    args = parser.parse_args()
    output_path = pathlib.Path(args.output)

    act_net = ddpg.DDPGActor(args.input_dim, ACTIONS_DIM)
    act_net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

    weights_data = []
    forward_pass = []

    for m in act_net.net:
        if isinstance(m, nn.Linear):
            w = [m.weight.detach().numpy(), m.bias.detach().numpy()]
            forward_pass.append(f"x = linear(x, WEIGHTS[{len(weights_data)}])")
            weights_data.append(w)
        elif isinstance(m, nn.ReLU):
            forward_pass.append("x = relu(x)")
        elif isinstance(m, nn.Tanh):
            forward_pass.append("x = tanh(x)")
        else:
            print('Unknown module! %s' % m)

    with output_path.open("wt", encoding='utf-8') as fd_out:
        write_prefix(fd_out)
        write_weights(fd_out, weights_data)
        write_forward_pass(fd_out, forward_pass)
        write_suffix(fd_out, args.input_dim)
