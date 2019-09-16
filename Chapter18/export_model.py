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
    fd.write("""from . import nn

""")


def write_weights(fd, weights):
    fd.write("WEIGHTS = [\n")
    for w, b in weights:
        fd.write("(%s, [%s]),\n" % (
            w.tolist(),  b.tolist()
        ))
    fd.write("]\n")


def write_forward_pass(fd, forward_pass):
    fd.write("""

def forward(x):
""")

    for f in forward_pass:
        fd.write("    %s\n" % f)

    fd.write("    return x\n")


def write_suffix(fd, input_dim):
    fd.write(f"""

def test():
    x = [[0.0]] * {input_dim}
    y = forward(x)
    print(y)
    
    
def show():
    for idx, (w, b) in enumerate(WEIGHTS):
        print("Layer %d:" % (idx+1))
        print("W: (%d, %d), B: (%d, %d)" % (len(w), len(w[0]), len(b), len(b[0])))

""")
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
            forward_pass.append(f"x = nn.linear(x, WEIGHTS[{len(weights_data)}])")
            weights_data.append(w)
        elif isinstance(m, nn.ReLU):
            forward_pass.append("x = nn.relu(x)")
        elif isinstance(m, nn.Tanh):
            forward_pass.append("x = nn.tanh(x)")
        else:
            print('Unsupported layer! %s' % m)

    with output_path.open("wt", encoding='utf-8') as fd_out:
        write_prefix(fd_out)
        write_weights(fd_out, weights_data)
        write_forward_pass(fd_out, forward_pass)
        write_suffix(fd_out, args.input_dim)
