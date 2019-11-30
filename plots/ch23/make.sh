#!/usr/bin/env bash
set -x

../plot.py -i win-ratio.csv -o 03-win-ratio.svg -x Steps -y "Win ratio for trained net"
../plot.py -i loss-policy.csv -o 04-01-loss-policy.svg -x Steps -y "Policy loss"
../plot.py -i loss-value.csv -o 04-02-loss-value.svg -x Steps -y "Value loss"
