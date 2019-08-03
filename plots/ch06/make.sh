#!/usr/bin/env bash
set -x
../plot.py -i 01_reward.csv -o 01_reward.svg -y Reward
../plot.py -i 02_avg_reward.csv -o 02_avg_reward.svg -y "Mean reward"
