#!/usr/bin/env bash
set -x
../plot.py -i viter-4x4-reward.csv -o viter-4x4-reward.svg -x Steps -y Reward
../plot.py -i viter-8x8-reward.csv -o viter-8x8-reward.svg -x Steps -y Reward
