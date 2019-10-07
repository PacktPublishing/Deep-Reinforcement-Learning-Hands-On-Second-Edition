#!/usr/bin/env bash
set -x
../plot.py -i dqn/egreedy_reward.csv -o dqn_egreedy_01_reward.svg -x Episodes -y Reward
../plot.py -i dqn/egreedy_steps.csv -o dqn_egreedy_02_steps.svg -x Episodes -y Steps
../plot.py -i dqn/egreedy_epsilon.csv -o dqn_egreedy_03_epsilon.svg -x Steps -y Epsilon
../plot.py -i dqn/egreedy_loss.csv -o dqn_egreedy_04_loss.svg -x Steps -y Loss
