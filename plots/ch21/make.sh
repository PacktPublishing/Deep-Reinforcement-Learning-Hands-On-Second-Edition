#!/usr/bin/env bash
set -x
#../plot.py -i dqn/egreedy_reward.csv -o dqn_egreedy_01_reward.svg -x Episodes -y Reward
#../plot.py -i dqn/egreedy_steps.csv -o dqn_egreedy_02_steps.svg -x Episodes -y Steps
#../plot.py -i dqn/egreedy_epsilon.csv -o dqn_egreedy_03_epsilon.svg -x Steps -y Epsilon
#../plot.py -i dqn/egreedy_loss.csv -o dqn_egreedy_04_loss.svg -x Steps -y Loss

#../plot.py -i dqn/nn_reward.csv -o dqn_nn_01_reward.svg -x Episodes -y Reward
#../plot.py -i dqn/nn_steps.csv -o dqn_nn_02_steps.svg -x Episodes -y Steps
#../plot.py -i dqn/nn_loss.csv -o dqn_nn_04_loss.svg -x Steps -y Loss

../plot.py -i dqn/counts_reward.csv -o dqn_counts_01_reward.svg -x Episodes -y Reward
../plot.py -i dqn/counts_steps.csv -o dqn_counts_02_steps.svg -x Episodes -y Steps
../plot.py -i dqn/counts_loss.csv -o dqn_counts_04_loss.svg -x Steps -y Loss
../plot.py -i dqn/counts_test_reward.csv -o dqn_counts_05_test_reward.svg -x Steps -y "Test Reward"
../plot.py -i dqn/counts_test_steps.csv -o dqn_counts_06_test_steps.svg -x Steps -y "Test Steps"
