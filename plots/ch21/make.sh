#!/usr/bin/env bash
set -x
#../plot.py -i dqn/egreedy_reward.csv -o dqn_egreedy_01_reward.svg -x Episodes -y Reward
#../plot.py -i dqn/egreedy_steps.csv -o dqn_egreedy_02_steps.svg -x Episodes -y Steps
#../plot.py -i dqn/egreedy_epsilon.csv -o dqn_egreedy_03_epsilon.svg -x Steps -y Epsilon
#../plot.py -i dqn/egreedy_loss.csv -o dqn_egreedy_04_loss.svg -x Steps -y Loss

#../plot.py -i dqn/nn_reward.csv -o dqn_nn_01_reward.svg -x Episodes -y Reward
#../plot.py -i dqn/nn_steps.csv -o dqn_nn_02_steps.svg -x Episodes -y Steps
#../plot.py -i dqn/nn_loss.csv -o dqn_nn_04_loss.svg -x Steps -y Loss

#../plot.py -i dqn/counts_reward.csv -o dqn_counts_01_reward.svg -x Episodes -y Reward
#../plot.py -i dqn/counts_steps.csv -o dqn_counts_02_steps.svg -x Episodes -y Steps
#../plot.py -i dqn/counts_loss.csv -o dqn_counts_04_loss.svg -x Steps -y Loss
#../plot.py -i dqn/counts_test_reward.csv -o dqn_counts_05_test_reward.svg -x Steps -y "Test Reward"
#../plot.py -i dqn/counts_test_steps.csv -o dqn_counts_06_test_steps.svg -x Steps -y "Test Steps"

#../plot.py -i ppo/basic_reward.csv -o ppo_basic_01_reward.svg -x Episodes -y Reward
#../plot.py -i ppo/basic_steps.csv -o ppo_basic_02_steps.svg -x Episodes -y Steps
#../plot.py -i ppo/basic_loss.csv -o ppo_basic_03_loss.svg -x Steps -y Loss
#../plot.py -i ppo/basic_loss_entropy.csv -o ppo_basic_04_loss_ent.svg -x Steps -y Loss
#../plot.py -i ppo/basic_test_reward.csv -o ppo_basic_05_test_reward.svg -x Steps -y "Test Reward"

#../plot.py -i ppo/nn_reward.csv -o ppo_nn_01_reward.svg -x Episodes -y Reward
#../plot.py -i ppo/nn_steps.csv -o ppo_nn_02_steps.svg -x Episodes -y Steps
#../plot.py -i ppo/nn_loss.csv -o ppo_nn_03_loss.svg -x Steps -y Loss
#../plot.py -i ppo/nn_loss_entropy.csv -o ppo_nn_04_loss_ent.svg -x Steps -y "Entropy Loss"
#../plot.py -i ppo/nn_test_reward.csv -o ppo_nn_05_test_reward.svg -x Steps -y "Test Reward"

#../plot.py -i ppo/counts_reward.csv -o ppo_counts_01_reward.svg -x Episodes -y Reward
#../plot.py -i ppo/counts_steps.csv -o ppo_counts_02_steps.svg -x Episodes -y Steps
#../plot.py -i ppo/counts_loss.csv -o ppo_counts_03_loss.svg -x Steps -y Loss
#../plot.py -i ppo/counts_loss_entropy.csv -o ppo_counts_04_loss_ent.svg -x Steps -y "Entropy Loss"
#../plot.py -i ppo/counts_test_reward.csv -o ppo_counts_05_test_reward.svg -x Steps -y "Test Reward"

../plot.py -i ppo/dist_reward.csv -o ppo_dist_01_reward.svg -x Episodes -y Reward
../plot.py -i ppo/dist_steps.csv -o ppo_dist_02_steps.svg -x Episodes -y Steps
../plot.py -i ppo/dist_loss.csv -o ppo_dist_03_loss.svg -x Steps -y Loss
../plot.py -i ppo/dist_loss_dist.csv -o ppo_dist_04_loss_dist.svg -x Steps -y "Distillation Loss"
../plot.py -i ppo/dist_test_reward.csv -o ppo_dist_05_test_reward.svg -x Steps -y "Test Reward"
