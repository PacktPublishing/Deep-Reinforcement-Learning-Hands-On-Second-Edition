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

#../plot.py -i ppo/dist_reward.csv -o ppo_dist_01_reward.svg -x Episodes -y Reward
#../plot.py -i ppo/dist_steps.csv -o ppo_dist_02_steps.svg -x Episodes -y Steps
#../plot.py -i ppo/dist_loss.csv -o ppo_dist_03_loss.svg -x Steps -y Loss
#../plot.py -i ppo/dist_loss_dist.csv -o ppo_dist_04_loss_dist.svg -x Steps -y "Distillation Loss"
#../plot.py -i ppo/dist_test_reward.csv -o ppo_dist_05_test_reward.svg -x Steps -y "Test Reward"

#../plot.py -i atari/dqn_egreedy_reward.csv -o atari_dqn_egreedy_01_reward.svg -x Episodes -y Reward
#../plot.py -i atari/dqn_egreedy_steps.csv -o atari_dqn_egreedy_02_steps.svg -x Episodes -y Steps

#../plot.py -i atari/ppo_reward.csv -o atari_ppo_01_reward.svg -x Episodes -y Reward
#../plot.py -i atari/ppo_steps.csv -o atari_ppo_02_steps.svg -x Episodes -y Steps
#../plot.py -i atari/ppo_loss.csv -o atari_ppo_03_loss.svg -x Steps -y "Total loss"
#../plot.py -i atari/ppo_loss_policy.csv -o atari_ppo_04_loss_policy.svg -x Steps -y "Policy loss"
#../plot.py -i atari/ppo_loss_value.csv -o atari_ppo_05_loss_value.svg -x Steps -y "Value loss"
#../plot.py -i atari/ppo_loss_entropy.csv -o atari_ppo_06_loss_entropy.svg -x Steps -y "Entropy loss"
#
#../plot.py -i atari/ppo_distill_reward.csv -o atari_ppo_distill_01_reward.svg -x Episodes -y Reward
#../plot.py -i atari/ppo_distill_steps.csv -o atari_ppo_distill_02_steps.svg -x Episodes -y Steps
#../plot.py -i atari/ppo_distill_test_reward.csv -o atari_ppo_distill_03_test_reward.svg -x Steps -y "Test reward"
#../plot.py -i atari/ppo_distill_ref_ext.csv -o atari_ppo_distill_04_ref_ext.svg -x Steps -y "Extrinsic reference"
#../plot.py -i atari/ppo_distill_ref_int.csv -o atari_ppo_distill_05_ref_int.svg -x Steps -y "Intrinsic reference"

../plot.py -i atari/ppo_nn_reward.csv -o atari_ppo_nn_01_reward.svg -x Episodes -y Reward
../plot.py -i atari/ppo_nn_steps.csv -o atari_ppo_nn_02_steps.svg -x Episodes -y Steps
../plot.py -i atari/ppo_nn_loss_entropy.csv -o atari_ppo_nn_03_loss_entropy.svg -x Steps -y "Entropy Loss"
