#!/usr/bin/env bash
set -x
../plot.py -i cp-reward_nobl.csv -o cp-reward-nobl.svg -x Steps -y Reward
../plot.py -i cp-reward_bl.csv -o cp-reward-bl.svg -x Steps -y Reward
../plot.py -i cp-reward100_nobl.csv -i cp-reward100_bl.csv -l "No baseline" -l "Baseline" -o cp-reward100.svg -x Steps -y "Smoothed reward"
../plot.py -i cp-l2-nobl.csv -i cp-l2-bl.csv -l "No baseline" -l "Baseline" -o cp-l2.svg -x Steps -y "Grads L2" --ylog
../plot.py -i cp-max-nobl.csv -i cp-max-bl.csv -l "No baseline" -l "Baseline" -o cp-max.svg -x Steps -y "Grads Max" --ylog
../plot.py -i cp-var-nobl.csv -i cp-var-bl.csv -l "No baseline" -l "Baseline" -o cp-var.svg -x Steps -y "Grads Var" --ylog

#../plot.py -i dqn_episodes.csv -i rf_episodes.csv -o 01_dqn_rf_episodes.svg -x Steps -y Episodes
#../plot.py -i dqn_rewards100.csv -i rf_rewards100.csv -o 02_dqn_rf_rewards100.svg -x Steps

#../plot.py -i pg_reward100.csv -o 03_pg_reward.svg -x Steps -y Reward
#../plot.py -i pg_baseline.csv -o 04_pg_baseline.svg -x Steps -y Baseline
#../plot.py -i pg_batch_scales.csv -o 05_pg_batch_scales.svg -x Steps -y "Batch scale"
#../plot.py -i pg_entropy.csv -o 06_pg_entropy.svg -x Steps -y Entropy
#../plot.py -i pg_l_entropy.csv -o 07_pg_l_entropy.svg -x Steps -y "Entropy loss"
#../plot.py -i pg_l_policy.csv -o 08_pg_l_policy.svg -x Steps -y "Policy loss"
#../plot.py -i pg_l_total.csv -o 09_pg_l_total.svg -x Steps -y "Total loss"
#../plot.py -i pg_grad_l2.csv -o 10_pg_grad_l2.svg -x Steps -y "Gradients L2"
#../plot.py -i pg_grad_max.csv -o 11_pg_grad_max.svg -x Steps -y "Gradients max"
#../plot.py -i pg_kl.csv -o 12_pg_kl.svg -x Steps -y "KL"
