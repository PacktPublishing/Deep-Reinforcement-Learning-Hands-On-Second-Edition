#!/usr/bin/env bash
set -x
#../plot.py -i cp-reward_nobl.csv -o cp-reward-nobl.svg -x Steps -y Reward
#../plot.py -i cp-reward_bl.csv -o cp-reward-bl.svg -x Steps -y Reward
#../plot.py -i cp-reward100_nobl.csv -i cp-reward100_bl.csv -l "No baseline" -l "Baseline" -o cp-reward100.svg -x Steps -y "Smoothed reward"
#../plot.py -i cp-l2-nobl.csv -i cp-l2-bl.csv -l "No baseline" -l "Baseline" -o cp-l2.svg -x Steps -y "Grads L2" --ylog
#../plot.py -i cp-max-nobl.csv -i cp-max-bl.csv -l "No baseline" -l "Baseline" -o cp-max.svg -x Steps -y "Grads Max" --ylog
#../plot.py -i cp-var-nobl.csv -i cp-var-bl.csv -l "No baseline" -l "Baseline" -o cp-var.svg -x Steps -y "Grads Var" --ylog

../plot.py -i a2c-batch_rewards.csv -o 08-01-batch_rewards.svg -x Steps -y "Batch reward"
../plot.py -i a2c-reward.csv -o 08-02-reward.svg -x Steps -y "Mean reward"

../plot.py -i a2c-loss_entropy.csv -o 09-01-loss_ent.svg -x Steps -y "Entropy loss"
../plot.py -i a2c-loss_policy.csv -o 09-02-loss_policy.svg -x Steps -y "Policy loss"
../plot.py -i a2c-loss_value.csv -o 10-01-loss_value.svg -x Steps -y "Value loss"
../plot.py -i a2c-loss_total.csv -o 10-02-loss_total.svg -x Steps -y "Total loss"

../plot.py -i a2c-advantage.csv -o 11-01-advanatage.svg -x Steps -y Advantage
../plot.py -i a2c-grad_l2.csv -o 11-02-grad-l2.svg -x Steps -y "Gradient L2"
../plot.py -i a2c-grad_max.csv -o 12-01-grad_max.svg -x Steps -y "Gradient max"
../plot.py -i a2c-grad_var.csv -o 12-02-grad_var.svg -x Steps -y "Gradient variance"




