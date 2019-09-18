#!/usr/bin/env bash
set -x
../plot.py -i 01_h_critic_ref.csv -o 01_h_critic_ref.svg -x Steps -y Value
../plot.py -i 01_h_loss_actor.csv -o 02_h_loss_actor.svg -x Steps -y Loss
../plot.py -i 01_h_loss_critic.csv -o 03_h_loss_critic.svg -x Steps -y Loss
../plot.py -i 01_h_reward.csv -o 04_h_reward.svg -x Steps -y Reward
../plot.py -i 01_h_test_reward.csv -o 05_h_test_reward.svg -x Steps -y Reward
