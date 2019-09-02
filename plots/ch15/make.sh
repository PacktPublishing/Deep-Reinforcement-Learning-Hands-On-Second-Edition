#!/usr/bin/env bash
set -x
#../plot.py -i basic_1_reward.csv -o 01_basic_1_reward.svg -x Episodes -y Reward
#../plot.py -i basic_1_steps.csv -o 02_basic_1_steps.svg -x Episodes -y Steps
#../plot.py -i basic_1_valreward.csv -o 03_basic_1_val.svg -x Steps -y Reward
#../plot.py -i basic_10_reward.csv -o 04_basic_10_reward.svg -x Episodes -y Reward
#../plot.py -i basic_10_steps.csv -o 05_basic_10_steps.svg -x Episodes -y Steps
#../plot.py -i basic_10_val.csv -o 06_basic_10_val.svg -x Steps -y Reward
#../plot.py -i basic_25_small_reward.csv -i basic_25_med_reward.csv -o 07_basic_25_reward.svg -x Episodes -y Reward -l small -l medium
#../plot.py -i basic_25_small_steps.csv -i basic_25_med_steps.csv -o 08_basic_25_steps.svg -x Episodes -y Steps -l small -l medium
#../plot.py -i basic_25_small_val.csv -i basic_25_med_val.csv -o 09_basic_25_val.svg -x Episodes -y Reward -l small -l medium
#../plot.py -i basic_200_reward.csv -o 10_basic_200_reward.svg -x Episodes -y Reward
#../plot.py -i basic_200_steps.csv -o 11_basic_200_steps.svg -x Episodes -y Steps
#../plot.py -i basic_200_val.csv -o 12_basic_200_val.svg -x Steps -y Reward
#../plot.py -i lm_pre_1_small_reward.csv -o 13_lm_pre_1_small_reward.svg -x Episodes -y Reward
#../plot.py -i lm_pre_1_small_loss.csv -o 14_lm_pre_1_small_loss.svg -x Steps -y Loss
#../plot.py -i lm_pre_1_med_reward.csv -o 15_lm_pre_1_med_reward.svg -x Episodes -y Reward
#../plot.py -i lm_pre_1_med_loss.csv -o 16_lm_pre_1_med_loss.svg -x Steps -y Loss --ylog
#../plot.py -i lm_pre_5_small_reward.csv -i lm_pre_5_med_reward.csv -o 17_lm_pre_5_reward.svg -x Episodes -y Reward -l small -l medium --lloc "upper right"
#../plot.py -i lm_pre_5_small_steps.csv -i lm_pre_5_med_steps.csv -o 18_lm_pre_5_steps.svg -x Episodes -y Steps -l small -l medium
#../plot.py -i lm_dqn_1_small_reward.csv -o 19_lm_dqn_1_small_reward.svg -x Episodes -y Reward
#../plot.py -i lm_dqn_1_small_loss.csv -o 20_lm_dqn_1_small_loss.svg -x Steps -y Loss
#../plot.py -i lm_dqn_1_med_reward.csv -o 21_lm_dqn_1_med_reward.svg -x Episodes -y Reward
#../plot.py -i lm_dqn_1_med_loss.csv -o 22_lm_dqn_1_med_loss.svg -x Steps -y Loss
../plot.py -i lm_dqn_1_small_val.csv -i lm_dqn_1_med_val.csv -o 23_lm_dqn_1_val.svg -x Steps -y Reward -l small -l medium
