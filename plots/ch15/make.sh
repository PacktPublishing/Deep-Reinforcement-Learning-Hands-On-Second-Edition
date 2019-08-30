#!/usr/bin/env bash
set -x
#../plot.py -i basic_1_reward.csv -o 01_basic_1_reward.svg -x Episodes -y Reward
#../plot.py -i basic_1_steps.csv -o 02_basic_1_steps.svg -x Episodes -y Steps
../plot.py -i basic_1_valreward.csv -o 03_basic_1_val.svg -x Steps -y Reward
#../plot.py -i basic_10_reward.csv -o 04_basic_10_reward.svg -x Episodes -y Reward
#../plot.py -i basic_10_steps.csv -o 05_basic_10_steps.svg -x Episodes -y Steps
../plot.py -i basic_10_val.csv -o 06_basic_10_val.svg -x Steps -y Reward
#../plot.py -i basic_25_small_reward.csv -i basic_25_med_reward.csv -o 07_basic_25_reward.svg -x Episodes -y Reward -l small -l medium
#../plot.py -i basic_25_small_steps.csv -i basic_25_med_steps.csv -o 08_basic_25_steps.svg -x Episodes -y Steps -l small -l medium
#../plot.py -i basic_25_small_val.csv -i basic_25_med_val.csv -o 09_basic_25_val.svg -x Episodes -y Reward -l small -l medium
../plot.py -i basic_200_reward.csv -o 10_basic_200_reward.svg -x Episodes -y Reward
../plot.py -i basic_200_steps.csv -o 11_basic_200_steps.svg -x Episodes -y Steps
../plot.py -i basic_200_val.csv -o 12_basic_200_val.svg -x Steps -y Reward
