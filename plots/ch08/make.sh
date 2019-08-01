#!/usr/bin/env bash
set -x
#../plot.py -i 01_original/avg_reward.csv -o s01_01_reward.svg -x Episodes -y Reward
#../plot.py -i 01_original/steps.csv -o s01_02_steps.svg -x Episodes -y "Episode steps"
#../plot.py -i 01_original/avg_loss.csv -o s01_03_loss.svg  -y "Loss"
#../plot.py -i 01_original/avg_fps.csv -o s01_04_avg_fps.svg  -y "FPS"

../plot.py -i 01_original/avg_reward.csv -i 00_slow_grads/avg_reward.csv -o s00_01_reward.svg -x Episodes -y Reward -l Baseline -l "Slow grads"
../plot.py -i 01_original/steps.csv      -i 00_slow_grads/steps.csv -o s00_02_steps.svg -x Episodes -y "Episode steps" -l Baseline -l "Slow grads"
../plot.py -i 01_original/avg_loss.csv   -i 00_slow_grads/avg_loss.csv -o s00_03_loss.svg  -y "Loss" -l Baseline -l "Slow grads"
../plot.py -i 01_original/avg_fps.csv    -i 00_slow_grads/avg_fps.csv -o s00_04_avg_fps.svg  -y "FPS" -l Baseline -l "Slow grads"
