#!/usr/bin/env bash
set -x
../plot.py -i 01_original/avg_reward.csv -o s01_01_reward.svg -x Episodes -y Reward
../plot.py -i 01_original/steps.csv -o s01_02_steps.svg -x Episodes -y "Episode steps"
../plot.py -i 01_original/avg_loss.csv -o s01_03_loss.svg  -y "Loss"
../plot.py -i 01_original/fps.csv -o s01_04_fps.svg  -y "FPS"
