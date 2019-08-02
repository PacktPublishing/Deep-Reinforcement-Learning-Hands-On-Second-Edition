#!/usr/bin/env bash
set -x
#../plot.py -i 01_original/avg_reward.csv -o s01_01_reward.svg -x Episodes -y Reward
#../plot.py -i 01_original/steps.csv -o s01_02_steps.svg -x Episodes -y "Episode steps"
#../plot.py -i 01_original/avg_loss.csv -o s01_03_loss.svg  -y Loss
#../plot.py -i 01_original/avg_fps.csv -o s01_04_avg_fps.svg  -y FPS

#../plot.py -i 01_original/avg_reward.csv -i 00_slow_grads/avg_reward.csv -o s00_01_reward.svg -x Episodes -y Reward -l Baseline -l "Slow grads"
#../plot.py -i 01_original/avg_fps.csv    -i 00_slow_grads/avg_fps.csv -o s00_02_avg_fps.svg  -y FPS -l Baseline -l "Slow grads"

../plot.py -i 01_original/avg_reward.csv -i 02_n_envs/2_avg_reward.csv -i 02_n_envs/3_avg_reward.csv -o s02_01_reward_b23.svg  -x Episodes -y Reward -l Baseline -l "2 envs" -l "3 envs"
../plot.py -i 01_original/avg_fps.csv    -i 02_n_envs/2_avg_fps.csv    -i 02_n_envs/3_avg_fps.csv    -o s02_02_avg_fps_b23.svg -y FPS -l Baseline -l "2 envs" -l "3 envs"

../plot.py -i 01_original/avg_reward.csv -i 02_n_envs/4_avg_reward.csv -i 02_n_envs/5_avg_reward.csv -i 02_n_envs/6_avg_reward.csv -o s02_01_reward_b456.svg  -x Episodes -y Reward -l Baseline -l "4 envs" -l "5 envs" -l "6 envs"
../plot.py -i 01_original/avg_fps.csv    -i 02_n_envs/4_avg_fps.csv    -i 02_n_envs/5_avg_fps.csv    -i 02_n_envs/6_avg_fps.csv    -o s02_02_avg_fps_b456.svg -y FPS -l Baseline -l "4 envs" -l "5 envs" -l "6 envs"

#../plot.py -i 01_original/avg_reward.csv -i 03_parallel/avg_reward.csv -o s03_01_reward.svg -x Episodes -y Reward -l Baseline -l Parallel
#../plot.py -i 01_original/avg_fps.csv    -i 03_parallel/avg_fps.csv -o s03_02_avg_fps.svg  -y FPS -l Baseline -l Parallel
