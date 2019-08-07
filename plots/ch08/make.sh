#!/usr/bin/env bash
set -x

#../plot.py -i 01_base/avg_reward.csv -o 01_reward.svg -x Episodes -y Reward
#../plot.py -i 01_base/steps.csv -o 01_steps.svg -x Episodes -y Steps
#../plot.py -i 01_base/avg_fps.csv -o 01_fps.svg -y FPS
#../plot.py -i 01_base/avg_loss.csv -o 01_loss.svg -y Loss

#../plot.py -i 01_base/avg_reward.csv -i 02_n_steps/2_avg_reward.csv -i 02_n_steps/3_avg_reward.csv \
#    -o 02_reward_b23.svg -x Episodes -y Reward -l Baseline -l Steps=2 -l Steps=3

#../plot.py -i 01_base/steps.csv -i 02_n_steps/3_steps.csv \
#    -o 02_steps_b3.svg -x Episodes -y "Episode steps" -l Baseline -l Steps=3

#../plot.py -i 02_n_steps/3_avg_reward.csv -i 02_n_steps/4_avg_reward.csv -i 02_n_steps/5_avg_reward.csv -i 02_n_steps/6_avg_reward.csv \
#    -o 02_reward_3456.svg -x Episodes -y Reward -l Steps=3 -l Steps=4 -l Steps=5 -l Steps=6

#../plot.py -i 03_double/false_avg_reward.csv -i 03_double/true_avg_reward.csv \
#    -o 03_reward.png -x Episodes -y Reward -l "DQN" -l "Double DQN"

#../plot.py -i 03_double/false_values.csv -i 03_double/true_values.csv \
#    -o 03_values.png -y Values -l "DQN" -l "Double DQN"

#../plot.py -i 01_base/avg_reward.csv -i 04_noisy/avg_reward.csv -o 04_reward.svg -x Episodes -y Reward -l Baseline -l "Noisy net"
#../plot.py -i 01_base/steps.csv -i 04_noisy/steps.csv -o 04_steps.svg -x Episodes -y "Episode steps" -l Baseline -l "Noisy net"
#../plot.py -i 04_noisy/snr_1.csv -o 04_snr_1.svg -y "Signal/noise in layer 1"
#../plot.py -i 04_noisy/snr_2.csv -o 04_snr_2.svg -y "Signal/noise in layer 2"

#../plot.py -i 01_base/avg_reward.csv -o 05_reward_baseline.svg -x Episodes -y Reward
#../plot.py -i 05_prio/avg_reward.csv -o 05_reward_prio.svg -x Episodes -y Reward
#../plot.py -i 01_base/avg_loss.csv -i 05_prio/avg_loss.csv -o 05_loss.svg -y Loss -l Baseline -l "Prioritized replay"

#../plot.py -i 01_base/avg_reward.csv -i 06_dueling/avg_reward.csv -o 06_reward.svg -x Episodes -y Reward -l Baseline -l Dueling
#../plot.py -i 01_base/steps.csv -i 06_dueling/steps.csv -o 06_steps.svg -x Episodes -y Steps -l Baseline -l Dueling
#../plot.py -i 01_base/avg_loss.csv -i 06_dueling/avg_loss.csv -o 06_loss.svg -y Loss -l Baseline -l Dueling
#../plot.py -i 06_dueling/adv.csv -o 06_adv.svg -y Advantage
#../plot.py -i 06_dueling/val.csv -o 06_val.svg -y Value

#../plot.py -i 01_base/avg_reward.csv -i 07_distrib/avg_reward.csv -o 07_reward.svg -x Episodes -y Reward -l Baseline -l "Categorical DQN"
#../plot.py -i 07_distrib/avg_loss.csv -o 07_loss.svg -y Loss

#../plot.py -i 01_base/avg_reward.csv -i 08_rainbow/avg_reward.csv -o 08_reward_comp.svg -x Episodes -y Reward -l Baseline -l "Combined system"
#../plot.py -i 08_rainbow/avg_reward.csv -o 08_reward_only.svg -x Episodes -y Reward
../plot.py -i 01_base/avg_fps.csv -i 08_rainbow/avg_fps.csv -o 08_fps.svg -y FPS -l Baseline -l "Combined system"
../plot.py -i 01_base/steps.csv -i 08_rainbow/steps.csv -o 08_steps.svg -y Steps -l Baseline -l "Combined system"
