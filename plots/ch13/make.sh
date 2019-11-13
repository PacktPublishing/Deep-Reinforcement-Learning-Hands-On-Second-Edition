#!/usr/bin/env bash
set -x
#../plot.py -i data_adv.csv -o data_adv.svg -x Steps -y Advantage
#../plot.py -i data_loss.csv -o data_loss.svg -x Steps -y "Total loss"
#../plot.py -i data_reward.csv -o data_reward.svg -x Steps -y "Smoothed reward"
../plot.py -i grad_adv_0.csv -i grad_adv_1.csv -i grad_adv_2.csv -i grad_adv_3.csv -o grad_adv.svg -x Steps -y Advantage -l 'Process 0' -l 'Process 1' -l 'Process 2' -l 'Process 3'
../plot.py -i grad_loss_0.csv -i grad_loss_1.csv -i grad_loss_2.csv -i grad_loss_3.csv -o grad_loss.svg -x Steps -y "Total Loss" -l 'Process 0' -l 'Process 1' -l 'Process 2' -l 'Process 3'
../plot.py -i grad_reward_0.csv -i grad_reward_1.csv -i grad_reward_2.csv -i grad_reward_3.csv -o grad_reward.svg -x Steps -y "Smoothed reward" -l 'Process 0' -l 'Process 1' -l 'Process 2' -l 'Process 3'
