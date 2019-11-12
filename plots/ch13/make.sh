#!/usr/bin/env bash
set -x
../plot.py -i data_adv.csv -o data_adv.svg -x Steps -y Advantage
../plot.py -i data_loss.csv -o data_loss.svg -x Steps -y "Total loss"
../plot.py -i data_reward.csv -o data_reward.svg -x Steps -y "Smoothed reward"
