#!/usr/bin/env bash
set -x
../plot.py -i ff-reward_train.csv -o ff-reward_train.svg -x Episodes -y Reward
../plot.py -i ff-steps_train.csv -o ff-steps_train.svg -x Episodes -y Steps
../plot.py -i ff-reward_test.csv -o ff-reward_test.svg
../plot.py -i ff-reward_val.csv -o ff-reward_val.svg
../plot.py -i ff-values_train.csv -o ff-values.svg

../plot.py -i cv-reward_train.csv -o cv-reward_train.svg -x Episodes -y Reward
../plot.py -i cv-reward_val.csv -o cv-reward_val.svg

