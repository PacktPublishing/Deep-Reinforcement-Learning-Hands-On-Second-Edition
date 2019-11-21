#!/usr/bin/env bash
set -x
../plot.py -i a2c-reward100.csv -o 02-01-reward100.svg -x Steps -y "Mean train reward"
../plot.py -i a2c-episode-steps.csv -o 02-02-episode-steps.svg -x Steps -y "Train episode steps"
../plot.py -i a2c-test-reward.csv -o 04-01-test-reward.svg -x Steps -y "Test reward"
../plot.py -i a2c-test-steps.csv -o 04-02-test-steps.svg -x Steps -y "Test episode steps"

../plot.py -i dd-reward100.csv -o 06-01-reward100.svg -x Steps -y "Mean train reward"
../plot.py -i dd-episode-steps.csv -o 06-02-episode-steps.svg -x Steps -y "Train episode steps"
../plot.py -i dd-loss-actor.csv -o 07-01-loss-actor.svg -x Steps -y "Actor loss"
../plot.py -i dd-loss-critic.csv -o 07-02-loss-critic.svg -x Steps -y "Critic loss"
../plot.py -i dd-test-reward.csv -o 08-01-test-reward.svg -x Steps -y "Test reward"
../plot.py -i dd-test-steps.csv -o 08-02-test-steps.svg -x Steps -y "Test episode steps"

../plot.py -i d4-reward100.csv -o 09-01-reward100.svg -x Steps -y "Mean train reward"
../plot.py -i d4-episode-steps.csv -o 09-02-episode-steps.svg -x Steps -y "Train episode steps"
../plot.py -i d4-loss-actor.csv -o 10-01-loss-actor.svg -x Steps -y "Actor loss"
../plot.py -i d4-loss-critic.csv -o 10-02-loss-critic.svg -x Steps -y "Critic loss"
../plot.py -i d4-test-reward.csv -o 11-01-test-reward.svg -x Steps -y "Test reward"
../plot.py -i d4-test-steps.csv -o 11-02-test-steps.svg -x Steps -y "Test episode steps"

