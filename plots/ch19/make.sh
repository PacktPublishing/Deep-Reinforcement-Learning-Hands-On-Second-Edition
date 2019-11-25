#!/usr/bin/env bash
set -x
#../plot.py -i ac-reward.csv -o 02-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i ac-steps.csv -o 02-02-steps.svg -x Steps -y "Train episode steps"
#
#../plot.py -i ac-loss-policy.csv -o 03-01-loss-policy.svg -x Steps -y "Policy loss"
#../plot.py -i ac-loss-value.csv -o 03-02-loss-value.svg -x Steps -y "Value loss"
#../plot.py -i ac-loss-entropy.csv -o 04-loss-entropy.svg -x Steps -y "Entropy loss"
#
#../plot.py -i ac-test-reward.csv -o 05-01-test-reward.svg -x Steps -y "Test reward"
#../plot.py -i ac-test-steps.csv -o 05-02-test-steps.svg -x Steps -y "Test episode steps"

#../plot.py -i aa-reward.csv -o 06-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i aa-steps.csv -o 06-02-steps.svg -x Steps -y "Train episode steps"
#
#../plot.py -i aa-loss-policy.csv -o 07-01-loss-policy.svg -x Steps -y "Policy loss"
#../plot.py -i aa-loss-value.csv -o 07-02-loss-value.svg -x Steps -y "Value loss"
#../plot.py -i aa-loss-entropy.csv -o 08-loss-entropy.svg -x Steps -y "Entropy loss"
#
#../plot.py -i aa-test-reward.csv -o 09-01-test-reward.svg -x Steps -y "Test reward"
#../plot.py -i aa-test-steps.csv -o 09-02-test-steps.svg -x Steps -y "Test episode steps"

#../plot.py -i pc-reward.csv -o 10-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i pc-steps.csv -o 10-02-steps.svg -x Steps -y "Train episode steps"
#
#../plot.py -i pc-test-reward.csv -o 11-01-test-reward.svg -x Steps -y "Test reward"
#../plot.py -i pc-test-steps.csv -o 11-02-test-steps.svg -x Steps -y "Test episode steps"

#../plot.py -i pa-reward.csv -o 12-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i pa-test-reward.csv -o 12-02-test-reward.svg -x Steps -y "Test reward"
#
#../plot.py -i pa-reward.csv -i aa-reward.csv -o 13-01-reward-a2c-ppo.svg -x Steps -y "Mean train reward" -l "PPO" -l "A2C"
#../plot.py -i pa-test-reward.csv -i aa-test-reward.csv -o 13-02-test-reward-a2c-ppo.svg -x Steps -y "Test reward" -l "PPO" -l "A2C"

#../plot.py -i tc-reward.csv -o 14-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i tc-test-reward.csv -o 14-02-test-reward.svg -x Steps -y "Test reward"
#
#../plot.py -i tc-reward.csv -i pc-reward.csv -o 15-01-reward-trpo-ppo.svg -x Steps -y "Mean train reward" -l "TRPO" -l "PPO"
#../plot.py -i tc-test-reward.csv -i pc-test-reward.csv -o 15-02-test-reward-a2c-ppo.svg -x Steps -y "Test reward" -l "TRPO" -l "PPO"

#../plot.py -i ta-reward.csv -o 16-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i ta-test-reward.csv -o 16-02-test-reward.svg -x Steps -y "Test reward"
#
#../plot.py -i ta-reward.csv -i pa-reward.csv -o 17-01-reward-trpo-ppo.svg -x Steps -y "Mean train reward" -l "TRPO" -l "PPO"
#../plot.py -i ta-test-reward.csv -i pa-test-reward.csv -o 17-02-test-reward-trpo-ppo.svg -x Steps -y "Test reward" -l "TRPO" -l "PPO"

#../plot.py -i kc-reward.csv -o 18-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i kc-test-reward.csv -o 18-02-test-reward.svg -x Steps -y "Test reward"

#../plot.py -i ka-reward.csv -o 19-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i ka-test-reward.csv -o 19-02-test-reward.svg -x Steps -y "Test reward"

#../plot.py -i sc-reward.csv -o 20-01-reward100.svg -x Steps -y "Mean train reward"
#../plot.py -i sc-test-reward.csv -o 20-02-test-reward.svg -x Steps -y "Test reward"
#../plot.py -i sc-reward.csv -i pa-reward.csv -o 21-01-reward-sac-ppo.svg -x Steps -y "Mean train reward" -l "SAC" -l "PPO"
#../plot.py -i sc-test-reward.csv -i pa-test-reward.csv -o 21-02-test-reward-sac-ppo.svg -x Steps -y "Test reward" -l "SAC" -l "PPO"

../plot.py -i sa-reward.csv -o 22-01-reward100.svg -x Steps -y "Mean train reward"
../plot.py -i sa-test-reward.csv -o 22-02-test-reward.svg -x Steps -y "Test reward"
../plot.py -i sa-reward.csv -i pa-reward.csv -o 23-01-reward-sac-ppo.svg -x Steps -y "Mean train reward" -l "SAC" -l "PPO"
../plot.py -i sa-test-reward.csv -i pa-test-reward.csv -o 23-02-test-reward-sac-ppo.svg -x Steps -y "Test reward" -l "SAC" -l "PPO"
