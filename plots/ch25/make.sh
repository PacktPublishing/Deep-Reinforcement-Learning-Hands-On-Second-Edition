#!/usr/bin/env bash
#../plot.py -i tigers-dqn-reward.csv -o tigers-dqn-reward.svg -x Episodes -y Reward
#../plot.py -i tigers-dqn-steps.csv -o tigers-dqn-steps.svg -x Episodes -y Steps
#../plot.py -i tigers-dqn-loss.csv -o tigers-dqn-loss.svg -x Steps -y "Loss"
#../plot.py -i tigers-dqn-epsilon.csv -o tigers-dqn-epsilon.svg -x Steps -y "Epsilon"
#../plot.py -i tigers-dqn-test-reward.csv -o tigers-dqn-test-reward.svg -x Steps -y "Test reward"
#../plot.py -i tigers-dqn-test-steps.csv -o tigers-dqn-test-steps.svg -x Steps -y "Test steps"

../plot.py -i double-dqn-reward.csv -o double-dqn-reward.svg -x Episodes -y Reward
../plot.py -i double-dqn-steps.csv -o double-dqn-steps.svg -x Episodes -y Steps
../plot.py -i double-dqn-loss.csv -o double-dqn-loss.svg -x Steps -y "Loss"
../plot.py -i double-dqn-epsilon.csv -o double-dqn-epsilon.svg -x Steps -y "Epsilon"
../plot.py -i double-dqn-test-reward.csv -o double-dqn-test-reward.svg -x Steps -y "Test reward"
../plot.py -i double-dqn-test-steps.csv -o double-dqn-test-steps.svg -x Steps -y "Test steps"


