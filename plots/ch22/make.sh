#!/usr/bin/env bash
set -x

#../plot.py -i a2c-test-reward.csv -o 05-01-test-reward.svg -x Steps -y "Test reward"
#../plot.py -i a2c-test-steps.csv -o 05-02-test-steps.svg -x Steps -y "Test steps"
#
#../plot.py -i a2c-total-reward.csv -o 06-01-total-reward.svg -x Steps -y "Train reward"
#../plot.py -i a2c-total-steps.csv -o 06-02-total-steps.svg -x Steps -y "Train steps"
#
#../plot.py -i a2c-adv.csv -o 07-01-adv.svg -x Steps -y "Advantage"
#../plot.py -i a2c-loss-ent.csv -o 07-02-loss-ent.svg -x Steps -y "Entropy loss"
#
#../plot.py -i a2c-loss-policy.csv -o 08-01-loss-policy.svg -x Steps -y "Policy loss"
#../plot.py -i a2c-loss-value.csv -o 08-02-loss-value.svg -x Steps -y "Value loss"

../plot.py -i em-loss-obs.csv -o 09-01-loss-obs.svg -x Steps -y "Observation loss"
../plot.py -i em-loss-reward.csv -o 09-02-loss-reward.svg -x Steps -y "Reward loss"
