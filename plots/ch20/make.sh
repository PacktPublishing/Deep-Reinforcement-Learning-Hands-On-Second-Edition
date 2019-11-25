#!/usr/bin/env bash
set -x
#../plot.py -i escp-reward-max.csv -o 01-01-reward-max.svg -x Steps -y "Maximum reward for batch"
#../plot.py -i escp-reward-mean.csv -o 01-02-reward-mean.svg -x Steps -y "Mean reward for batch"
#
#../plot.py -i escp-reward-std.csv -o 02-01-reward-std.svg -x Steps -y "Standard deviation of batch reward"
#../plot.py -i escp-update-l2.csv -o 02-02-update-l2.svg -x Steps -y "Policy update"

#../plot.py -i eshc-reward-max.csv -o 03-01-reward-max.svg -x Steps -y "Maximum reward for batch"
#../plot.py -i eshc-reward-mean.csv -o 03-02-reward-mean.svg -x Steps -y "Mean reward for batch"
#
#../plot.py -i eshc-reward-std.csv -o 04-01-reward-std.svg -x Steps -y "Standard deviation of batch reward"
#../plot.py -i eshc-update-l2.csv -o 04-02-update-l2.svg -x Steps -y "Policy update"

#../plot.py -i gacp-reward-max.csv -o 05-01-reward-max.svg -x Steps -y "Maximum reward for batch"
#../plot.py -i gacp-reward-mean.csv -o 05-02-reward-mean.svg -x Steps -y "Mean reward for batch"
#../plot.py -i gacp-reward-std.csv -o 06-reward-std.svg -x Steps -y "Standard deviation of batch reward"

../plot.py -i gahc-reward-max.csv -o 07-01-reward-max.svg -x Steps -y "Maximum reward for batch"
../plot.py -i gahc-reward-mean.csv -o 07-02-reward-mean.svg -x Steps -y "Mean reward for batch"
../plot.py -i gahc-reward-std.csv -o 08-reward-std.svg -x Steps -y "Standard deviation of batch reward"
