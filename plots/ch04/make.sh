#!/usr/bin/env bash
set -x
../plot.py -i cp-loss.csv -o cp-loss.svg -x Steps -y Loss
../plot.py -i cp-rw_bound.csv -o cp-rw_bound.svg -x Steps -y "Reward boundary"
../plot.py -i cp-reward.csv -o cp-reward.svg -x Steps -y "Mean reward"
../plot.py -i fln-loss.csv -o fln-loss.svg -x Steps -y Loss
../plot.py -i fln-rw_bound.csv -o fln-rw_bound.svg -x Steps -y "Reward boundary"
../plot.py -i fln-reward.csv -o fln-reward.svg -x Steps -y "Mean reward"
../plot.py -i flt-loss.csv -o flt-loss.svg -x Steps -y Loss
../plot.py -i flt-rw_bound.csv -o flt-rw_bound.svg -x Steps -y "Reward boundary"
../plot.py -i flt-reward.csv -o flt-reward.svg -x Steps -y "Mean reward"
../plot.py -i flns-loss.csv -o flns-loss.svg -x Steps -y Loss
../plot.py -i flns-rw_bound.csv -o flns-rw_bound.svg -x Steps -y "Reward boundary"
../plot.py -i flns-reward.csv -o flns-reward.svg -x Steps -y "Mean reward"
