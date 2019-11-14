#!/usr/bin/env bash
set -x
../plot.py -i xe-bleu.csv -o xe-bleu.svg -x Epoches -y "BLEU on train"
../plot.py -i xe-bleu-test.csv -o xe-bleu-test.svg -x Epoches -y "BLEU on test"
../plot.py -i xe-loss.csv -o xe-loss.svg -x Epoches -y Loss
