#!/usr/bin/env bash
set -x
../plot.py -i xe-bleu.csv -o xe-bleu.svg -x Epochs -y "BLEU on train"
../plot.py -i xe-bleu-test.csv -o xe-bleu-test.svg -x Epochs -y "BLEU on test"
../plot.py -i xe-loss.csv -o xe-loss.svg -x Epochs -y Loss
#../plot.py -i sc1-bleu-argmax.csv -o sc1-bleu-argmax.svg -x Steps -y "BLEU from argmax"
#../plot.py -i sc1-bleu-sample.csv -o sc1-bleu-sample.svg -x Steps -y "BLEU from sampling"
#../plot.py -i sc1-bleu-test.csv -o sc1-bleu-test.svg -x Steps -y "BLEU on test"
#../plot.py -i sc1-skipped-samples.csv -o sc1-skipped-samples.svg -x Steps -y "Skipped samples"

#../plot.py -i sc2-bleu-argmax.csv -o sc2-bleu-argmax.svg -x Steps -y "BLEU from argmax"
#../plot.py -i sc2-bleu-sample.csv -o sc2-bleu-sample.svg -x Steps -y "BLEU from sampling"
#../plot.py -i sc2-bleu-test.csv -o sc2-bleu-test.svg -x Steps -y "BLEU on test"
