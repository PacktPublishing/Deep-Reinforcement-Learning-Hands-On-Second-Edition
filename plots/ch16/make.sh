#!/usr/bin/env bash
set -x
#../plot.py -i cd-reward100.csv -o 05-reward100.svg -x Steps -y "Mean reward" --max-dt 2
#../plot.py -i cd-loss-total.csv -o 06-01-loss-total.svg -x Steps -y "Total loss" --max-dt 2
#../plot.py -i cd-loss-ent.csv -o 06-02-loss-ent.svg -x Steps -y "Entropy loss" --max-dt 2
#../plot.py -i cd-episode-steps.csv -o 07-episode-steps.svg -x Steps -y "Episode steps" --max-dt 2
#../plot.py -i s-nodemo-reward100.csv -o 15-reward100.svg -x Steps -y "Mean reward"
#../plot.py -i s-nodemo-loss-total.csv -o 16-01-loss-total.svg -x Steps -y "Total loss"
#../plot.py -i s-nodemo-loss-ent.csv -o 16-02-loss-ent.svg -x Steps -y "Entropy loss"
#../plot.py -i s-nodemo-episode-steps.csv -o 17-episode-steps.svg -x Steps -y "Episode steps"
#../plot.py -i s-demo-reward100.csv -o 18-reward100.svg -x Steps -y "Mean reward"
#../plot.py -i s-demo-loss-total.csv -o 19-01-loss-total.svg -x Steps -y "Total loss"
#../plot.py -i s-demo-loss-ent.csv -o 19-02-loss-ent.svg -x Steps -y "Entropy loss"
#../plot.py -i s-demo-episode-steps.csv -o 20-episode-steps.svg -x Steps -y "Episode steps"
#../plot.py -i s-demo-reward100.csv -i s-nodemo-reward100.csv -o 21-reward100.svg -l "with demo" -l "without demo" -x Steps -y "Mean reward" --lloc "upper right"
#../plot.py -i s-demo-loss-total.csv -i s-nodemo-loss-total.csv -o 22-01-loss-total.svg -x Steps -y "Total loss" -l "with demo" -l "without demo" --lloc 'upper right'
#../plot.py -i s-demo-loss-ent.csv -i s-nodemo-loss-ent.csv -o 22-02-loss-ent.svg -x Steps -y "Entropy loss" -l "with demo" -l "without demo" --lloc 'upper right'
#../plot.py -i s-demo-episode-steps.csv -i s-nodemo-episode-steps.csv -o 22-episode-steps.svg -x Steps -y "Episode steps" -l "with demo" -l "without demo" --lloc 'upper right'
#../plot.py -i t3-reward100.csv -o 23-reward100.svg -x Steps -y "Mean reward"
#../plot.py -i t3-loss-total.csv -o 24-01-loss-total.svg -x Steps -y "Total loss"
#../plot.py -i t3-loss-ent.csv -o 24-02-loss-ent.svg -x Steps -y "Entropy loss"
#../plot.py -i t3-episode-steps.csv -o 25-episode-steps.svg -x Steps -y "Episode steps"
../plot.py -i cb-nomm-reward100.csv -o 32-reward100.svg -x Steps -y "Mean reward"
../plot.py -i cb-nomm-loss-total.csv -o 33-01-loss-total.svg -x Steps -y "Total loss"
../plot.py -i cb-nomm-loss-ent.csv -o 33-02-loss-ent.svg -x Steps -y "Entropy loss"
../plot.py -i cb-nomm-episode-steps.csv -o 34-episode-steps.svg -x Steps -y "Episode steps"
../plot.py -i cb-mm-reward100.csv -o 35-reward100.svg -x Steps -y "Mean reward"
../plot.py -i cb-mm-loss-total.csv -o 36-01-loss-total.svg -x Steps -y "Total loss"
../plot.py -i cb-mm-loss-ent.csv -o 36-02-loss-ent.svg -x Steps -y "Entropy loss"
../plot.py -i cb-mm-episode-steps.csv -o 37-episode-steps.svg -x Steps -y "Episode steps"
