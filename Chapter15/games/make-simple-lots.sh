#!/usr/bin/env bash
tw-make custom --world-size 5 --nb-objects 10 --quest-length 5  --quest-breadth 1 --seed 0 --output simple-val.ulx

for i in `seq 1 50`; do
    tw-make custom --world-size 5 --nb-objects 10 --quest-length 5 --quest-breadth 1 --seed $i --output simple$i.ulx
done
