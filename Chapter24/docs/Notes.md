# 2018-11-08

MCTS performance drops with increase of c:
````
(art_01_cube) shmuma@gpu:~/work/rl/articles/01_rubic$ ./solver.py -e cube2x2 -m saves/cube2x2-zero-goal-d200-t1/best_1.4547e-02.dat --max-steps 1000 --cuda -r 20
2018-11-08 06:33:56,195 INFO Using environment CubeEnv('cube2x2')
2018-11-08 06:33:58,169 INFO Network loaded from saves/cube2x2-zero-goal-d200-t1/best_1.4547e-02.dat
2018-11-08 06:33:58,169 INFO Got task [10, 1, 0, 11, 4, 3, 3, 2, 11, 1, 10, 11, 8, 1, 9, 6, 1, 3, 3, 8], solving...
2018-11-08 06:34:01,330 INFO Maximum amount of steps has reached, cube wasn't solved. Did 1001 searches, speed 316.77 searches/s
````

* c=10k: 316 searches/s
* c=100k: 58 searches/s
* c=1m: 4.94 searches/s

Root tree state is the same for 10k and 100k.

Mean search depth: 1k: 57, 10k: 129.7, 100k: 861

Conclusion:
Larger C makes tree taller by exploring less options around, but delving deeper into the search space.
This leads to longer search paths which take more and more time to back up.
It is likely that my C value is too large and I just need to speed up MCTS.

TODO: 
* measure depths of resulting tree
* analyze the length of solution (both naive and BFS)
* check effect of C on those parameters

Depths with 1000 steps limit:
* c=1m:   {'min': 1, 'max': 16, 'mean': 7.849963731321631, 'leaves': 34465}
* c=100k: {'min': 1, 'max': 17, 'mean': 9.103493774652236, 'leaves': 71241}
* c=10k:  {'min': 1, 'max': 18, 'mean': 10.28626504647809, 'leaves': 70033}
* c=1k:   {'min': 1, 'max': 18, 'mean': 9.942448384493218, 'leaves': 76818}
* c=100:  {'min': 1, 'max': 14, 'mean': 8.938883245826121, 'leaves': 69899}
* c=10:   {'min': 1, 'max': 13, 'mean': 8.59500956472128,  'leaves': 59594}

Depths with 10000 steps limit:
* c=10k:  {'min': 1, 'max': 27, 'mean': 15.374430775030191, 'leaves': 1289253}
* c=1k:   {'min': 1, 'max': 26, 'mean': 14.057022074409328, 'leaves': 1004874}
* c=100:  {'min': 1, 'max': 19, 'mean': 12.376234716455224, 'leaves': 1113616}
* c=10:   {'min': 1, 'max': 19, 'mean': 11.707333613164712, 'leaves': 886248}


# Weird case in search
Looks like virtual loss needs tuning as well

````
(art_01_cube) shmuma@gpu:~/work/rl/articles/01_rubic$ ./solver.py -e cube2x2 -m saves/cube2x2-zero-goal-d200-t1/best_1.4547e-02.dat --max-steps 10000 --cuda -r 10 --seed 41
2018-11-08 14:43:34,360 INFO Using environment CubeEnv('cube2x2')
2018-11-08 14:43:36,328 INFO Network loaded from saves/cube2x2-zero-goal-d200-t1/best_1.4547e-02.dat
2018-11-08 14:43:36,329 INFO Got task [6, 5, 3, 2, 6, 9, 11, 4, 8, 4], solving...
2018-11-08 14:43:59,362 INFO On step 8544 we found goal state, unroll. Speed 370.94 searches/s
2018-11-08 14:43:59,627 INFO Tree depths: {'max': 22, 'mean': 11.557521172600728, 'leaves': 604673}
2018-11-08 14:43:59,627 INFO Solutions: naive [10, 0, 6, 3, 9, 6, 0, 2, 8, 4, 0, 10, 8, 2, 4, 6, 1, 0, 5, 6, 0, 9, 3, 11, 6, 7, 3, 9, 2, 8, 6, 0, 9, 3, 8, 2, 4, 10, 11, 5, 7, 1, 5, 11, 10, 1, 7, 11, 5, 9, 3, 8, 2, 5, 11, 10, 4, 7, 1, 0, 2, 6, 0, 8, 8, 2, 0, 0, 3, 9, 6, 6, 11, 5, 4, 6, 10, 0, 6, 8, 2, 4, 4, 1, 10, 10, 4, 4, 4, 4, 10, 10, 1, 7, 0, 6, 7, 10, 3, 9, 0, 10, 1, 6, 4, 7, 10, 4, 6, 0, 2, 8, 8, 2, 7, 1, 4, 10, 1, 10, 3, 7, 7, 1, 9, 3, 1, 4, 7, 7, 10, 3, 0, 6, 9, 4, 1, 7, 3, 9, 8, 2, 0, 6, 4, 10, 7, 1, 11, 5, 9, 3, 2, 8, 5, 11, 6, 0, 10, 6, 3] (161)
````

# 2018-11-16
## Experiment with lower c, but more steps

With decrease of C, solve ratio drops (with fixed amount of steps). But lower C is generally faster. 
Maybe more steps will increase the solve ratio and will fit the same time frame?

Experiments:
* t3.1-c2x2-mcts-c=1000.csv
* t3.1-c2x2-mcts-c=100.csv
* t3.1-c2x2-mcts-c=100-steps=60k.csv
* t3.1-c2x2-mcts-c=100-steps=100k.csv
* t3.1-c2x2-mcts-c=10.csv
* t4-c2x2-mcts-c=10-steps=100k.csv
* t4-c2x2-mcts-c=10-steps=200k.csv
* t4-c2x2-mcts-c=10-steps=500k.csv

Charts are in 04_mcts_C-extra-data.ipynb

Conclusion: c=100 is optimal, more steps solve all cubes (checked up to depth 50)

## Experiment with batched search

Main question: what increase in speed we've got and did it decreased quality of search?

Experiments:
* t4-c2x2-c=100-steps=100k.csv: batch=1
* t4-c2x2-c=100-steps=100k-b10.csv: batch=10
* t4-c2x2-c=100-steps=100k-b100.csv: batch=100

Charts are in 05_batch_search.ipynb

With larger batch, solve ration drops. Speed increases, but not proportionally - b=100 has speed increase 2-3 times in 
terms of raw steps.

Maybe, we need to tune virtual loss as well. Do an experiment on it.

## Experiment with models with different loss

Main question: does lower loss mean better solve ratio?

Setup:
cube 2x2, c=100, steps=100k, batch=1, models:
* t2-zero-goal-best_1.4547e-02.dat
* best_3.0742e-02.dat
* best_6.0737e-02.dat
* best_1.0366e-01.dat

Results:
* t4-c2x2-mcts-c=100-steps=100k.csv
* t5-c2x2-3.0742e-02.csv
* t5-c2x2-6.0737e-02.csv
* t5-c2x2-1.0366e-01.csv

~~Started, waiting for results~~
**2018-11-19**: run done, notebook is in nbs/06_compare_models
Conclusion: with lower loss, solve ratio increases significantly.

## Experiment with different virtual loss

Setup:
cube 2x2, c=100, steps=100k, batch=10
* nu=100 (default)
* nu=10
* nu=1
* nu=1000

Results:
* t4-c2x2-mcts-c=100-steps=100k-b10.csv
* t6-c2x2-nu=10.csv
* t6-c2x2-nu=1.csv
* t6-c2x2-nu=1000.csv

## Final check -- compare best paper solution with best zero-goal

Results:
* t4-c2x2-mcts-c=100-steps=100k.csv
* t7-best-paper-1.8184e-1.csv
