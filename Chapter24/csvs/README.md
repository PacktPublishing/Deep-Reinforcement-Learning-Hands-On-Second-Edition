Description of produced test results

# First results

Test results from first models (paper versus zero-goal method). Solve tool run for 30k MCTS searches 
(but due to bug, actual amount of steps in some tests was much lower).

````
c2x2-paper-d200-t1.csv
c2x2-zero-goal-d200-t1.csv
c3x3-paper-d200-t1.csv
c3x3-zero-goal-d200-no-decay.csv
c3x3-zero-goal-d200-t1.csv
````

Analysis of the results are in notebook 
https://github.com/Shmuma/rl/blob/master/articles/01_rubic/nbs/01_paper-vs-zero_goal.ipynb

# Fix of wrong steps

Fixed with https://github.com/Shmuma/rl/commit/793aebc81b7bf323a8db930e8224521700383af5#diff-b9a7f0478383b0f6ad54ae87c8769b03

````
c2x2-paper-d200-t1-v2.csv
c2x2-zero-goal-d200-t1-v2.csv
c3x3-paper-d200-t1-v2.csv
c3x3-zero-goal-d200-no-decay-v2.csv
c3x3-zero-goal-d200-t1-v2.csv
````

