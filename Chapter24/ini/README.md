Configuration files with training/testing settings.

# cube2x2-paper-d200
Method from the paper applied to 2x2 cube with scrambling depth 200 during the training.

Best policy is achieved after 8k batches (3.5 hours on 1080Ti), after 10k batches training diverges.

# cube2x2-zero-goal-d200
The same as in paper, but target value for goal states set to zero, which helps convergence a lot
