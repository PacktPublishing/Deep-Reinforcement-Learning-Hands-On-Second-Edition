import ptan
import numpy as np


if __name__ == "__main__":
    q_vals = np.array([[1, 2, 3], [1, -1, 0]])
    print("q_vals")
    print(q_vals)

    selector = ptan.actions.ArgmaxActionSelector()
    print("argmax:", selector(q_vals))

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.0)
    print("epsilon=0.0:", selector(q_vals))

    selector.epsilon = 1.0
    print("epsilon=1.0:", selector(q_vals))

    selector.epsilon = 0.5
    print("epsilon=0.5:", selector(q_vals))
    selector.epsilon = 0.1
    print("epsilon=0.1:", selector(q_vals))

    selector = ptan.actions.ProbabilityActionSelector()
    print("Actions sampled from three prob distributions:")
    for _ in range(10):
        acts = selector(np.array([
            [0.1, 0.8, 0.1],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0]
        ]))
        print(acts)
