"""
Method: greedy Bayesian.

-------------------------------
Suppose we are given 3 machines with payouts following Bern(0.33), Bern(0.55)
and Bern(0.6) respectively (same as used in thompson_sampling_bernoulli.py)
and we play 1000 rounds in total.

We use a greedy Bayesian approach which can be seen as a naive precursor of
Thompson Sampling.
"""

from infrastructure import *  # noqa: F403
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# initialize machines and beta priors (uniform dist.)
machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]]  # noqa: F405
priors = [[1, 1] for i in range(len(machines))]


class GreedyBayesianBernoulli(Game):
    """Greedy Bayesian using beta prior."""

    def __init__(self, prior_parameters, turns, *machines):
        super().__init__(turns, *machines)
        self.parameters = prior_parameters

    def _update(self, index, outcome):
        super()._update(index, outcome)

        if outcome == 1:
            self.parameters[index][0] += 1
        else:
            self.parameters[index][1] += 1

    def decide(self):
        pre_mean = [beta[0] / (beta[0] + beta[1])
                    for beta in self.parameters]
        decision_index = np.argmax(pre_mean)

        return decision_index


# # visualisation of posterior distributions
# def plot_beta_pdf(ax, a, b):
#     x = np.linspace(0.01, 0.99, 99)
#     ax.plot(x, beta(a, b).pdf(x))


# turns = [1, 2, 3, 4, 100, 500, 1000]
# g = [GreedyBayesianBernoulli(priors, i, *machines) for i in turns]
# for g_turn in g:
#     g_turn.simulate()

# n = 6

# fig, ax = plt.subplots(3, figsize=(5, 5))
# a, b = g[n].parameters[0][0], g[n].parameters[0][1]
# plot_beta_pdf(ax[0], a, b,)
# a, b = g[n].parameters[1][0], g[n].parameters[1][1]
# plot_beta_pdf(ax[1], a, b,)
# a, b = g[n].parameters[2][0], g[n].parameters[2][1]
# plot_beta_pdf(ax[2], a, b,)

# plt.show()


"Begin debug"
turns = 10
obj1 = GreedyBayesianBernoulli(priors, turns, *machines)
obj2 = GreedyBayesianBernoulli(priors, turns, *machines)
obj1.simulate()
obj2.simulate()
print(obj2.parameters)

print(f"The sum of all parameters should be {turns} + 6 = {turns + 6} "
      f"but the sum here is {sum([sum(sublist) for sublist in obj2.parameters])}")