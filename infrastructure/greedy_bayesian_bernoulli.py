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
from copy import copy, deepcopy
import random

# initialize machines and beta (1, 1) priors (uniform dist.)
machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]]  # noqa: F405
priors = [[1, 1] for i in range(len(machines))]


class GreedyBayesianBernoulli(Game):  # noqa: F405
    """Greedy Bayesian using beta prior."""

    def __init__(self, prior_parameters, threshold, ucb, turns, *machines):
        """
        We define additional attributes to implement this method.

        self.parameters     store the parameters of the beta distribution
                            at the current time step. This gets updated
                            to the posterior when an outcome is observed
                            and gets carried over as the prior for the
                            next time step.
        self.post_param     history of beta paramters for all time steps
        eters_history
        self.threshold      the probability that the algorithm 'explores'
                            instead of exploiting at each step. Pass in
                            0 to make the method purely greedy.
        self.ucb            the upper confidence percentile of the
                            parameter - calculated at each time step
                            from the prior
        """
        super().__init__(turns, *machines)  # inherit class attributes
        self.parameters = copy([copy(sublist) for sublist in prior_parameters])
        self.post_parameters_history = [deepcopy(prior_parameters)]
        self.threshold = threshold
        self.ucb = ucb

    def _update(self, index, outcome):  # need to overwrite update for Bayesian
        super()._update(index, outcome)

        # the following results follow from cojugate priors
        if outcome == 1:
            self.parameters[index][0] += 1
        else:
            self.parameters[index][1] += 1

        # update history
        self.post_parameters_history.append(deepcopy(self.parameters))

    def decide(self):  # the decision-making step based on the current model
        e = random.uniform(0, 1)  # used later for exploitation/exploration
        # pre_mean = [beta[0] / (beta[0] + beta[1])  # noqa: F841
        #             for beta in self.parameters]

        # compute the UCB for each machine given the user-input percentile
        pre_ucb = [beta.ppf(self.ucb, para[0], para[1], loc=0, scale=1)
                   for para in self.parameters]  # from scipy.stats

        # decide: exploit the best option (UCB)/explore another random machine
        if e > self.threshold:
            decision_index = np.argmax(pre_ucb)
        else:
            # can improve on code readability
            index = [i for i, _ in enumerate(machines)]
            index.pop(np.argmax(pre_ucb))
            decision_index = random.choice(index)

        return decision_index


# define a plotting template for a beta pdf
def plot_beta_pdf(ax, a, b):
    x = np.linspace(0.01, 0.99, 99)
    ax.plot(x, beta(a, b).pdf(x))


# simulate for 1000 turns
turns = 1000
# example in git repo passes a 0.02 chance of random exploration with
# exploitation determined by the 90% upper bound
g = GreedyBayesianBernoulli(priors, 0.02, 0.95, turns, *machines)
g.simulate()

# generate and fill out the plot
fig, ax = plt.subplots(len(machines) + 1, figsize=(5, 10))

for i in [10, 100, 1000]:
    for j in range(len(machines)):
        a, b = g.post_parameters_history[i][j]
        plot_beta_pdf(ax[j], a, b)

# add in a plot of the decision history - note that this is only implicative
# of regret
ax[-1].plot(g.decision_history, marker='.', markersize=2, linestyle="None")

plt.show()
