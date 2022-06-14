"""
Suppose we are given 3 machines with payouts following Bern(0.33), Bern(0.55)
and Bern(0.6) respectively and we play 1000 rounds in total.
We employ the method of Thompson sampling.
"""

from infrasturcture import *
from copy import deepcopy

# initialise machines
machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]]

class ThompsonSamplingBernoulli(Game):
    """
    Thompson sampling using beta prior.
    
    Parameters
    ----------
        prior_parameters : list
            Nested list containing the beta prior parameters for each machine
            e.g. [[1,1], [1,1], [1,1]]
        turns : int
            Number of turns to be played
        *machines : list
            List of Machines
    """
    
    # add posterior parameters
    def __init__(self, prior_parameters, turns, *machines):
        """
        Constructs attributes. 
        
        Attributes
        ----------
            post_parameters : list
                Nested list containing the beta posterior parameters for each machine after each turn
            post_parameters_history : list
                Stores the beta posterior parameters for each machine and each turn
        """
        super().__init__(turns, *machines)
        self.post_parameters = prior_parameters
        self.post_parameters_history = [deepcopy(prior_parameters)]
        
    # overwrite decide
    def decide(self):
        
        # beta posterior distribution parameters
        post_samples = [np.random.beta(a, b) for a, b in self.post_parameters]
        return np.argmax(post_samples)
    
    # overwrite _update to store posterior parameters
    def _update(self, index, outcome):
        super()._update(index, outcome)
        
        # update the posterior distribution at given index
        if outcome == 1:
            self.post_parameters[index][0] += 1
        else:
            self.post_parameters[index][1] += 1
        self.post_parameters_history.append(deepcopy(self.post_parameters))

# α, β parameters for beta prior
# α = β = 1 gives uniform distribution
priors = [[1,1] for i in range(len(machines))]

g = ThompsonSamplingBernoulli(priors, 1000, *machines)
g.simulate()

# visualisation of posterior distributions
from scipy.stats import beta
import matplotlib.pyplot as plt

def plot_beta_pdf(ax, a, b):
    x = np.linspace(0.01, 0.99, 99)
    ax.plot(x, beta(a, b).pdf(x))

turns = [1,2,3,4,5,10,50,500,1000]
g.post_parameters_history
fig, ax = plt.subplots(len(turns)+1, figsize=(5,40))

# prior distributions
for a, b in g.post_parameters_history[0]:
    plot_beta_pdf(ax[0], a, b,)
    ax[0].legend([i+1 for i in range(len(machines))])
    ax[0].set_title("Prior distributions")

# posterior distributions at different turns
for n, turn in enumerate(turns):
    for index in range(len(machines)):
        a, b = g.post_parameters_history[turn][index]
        plot_beta_pdf(ax[n+1], a, b,)
        ax[n+1].legend([i+1 for i in range(len(machines))])
    ax[n+1].set_title(f"Posterior distributions after {turn} turn(s)")
