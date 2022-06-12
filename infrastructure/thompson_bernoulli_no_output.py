"""
Suppose we are given 3 machines with payouts following Bern(0.33), Bern(0.55)
and Bern(0.6) respectively and we play 1000 rounds in total.
We employ the method of Thompson sampling.
"""

from infrastructure import *
from copy import deepcopy
import matplotlib.pyplot as plt

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
        self.post_parameters = deepcopy(prior_parameters)
        self.post_parameters_history = [deepcopy(prior_parameters)]
        
    # overwrite decide
    def decide(self):
        
        # beta posterior distribution parameters
        post_samples = []
        for a, b in self.post_parameters:
            post_samples.append(np.random.beta(a, b))
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
# priors = [[1,1] for i in range(len(machines))]

# g = ThompsonSamplingBernoulli(priors, 1000, *machines)
# historical_regret = g.simulate("obj").historical_regret

# fig, (ax1, ax2) = plt.subplots(1, 2)
# historical_regret = np.array(g.historical_regret)
# # bounds = g[-1].bounds
# ax1.plot(historical_regret)
# # ax1.plot(bounds)
# ax2.plot(historical_regret)
# # ax2.plot(bounds)
# ax2.set_xscale('log')
# # plt.show()
# print(g.regret)