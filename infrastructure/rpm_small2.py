from infrastructure import *
from copy import copy, deepcopy
import random
import dill
import bz2

"""
Suppose we are given 3 machines with payouts following Bern(0.33), Bern(0.55)
and Bern(0.6) respectively and we play 1000 rounds in total.
We employ the method of randomised probability matching. See https://arxiv.org/pdf/1709.03162.pdf. 
"""

# initialise machines
machines = [bernoulli_machine(i) for i in [0.5, 0.52]]

class RPMBernoulli(Game):
    """
    Randomised probability matching approach with beta priors.  
        
    Parameters
    ----------
        prior_parameters : list
            Nested list containing the beta prior parameters for each machine
            e.g. [[1,1], [1,1], [1,1]]
        m : int
            Number of Monte Carlo samples
        turns : int
            Number of turns to be played
        *machines : list
            List of Machines
    """
    
    # add prior parameters, number of Monte Carlo samples
    def __init__(self, prior_parameters, m, turns, *machines):
        """
        Constructs attributes. 
        
        Attributes
        ----------
            post_parameters : list
                Nested list containing the beta posterior parameters for each machine after each turn
            post_parameters_history : list
                Stores the beta posterior parameters for each machine and each turn
            m : int
                Number of Monte Carlo samples
        """
        super().__init__(turns, *machines)
        self.post_parameters = deepcopy(prior_parameters)
        # self.post_parameters_history = [deepcopy(prior_parameters)]
        self.m = m
        
    # overwrite decide
    def decide(self):
        
        # Monte Carlo sampling
        mc_samples = np.array([np.random.beta(a, b, self.m) for a, b in self.post_parameters])
        
        # Approximate the conditional probability integral using indicator functions
        indicator_mat = np.zeros((len(machines), self.m), dtype=int)
        for j in range(self.m):
            i = np.argmax(mc_samples[:,j])
            indicator_mat[i, j] = 1
        weights = np.sum(indicator_mat, axis=1)
        
        # Randomised probability matching
        return random.choices(range(len(machines)), weights=weights)[0]
    
    # overwrite _update to store posterior parameters
    def _update(self, index, outcome):
        super()._update(index, outcome)
        
        # update the posterior distribution at given index
        if outcome == 1:
            self.post_parameters[index][0] += 1
        else:
            self.post_parameters[index][1] += 1
        # self.post_parameters_history.append(deepcopy(self.post_parameters))

# α, β parameters for beta prior
# α = β = 1 gives uniform distribution
priors = [[1,1] for i in range(len(machines))]
rpm_games = [RPMBernoulli(priors, 100, 5000, *machines).simulate("obj") for i in range(100)]
with bz2.BZ2File('rpm_small2.bz2', 'wb') as handle:
    dill.dump(rpm_games, handle)
