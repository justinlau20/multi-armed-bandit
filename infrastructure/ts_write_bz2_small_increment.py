"""
Suppose we are given 3 machines with payouts following Bern(0.33), Bern(0.55)
and Bern(0.6) respectively and we play 1000 rounds in total.
We employ the method of Thompson sampling.
"""

from infrastructure import *
from copy import deepcopy
import matplotlib.pyplot as plt
import dill
import bz2

# initialise machines
machines = [bernoulli_machine(i) for i in [0.01]*2+[0.02]]

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


priors = [[1,1] for i in range(len(machines))]
TS_games = [ThompsonSamplingBernoulli(priors, 5000, *machines).simulate("obj") for i in range(100)]
with bz2.BZ2File('TS_small_increment.bz2', 'wb') as handle:
    dill.dump(TS_games, handle)