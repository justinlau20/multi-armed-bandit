from infrastructure import *
from copy import deepcopy
import random

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
        self.post_parameters = prior_parameters
        self.post_parameters_history = [deepcopy(prior_parameters)]
        self.m = m
        
    # overwrite decide
    def decide(self):
        
        # Monte Carlo sampling
        mc_samples = np.array([np.random.beta(a, b, self.m) for a, b in self.post_parameters])
        
        # Approximate the conditional probability integral using indicator functions
        indicator_mat = np.zeros((self.machine_count, self.m), dtype=int)
        for j in range(self.m):
            i = np.argmax(mc_samples[:,j])
            indicator_mat[i, j] = 1
        weights = np.sum(indicator_mat, axis=1)
        
        # Randomised probability matching
        return random.choices(range(self.machine_count), weights=weights)[0]
    
    # overwrite _update to store posterior parameters
    def _update(self, index, outcome):
        super()._update(index, outcome)
        
        # update the posterior distribution at given index
        if outcome == 1:
            self.post_parameters[index][0] += 1
        else:
            self.post_parameters[index][1] += 1
        self.post_parameters_history.append(deepcopy(self.post_parameters))
