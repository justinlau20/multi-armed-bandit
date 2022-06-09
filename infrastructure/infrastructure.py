from platform import machine
import numpy as np
from random import choices, randint, random


class Machine:
    """
    A class representing a machine.
    -------------------------------
    Attributes:
    func:
        the function which generates the random output of the machine
    mean:
        mean of the distribution of the output
    var:
        var of the distribution of the output
    
    Methods:
    --------
    spin:
        spin the machine by calling func, returning the output of the machine
    """
    def __init__(self, func, mean=None, var=None):
        self.func = func
        self.mean, self.var = mean, var

    def spin(self):
        return self.func()


def bernoulli_machine(p):
    """
    Returns a machine with Bernoulli payout.
    """
    return Machine(lambda: choices([0, 1], [1-p, p])[0], p, p*(1-p))


def normal_machine(mean, sd):
    """
    Returns a machine with normal payout.
    """
    return Machine(lambda: np.random.normal(mean, sd), mean, sd**2)


class Game:
    """A class to represent the game, including the state of the game
    and also the strategy chosen.

    Attributes
    ----------
    machine_count:
        total number of machines
    history:
        a tuple of tuples; the ith of which represents the history of machine i
    turns:
        total number of turns
    next_turn:
        turn number of next turn
    machines:
        a tuple containing the machines of the game
    means:
        a tuple of mean outcomes for each machine
    wealth:
        self explanatory
    decision_history:
        a list of the decisions made for all turns (indicies of machines)
    """

    def __init__(self, turns, *machines):
        self.machine_count = len(machines)
        self.history = [[] for i in range(self.machine_count)]
        self.turns = turns
        self.next_turn = 1
        self.machines = machines
        self.means = [0]*self.machine_count
        self.wealth = 0
        self.decision_history = []
        if all([m.mean is not None for m in machines]):
            self.regret = 0
            self.best_machine_mean = max([m.mean for m in machines])
        else:
            self.regret = None

    def _update(self, index, outcome):
        """
        Updates the attributes after each turn.
        Can be modified to include update new attributes
        (e.g. running variance).
        """
        self.history[index].append(outcome)
        l = len(self.history[index])
        self.means[index] = (self.means[index] * (l-1) + outcome)/l
        self.wealth += outcome
        if self.regret is not None:
            self.regret += self.best_machine_mean - self.machines[index].mean

    def _step(self):
        """
        Progresses the game by one time step.
        """
        decision = self.decide()
        self.decision_history.append(decision)
        outcome = self.machines[decision].spin()
        self._update(decision, outcome)
        self.next_turn += 1

    def decide(self):
        """
        Makes a decision based on current game stage (i.e. attributes of self)
        and returns the index of chosen machine.
        Overwrite this.
        """
        pass

    def simulate(self, output='outcome'):
        """
        Runs the simulation once. Returns the wealth at the end
        of the game by default. Can also return self for debugging/ other uses.

        Parameters:
        -----------
        output:
            'outcome' (default); returns final wealth
            'regret'; returns final regret
            'all'; returns all relevant attributes as a tuple
            'obj'; returns the final Game object
        """
        while self.next_turn <= self.turns:
            self._step()
        if output == 'outcome':
            return self.wealth

        if output == "regret":
            return self.regret

        if output == "all":
            return self.wealth, self.regret

        if output == 'obj':
            return self
        raise Exception("Incorrect input type.")

