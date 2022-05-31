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
    """

    def __init__(self, turns, *machines):
        self.machine_count = len(machines)
        self.history = [[] for i in range(self.machine_count)]
        self.turns = turns
        self.next_turn = 1
        self.machines = machines
        self.means = [0]*self.machine_count
        self.wealth = 0

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

    def _step(self):
        """
        Progresses the game by one time step.
        """
        decision = self.decide()
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
        """
        while self.next_turn <= self.turns:
            self._step()
        if output == 'outcome':
            return self.wealth
        if output == 'obj':
            return self
        raise Exception("Incorrect input type.")

