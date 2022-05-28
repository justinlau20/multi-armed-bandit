import numpy as np
from random import choices, randint, random

# class Strategy:
#     """A class to represent a strategy.

#     Attributes
#     ----------
#     game : Game 
#         a Game object representing the state of the game

#     Methods:
#     --------
#     decide:
#         returns the index of the machine chosen
#     """
#     def __init__(self, game):
#         self.game = game

#     def decide(self):
#         raise NotImplementedError


class Machine:
    def __init__(self, func, mean=None, var=None):
        self.func = func
        self.mean, self.var = mean, var

    def spin(self):
        return self.func()


class Game:
    """A class to represent the game.

    Attributes
    ----------
    turns : int
        total number of turns

    next_turn : int
        next turn count

    history : array
        an array of arrays, with the nth array storing the outcomes from
        machine n

    machines : array
        an array of functions, each returning the output of one machine
    """

    def __init__(self, turns, *machines):
        self.machine_count = len(machines)
        self.history = [[] for i in range(self.machine_count)]
        self.wealth_series = np.array([0])
        self.turns = turns
        self.next_turn = 1
        self.machines = machines

    def _update_history(self, index, outcome):
        self.history[index].append(outcome)

    def _step(self, strategy):
        decision = strategy(self)
        outcome = self.machines[decision].spin()
        self._update_history(decision, outcome)
        self.wealth_series = np.append(self.wealth_series, 
                                       self.wealth_series[-1] + outcome)
        self.next_turn += 1


def simulate(strategy, turns, output='outcome', *machines):
    game = Game(turns, *machines)
    while game.next_turn <= game.turns:
        game._step(strategy)
    if output == 'series':
        return game.wealth_series
    if output == 'outcome':
        return game.wealth_series[-1]
    if output == 'obj':
        return game
    raise Exception("Incorrect input type.")



def bernoulli_machine(p):
    """
    Returns a function representing a machine which
    outputs 1 with probability p and 0 otherwiswe.
    """
    return Machine(lambda: choices([0, 1], [1-p, p])[0], p, p*(1-p))


example_game = Game(100, bernoulli_machine(0.2), bernoulli_machine(0.8))


def choose_random(game):
    return randint(0, game.machine_count-1)


def half_explore(game):
    while game.next_turn <= game.turns//2:
        return game.next_turn % game.machine_count
    mean_list = list(map(lambda l: sum(l)/len(l), game.history))
    index_max = max(range(len(mean_list)), key=mean_list.__getitem__)
    return index_max


