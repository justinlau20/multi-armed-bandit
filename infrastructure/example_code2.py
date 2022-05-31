"""
Suppose we are given 2 machines with payouts following N(0, 1), and N(1, 2)
respectively and we play 50 rounds in total.

Here, a strategy which explores during the first p proportion of the rounds
and exploits in the remaining rounds is implemented.
"""


import numpy as np
from infrastructure import *


# set up machines
machines = [normal_machine(i, j**0.5) for i, j in [[0, 1], [1, 2]]]


# overwrite __init__ so p can be stored, since the
# strategy now depends on p
class ExploreExploit(Game):
    def __init__(self, p, turns, *machines):
        super().__init__(turns, *machines)
        self.p = p

    def decide(self):
        while self.next_turn <= self.turns * self.p:
            return self.next_turn % self.machine_count
        index_max = max(range(len(self.means)), key=self.means.__getitem__)
        return index_max


p, turns, times = 0.5, 50, 1000
# simulate once
g = ExploreExploit(p, turns, *machines)
print(f"The output of simulating the scenario once is {g.simulate()}.")


def simulate_n_times(n, p, obj, turns, *machines):
    return [obj(p, turns, *machines).simulate() for i in range(n)]


# simulate 1000 times
output = simulate_n_times(times, 0.5, ExploreExploit, 50, *machines)
mean = sum(output) / times
print(f"The mean outcome of simulating the scenario {times} times is {mean}.")
