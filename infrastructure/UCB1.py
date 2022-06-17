from infrastructure import *
import numpy as np
from math import log, sqrt
import matplotlib.pyplot as plt

machines = [bernoulli_machine(p) for p in [0.1, 0.5, 0.8, 0.05]]


class UCB1(Game):
    def __init__(self, turns, *machines):
        super(). __init__(turns, *machines)
        self.UCB1_indices = [0]*self.machine_count
        max_mean = max(self.means)
    def decide(self):
        if self.next_turn <= self.machine_count:
            return self.next_turn % self.machine_count
        return np.argmax(self.UCB1_indices)

    def _update(self, index, outcome):
        super()._update(index, outcome)
        true_means = [m.mean for m in machines]
        max_mean = max(true_means)
        if self.next_turn >= self.machine_count + 1:
            self.UCB1_indices = [self.means[i] + sqrt(2*log(self.next_turn)
                                 / len(self.history[i]))
                                 for i in range(self.machine_count)]


obj1 = UCB1(10000, *machines).historical_regret
