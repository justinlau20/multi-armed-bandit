from infrastructure import *
from math import sqrt, log
import numpy as np
import matplotlib.pyplot as plt

machines = [bernoulli_machine(p) for p in [0.1, 0.5, 0.78, 0.8, 0.05]]

class UCB1_normal(Game):
    def __init__(self, turns, *machines):
        super().__init__(turns, *machines)
        self.sum_of_squared_rewards = [0] * self.machine_count
        self.UCB_normal_indices = [0]* self.machine_count

    def decide(self):
        if self.next_turn <= 2* self.machine_count:
            return self.next_turn % self.machine_count
        for i in range(self.machine_count):
            if len(self.history[i]) < 8 *log(self.next_turn): return i
        return np.argmax(self.UCB_normal_indices)


    def _update(self, index, outcome):
        super()._update(index, outcome)
        self.sum_of_squared_rewards[index] += outcome**2
        if self.next_turn > 2* self.machine_count:
            self.UCB_normal_indices = [self.means[i] + sqrt(16* (s - len(self.history[i])*self.means[i]**2)* log(self.next_turn -1)/
                                        ((len(self.history[i]) -1)*len(self.history[i])))
                                        for i,s in enumerate(self.sum_of_squared_rewards)]



