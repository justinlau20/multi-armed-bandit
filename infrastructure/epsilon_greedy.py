from infrastructure import *
import numpy as np
import matplotlib.pyplot as plt
from random import choices

machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]]

class epsilon_greedy(Game):
    def __init__(self, turns, *machines, epsilon=0.5):
        super().__init__(turns, *machines)
        self.epsilon = epsilon

    def decide(self):
        if self.next_turn <= self.machine_count:
            return self.next_turn % self.machine_count
        exploit = choices([0,1],[self.epsilon,1-self.epsilon])[0]
        if exploit == 1: 
            return np.argmax(self.means)
        return choices(range(self.machine_count))[0]

    def _update(self,index,outcome):
        super()._update(index,outcome)

obj5 = epsilon_greedy(100, *machines, epsilon = 0.5)
print(obj5.simulate()/100)

