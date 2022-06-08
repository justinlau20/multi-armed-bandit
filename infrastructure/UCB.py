from infrastructure import *
from math import sqrt, log
import numpy as np

machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]] 


class UCB_bernoulli(Game):
    def __init__(self, turns, alpha, *machines):
        super().__init__(turns, *machines)
        self.alpha = alpha
        self.UCB_indices = [0]*self.machine_count

    def decide(self):
        if self.next_turn <= self.machine_count:
            return self.next_turn % self.machine_count
        return np.argmax(self.UCB_indices)

    def _update(self, index, outcome):
        super()._update(index, outcome)
        if self.next_turn >= self.machine_count + 1:
            self.UCB_indices = [m + sqrt(self.alpha * log(self.next_turn-1) /
                                (2 * len(self.history[i])))
                                for i, m in enumerate(self.means)]


alpha = 2
turns = [10**i for i in range(1, 7)]
g = [UCB_bernoulli(i, alpha, *machines) for i in turns]
for g_turn in g:
    g_turn.simulate()


for i, obj in enumerate(g):
    print(f"The regret of simulating the situation with total turns T"
          f"={turns[i]} is {obj.regret}")