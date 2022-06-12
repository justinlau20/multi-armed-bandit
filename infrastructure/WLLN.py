from telnetlib import GA
from infrastructure import *
from math import sqrt
from numpy import log
import numpy as np
import matplotlib.pyplot as plt


machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]]


class WLLN(Game):
    def __init__(self, turns, explore_count, z, *machines):
        super().__init__(turns, *machines)
        self.explore_count = explore_count
        self.sds = [0] * self.machine_count
        self.scores = [0] * self.machine_count
        self.times_chosen = [0] * self.machine_count
        self.z = z

    def _update(self, index, outcome):
        super()._update(index, outcome)
        self.sds[index] = np.std(self.history[index])
        self.times_chosen[index] += 1
        self.scores[index] = self.means[index] + self.z * self.sds[index] / \
                            sqrt(self.times_chosen[index])

    def decide(self):
        if self.next_turn <= self.machine_count * self.explore_count:
            return self.next_turn % self.machine_count
        return np.argmax(self.scores)


class WLLNExampleScenario(WLLN):
    def __init__(self, turns):
        super().__init__(turns, 10, 1.96, *machines)


turns = [10**i for i in range(1, 5)]
g = [WLLNExampleScenario(i) for i in turns]
for g_turn in g:
    g_turn.simulate()


for i, obj in enumerate(g):
    print(f"The regret of simulating the situation with total turns T"
          f"={turns[i]} is {obj.regret}")

# print(g[-1].del_i)
fig, (ax1, ax2) = plt.subplots(1, 2)
historical_regret = np.array(g[-1].historical_regret)
# bounds = g[-1].bounds
ax1.plot(historical_regret)
# ax1.plot(bounds)
ax2.plot(historical_regret)
# ax2.plot(bounds)
ax2.set_xscale('log')

plt.show()