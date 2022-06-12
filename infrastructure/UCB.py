from infrastructure import *
from math import sqrt
from numpy import log
import numpy as np
import matplotlib.pyplot as plt


machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]] 


class UCB_bernoulli(Game):
    def __init__(self, turns, alpha, *machines):
        super().__init__(turns, *machines)
        self.alpha = alpha
        self.UCB_indices = [0]*self.machine_count
        true_means = [m.mean for m in machines]
        max_mean = max(true_means)
        self.del_i = np.sort(max_mean - np.array(true_means))
        # self.bounds = []

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
        # self.bounds.append(get_bound(self.del_i, self.next_turn - 1, self.alpha))


def get_bound(del_i, T, a):
    if T <= 2:
        return 0
    temp = []
    for d in del_i[1:]:
        temp.append((a+1)*d/(a-1) + 2*a*log(T)/d)
    return sum(temp)


# alpha = 2
# turns = [10**i for i in range(1, 6)]
# g = [UCB_bernoulli(i, alpha, *machines) for i in turns]
# for g_turn in g:
#     g_turn.simulate()


# for i, obj in enumerate(g):
#     print(f"The regret of simulating the situation with total turns T"
#           f"={turns[i]} is {obj.regret}")

# # print(g[-1].del_i)
# fig, (ax1, ax2) = plt.subplots(1, 2)
# historical_regret = np.array(g[-1].historical_regret)
# # bounds = g[-1].bounds
# ax1.plot(historical_regret)
# # ax1.plot(bounds)
# ax2.plot(historical_regret)
# # ax2.plot(bounds)
# ax2.set_xscale('log')

# plt.show()