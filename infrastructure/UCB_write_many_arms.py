from infrastructure import *
import numpy as np
from math import log,sqrt
import matplotlib.pyplot as plt
import dill
import bz2
from running_stats import RunningStats

machines = [bernoulli_machine(i) for i in [0.2 + j * 0.05 for j in range(10)]]

class UCB1_tuned(Game):
    def __init__(self, turns, *machines):
        super().__init__(turns,*machines)
        self.UCB_indices = [0]*self.machine_count
        self.UCB_for_var = [0]*self.machine_count
        self.rs = [RunningStats() for i in range(self.machine_count)]

    def _update(self, index, outcome):
        super()._update(index,outcome)
        if self.next_turn >= self.machine_count + 1:
            self.rs[index].push(outcome)
            self.UCB_for_var = [self.rs[i].variance() +
                            sqrt(2*log(self.next_turn)/len(history)) for i,history in enumerate(self.history)]
            
            self.UCB_indices =  [self.means[i] + sqrt(log(self.next_turn)/len(self.history[i])*
                                min(1/4,self.UCB_for_var[i])) for i in range(self.machine_count)]


    def decide(self):
        if self.next_turn <= self.machine_count:
            return self.next_turn % self.machine_count
        return np.argmax(self.UCB_indices)


UCB_games = [UCB1_tuned(5000, *machines).simulate("obj") for i in range(100)]
with bz2.BZ2File('UCB_many_arm.bz2', 'wb') as handle:
    dill.dump(UCB_games, handle)

