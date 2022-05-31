"""
Suppose now we want to compare different strategies in the same scenario
(i.e. same turn count and machines). We can construct a class encoding the
scenario and have different strategies inherit from it.

Suppose also that we want to make decisions based on other statistics
(e.g. running variance).
By default, the Game class only stores the history of the game and
the historical mean of each machine. We can overwrite the init and _update
methods so the class stores more statistics. 

For example, we may use the scenario from example_code
(i.e. 100 turns, Bernoulli 0.2, 0.4, 0.6)
and compare the following strategies:
    - spend first half of turns picking machines uniformly,
    then pick the machine with highest variance

    - spend first half of turns picking machines uniformly,
    then pick the machine with lowest variance
"""


from infrastructure import *

turns = 100
machines = [bernoulli_machine(i) for i in [0.2, 0.4, 0.6]]


# create a class representing the scenario
class BernoilliScenario(Game):
    def __init__(self):
        super().__init__(turns, *machines)


# modify init, _update to store variance
# also overwrite decide
class HighestVariance(BernoilliScenario):
    def __init__(self):
        super().__init__()
        self.vars = [0] * self.machine_count

    def _update(self, index, outcome):
        super()._update(index, outcome)
        self.vars[index] = np.var(self.history[index])

    def decide(self):
        while self.next_turn <= self.turns * 0.5:
            return self.next_turn % self.machine_count
        index_max = max(range(len(self.vars)), key=self.vars.__getitem__)
        # print(index_max, self.vars)
        return index_max


class LowestVariance(BernoilliScenario):
    def __init__(self):
        super().__init__()
        self.vars = [0] * self.machine_count

    def _update(self, index, outcome):
        super()._update(index, outcome)
        self.vars[index] = np.var(self.history[index])

    def decide(self):
        while self.next_turn <= self.turns * 0.5:
            return self.next_turn % self.machine_count
        index_min = min(range(len(self.vars)), key=self.vars.__getitem__)
        # print(index_max, self.vars)
        return index_min


times = 100
hv = [HighestVariance().simulate() for i in range(times)]
lv = [LowestVariance().simulate() for i in range(times)]

print(f"The mean outcome of simulating the scenario {times} times "
      f"picking highest variance is {np.mean(hv)}.")

print(f"The mean outcome of simulating the scenario {times} times "
      f"picking lowest variance is {np.mean(lv)}.")
