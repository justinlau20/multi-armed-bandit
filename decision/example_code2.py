from decision import *
import matplotlib.pyplot as plt
from example_code_helper import *
import numpy as np


bernoulli_machines = [bernoulli_machine(0.4)] + [bernoulli_machine(0.6)]
n = 500
proportions = np.linspace(0.1, 1, 10, False)
outcomes = [[] for i in range(len(proportions))]

for i in range(n):
    for index, p in enumerate(proportions):
        outcomes[index].append(simulate(explore_exploit_wrapper(p), 100, 'outcome', *bernoulli_machines))


def mean(l):
    return sum(l)/len(l)

mean_outcomes = list(map(mean, outcomes))
for proportion, outcome in zip(proportions, mean_outcomes):
    print(f'Mean outcome for proportion {proportion} is {outcome}')
