from WLLN import WLLNExampleScenario, WLLN
from UCB import UCB_bernoulli
from thompson_bernoulli_no_output import ThompsonSamplingBernoulli
from infrastructure import *
from math import sqrt
from numpy import log
import numpy as np
import matplotlib.pyplot as plt


machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6, 0.55, 0.55, 0.55]]
def ucb(turns):
    alpha = 2
    g = UCB_bernoulli(turns, alpha, *machines).simulate('obj')
    # print(f"The regret of simulating the situation with total turns T"
    #         f"={turns} is {g.regret}")
    return g

def wlln(turns):
    g = WLLN(turns, 10, 1.96, *machines).simulate('obj')
    # print(f"The regret of simulating the situation with total turns T"
    #         f"={turns} is {g.regret}")
    
    return g

priors = [[1,1] for i in range(len(machines))]
def thompson(turns):
    g = ThompsonSamplingBernoulli(priors, turns, *machines).simulate("obj")
    return g


turns = 1000
trials = 100
funcs = [ucb, wlln, thompson]
outputs = [[] for func in funcs]
for i in range(trials):
    for i, func in enumerate(funcs):
        outputs[i].append(func(turns).regret)
for i in range(len(funcs)):
    print(f'The mean regret of {funcs[i].__name__} over {trials} trials,'
          f' each having turn count {turns} is {np.mean(outputs[i])}')
# print(outputs[2])
# print(f'The mean regret of wlln over {trials} trials, each having turn count {turns} is {np.mean(wlln_outs)}')