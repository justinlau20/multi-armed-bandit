from WLLN import WLLNExampleScenario, WLLN
from UCB import UCB_bernoulli
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

turns = 1000
trials = 100
ucb_outs = []
wlln_outs = []
for i in range(trials):
    ucb_outs.append(ucb(turns).regret)
    wlln_outs.append(wlln(turns).regret)
print(f'The mean regret of ucb over {trials} trials, each having turn count {turns} is {np.mean(ucb_outs)}')
print(f'The mean regret of wlln over {trials} trials, each having turn count {turns} is {np.mean(wlln_outs)}')