from decision import *
from example_code_helper import *
import matplotlib.pyplot as plt


bernoulli_machines = [bernoulli_machine(0.55)] + [bernoulli_machine(0.6)]


r = simulate(choose_random, 1000, 'series', *bernoulli_machines)
explore_explot_array = [simulate(explore_exploit_wrapper(0.3*i), 1000, 
                       'series', *bernoulli_machines) 
                        for i in range(1, 4)]


plt.figure()
plt.plot(r, label='random')
for i, array in enumerate(explore_explot_array):
    plt.plot(array, label=f'explore_exploit: p={(i+1)*0.1}')
plt.xlabel('Time')
plt.ylabel('Wealth')
plt.legend()
plt.show()