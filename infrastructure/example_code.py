"""
Suppose we are given 3 machines with payouts following Bern(0.2), Bern(0.4)
and Bern(0.6) respectively and we play 100 rounds in total.

Here, a strategy which picks the first machine every time is implemented.
"""


from infrastructure import *


# set up the machines
machines = [bernoulli_machine(i) for i in [0.2, 0.4, 0.6]]


# create a new class overwriting the decide method
class AlwaysPickFirst(Game):
    def decide(self):
        return 0


# simulate once
g = AlwaysPickFirst(100, *machines)
print(f"The output of simulating the scenario once is {g.simulate()}.")


def simulate_n_times(n, obj, turns, *machines):
    return [obj(turns, *machines).simulate() for i in range(n)]


# simulate 1000 times
k = 1000
output = simulate_n_times(k, AlwaysPickFirst, 100, *machines)
mean = sum(output) / k
print(f"The mean outcome of simulating the scenario {k} times is {mean}.")
