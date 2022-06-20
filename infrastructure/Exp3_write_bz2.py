from infrastructure import *
from math import sqrt, log, exp
import dill
import bz2

# machines = [bernoulli_machine(i) for i in [0.2, 0.4, 0.6]]

class RiggedGame(Game):
    """A class to represent the game, including the state of the game
    and also the strategy chosen.

    Attributes
    ----------
    machine_count:
        total number of machines
    history:
        a tuple of tuples; the ith of which represents the history of machine i
    turns:
        total number of turns
    next_turn:
        turn number of next turn
    machines:
        a tuple containing the machines of the game
    means:
        a tuple of mean outcomes for each machine
    wealth:
        self explanatory
    decision_history:
        a list of the decisions made for all turns (indicies of machines)
    """

    def __init__(self, turns, *machines):
        super().__init__(turns, *machines)
        self.Gvec = np.array([0]*self.machine_count)
        self.gamma = min([1,
                          sqrt((self.machine_count*log(self.machine_count)/((exp(1)-1)*self.turns)))])
        self.en = self.gamma/self.machine_count
        self.Phatvec = np.array([])
        self.PhatvecHistory = []


    def decide(self):
        if self.next_turn != 1:
            currentrewardvec = np.array([0] * self.machine_count)
            lastmachineindex = self.decision_history[-1]
            currentrewardvec[lastmachineindex] = self.history[lastmachineindex][-1] / self.PhatvecHistory[-1]
            self.Gvec = self.Gvec + currentrewardvec
        Pvec = np.exp(self.en * self.Gvec) / np.sum(np.exp(self.en * self.Gvec))
        Phatvec = (1-self.gamma)*Pvec + (self.gamma/self.machine_count)
        nextplaychoice = choices([i for i in range(self.machine_count)], Phatvec)[0]
        nextplayprobability = Phatvec[nextplaychoice]
        self.PhatvecHistory.append(nextplayprobability)
        return nextplaychoice
    


# set up the machines
machines = [bernoulli_machine(i) for i in [0.33, 0.55, 0.6]]


# simulate once
# g = RiggedGame(100, *machines)
# print(f"The wealth and regret of simulating the scenario once is"
#       f" {g.simulate('all')}.")

Exp3_games = [RiggedGame(5000, *machines).simulate("obj") for i in range(100)]
with bz2.BZ2File('Exp3_n100_T5000.bz2', 'wb') as handle:
    dill.dump(Exp3_games, handle)