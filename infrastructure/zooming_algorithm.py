import numpy as np
import abc
import matplotlib.pyplot as plt
from scipy.stats import pareto

class Algorithm(metaclass=abc.ABCMeta):  # abstract class of `Algorithm`
    def __init__(self, delta, T, c):
        self.delta = delta
        self.T = T
        self.c = c
        self.pulled_idx = None
        self.active_arms = None
        self.mu = None
        self.n = None
        self.r = None

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def output(self):
        pass

    @abc.abstractmethod
    def observe(self, t, y):
        pass

    def get_uncovered(self):
        covered = [[arm - r, arm + r] for arm, r in zip(self.active_arms, self.r)]
        if covered == []:
            return [0, 1]
        covered.sort(key=lambda x: x[0])
        low = 0
        for interval in covered:
            if interval[0] <= low:
                low = max(low, interval[1])
                if low >= 1:
                    return None
            else:
                return [low, interval[0]]
        return [low, 1]


class Zooming(Algorithm):
    def __init__(self, delta, T, c, nu):
        super().__init__(delta, T, c)
        self.nu = nu

    def initialize(self):
        self.active_arms = []
        self.mu = []
        self.n = []
        self.r = []

    def output(self):
        uncovered = self.get_uncovered()
        if uncovered is None:
            score = [mu + 2 * r for mu, r in zip(self.mu, self.r)]
            self.pulled_idx = np.argmax(score)
        else:
            new_arm = np.random.uniform(*uncovered)
            self.active_arms.append(new_arm)
            self.mu.append(0)
            self.n.append(0)
            self.r.append(0)
            self.pulled_idx = len(self.active_arms) - 1
        return self.pulled_idx

    def observe(self, t, y):
        idx = self.pulled_idx
        self.mu[idx] = (self.mu[idx] * self.n[idx] + y) / (self.n[idx] + 1)
        self.n[idx] += 1
        for i, n in enumerate(self.n):
            self.r[i] = self.c * self.nu * np.power(t, 1 / 3) / np.sqrt(n)

def simulate(algorithm, a, alpha, T, trials):
    cum_regret = np.zeros((len(algorithm), T + 1))
    for trial in range(trials):
        inst_regret = np.zeros((len(algorithm), T + 1))
        for alg in algorithm:
            alg.initialize()

        for t in range(1, T + 1):
            for i, alg in enumerate(algorithm):
                idx = alg.output()
                arm = alg.active_arms[idx]
                inst_regret[i, t] = min(abs(arm - 0.4), abs(arm - 0.8))
                y = a - min(abs(arm - 0.4), abs(arm - 0.8)) + pareto.rvs(alpha) - alpha / (alpha - 1)
                alg.observe(t, y)

        cum_regret += np.cumsum(inst_regret, axis=-1)
    return cum_regret/trials

def upper_bound(t, delta, multiplier, zooming_dim):
    return delta*t + multiplier*np.log(t)*((1/delta)**(1+zooming_dim))

def run_experiment(a):
    # configure parameters of experiments
    T = 10000 #pre: 2000
    trials = 500 #pre: 40
    delta = 0.1
    alpha = 3.1
    epsilon = 1

    # compute upper bounds for moments of different orders
    a_hat = max(abs(a), abs(a - 0.4))
    sigma_second = max(alpha / ((alpha - 1) ** 2 * (alpha - 2)), 1 / (36 * np.sqrt(2)))
    nu_second = max(a_hat ** 2 + sigma_second, np.power(12 * np.sqrt(2), -(1 + epsilon)))
    nu_third = a_hat ** 3 + 2 * alpha * (alpha + 1) / (
            (alpha - 1) ** 3 * (alpha - 2) * (alpha - 3)) + 3 * a_hat * sigma_second

    # simulate
    c_zooming = 0.01  # searched within {1, 0.1, 0.01} and `0.01` is the best choice
    algorithm = [Zooming(delta, T, c_zooming, nu_third)]
    cum_regret = simulate(algorithm, a, alpha, T, trials)

    # plot figure
    plt.figure(figsize=(7, 4))
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    names = [f'{alg.__class__.__name__}' for alg in algorithm]
    linestyles = ['-', '--', '-.']
    for result, name, linestyle in zip(cum_regret, names, linestyles):
        plt.plot(result, label=name, linewidth=2.0, linestyle=linestyle)
    time_list = np.arange(1, 10001, 1)
    regret_upper_bound = upper_bound(time_list, delta, 1, c_zooming)
    plt.plot(time_list, regret_upper_bound,label="Upper Bound")
    plt.legend(loc='upper left', frameon=True, fontsize=10)
    plt.xlabel('Time', labelpad=1, fontsize=15)
    plt.ylabel('Cumulative Regret', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(f'cum_regret_{a}.png', dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    run_experiment(a=0)