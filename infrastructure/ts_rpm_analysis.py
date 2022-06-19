from infrastructure import *
from RPM import RPMBernoulli
from TS import ThompsonSamplingBernoulli
import matplotlib.pyplot as plt
import dill
import bz2
import numpy as np

# with bz2.BZ2File('ts_n100_T5000.bz2', 'wb') as handle:
#     dill.dump(ts_games, handle)
# with bz2.BZ2File('rpm_n100_T5000.bz2', 'wb') as handle:
#     dill.dump(rpm_games, handle)
files = ['ts_n100_T5000.bz2', 'rpm_n100_T5000.bz2', 'gbb_n100_T5000.bz2']
names = ['Thompson sampling', 'Randomised probability matching', 'Greedy Bayesian']
colors = ['red', 'black', 'green']
strats = []
for file in files:
    with bz2.open(file, 'rb') as handle:
        strats.append(dill.load(handle))

# with bz2.open('ts_n100_T5000.bz2', 'rb') as handle:
#     ts_games = dill.load(handle)
# with bz2.open('rpm_n100_T5000.bz2', 'rb') as handle:
#     rpm_games = dill.load(handle)
# with bz2.open('gbb_n100_T5000.bz2', 'rb') as handle:
#     gbb_games = dill.load(handle)


n = 100     # no of simulations
T = 5000    # horizon

# get regrets as arrays
hist_regret_arrs = [np.array([game.historical_regret for game in games]) for games in strats]

# # compare cumulative regret
# ts_regret = np.array([ts_game.historical_regret for ts_game in ts_games])
# rpm_regret = np.array([rpm_game.historical_regret for rpm_game in rpm_games])


fig, ax = plt.subplots()
for arr, col in zip(hist_regret_arrs, colors):
    ax.plot(range(T+1), np.mean(arr, axis=0), color=col)
ax.legend(names)

# # plot averaged cumulative regret with 95% confidence interval
# fig, ax = plt.subplots()
# ax.plot(range(T+1), np.mean(ts_regret, axis=0), color = 'red')
# ax.plot(range(T+1), np.mean(rpm_regret, axis=0), color = 'black')
# ax.legend(['Thompson sampling', 'Randomised probability matching'])

upper, lower = 97.5, 2.5
uppers = [np.percentile(hist_regret, upper, axis=0) for hist_regret in hist_regret_arrs]
lowers = [np.percentile(hist_regret, lower, axis=0) for hist_regret in hist_regret_arrs]
# ts_regret_u = np.percentile(ts_regret, 97.5, axis=0)
# ts_regret_b = np.percentile(ts_regret, 2.5, axis=0)
# rpm_regret_u = np.percentile(rpm_regret, 97.5, axis=0)
# rpm_regret_b = np.percentile(rpm_regret, 2.5, axis=0)


for b, u, col in zip(lowers, uppers, colors):
    ax.fill_between(range(T+1), b, u, alpha=0.5, color=col)

# ax.fill_between(range(T+1), ts_regret_b, ts_regret_u, color='red', alpha=0.5)
# ax.fill_between(range(T+1), rpm_regret_b, rpm_regret_u, color='gray', alpha=0.5)

ax.set_title(f"Averaged cumulative regret over {n} simulations \n with a 95% confidence interval")
ax.set_xlabel("Horizon")
ax.set_ylabel("Cumulative regret")

# plt.savefig("fig1.pdf")
plt.show()