from infrastructure import *
from RPM import RPMBernoulli
from TS import ThompsonSamplingBernoulli
import matplotlib.pyplot as plt
import dill
import bz2

# with bz2.BZ2File('ts_n100_T5000.bz2', 'wb') as handle:
#     dill.dump(ts_games, handle)
# with bz2.BZ2File('rpm_n100_T5000.bz2', 'wb') as handle:
#     dill.dump(rpm_games, handle)

with bz2.open('ts_n100_T5000.bz2', 'rb') as handle:
    ts_games2 = dill.load(handle)
with bz2.open('rpm_n100_T5000.bz2', 'rb') as handle:
    rpm_games2 = dill.load(handle)

n = 100     #no of simulations
T = 5000    #horizon

ts_regret = np.array([ts_game.historical_regret for ts_game in ts_games2])
rpm_regret = np.array([rpm_game.historical_regret for rpm_game in rpm_games2])

# compare cumulative regret
fig, ax = plt.subplots()
ax.plot(range(T+1), np.mean(ts_regret, axis=0), color = 'red')
ax.plot(range(T+1), np.mean(rpm_regret, axis=0), color = 'black')
ax.legend(['Thompson sampling', 'Randomised probability matching'])

ts_regret_u = np.percentile(ts_regret, 97.5, axis=0)
ts_regret_b = np.percentile(ts_regret, 2.5, axis=0)
rpm_regret_u = np.percentile(rpm_regret, 97.5, axis=0)
rpm_regret_b = np.percentile(rpm_regret, 2.5, axis=0)

ax.fill_between(range(T+1), ts_regret_b, ts_regret_u, color='red', alpha=0.5)
ax.fill_between(range(T+1), rpm_regret_b, rpm_regret_u, color='gray', alpha=0.5)

ax.set_title(f"Averaged cumulative regret over {n} simulations \n with a 95% confidence interval")
ax.set_xlabel("Horizon")
ax.set_ylabel("Cumulative regret")

# plt.savefig("fig1.pdf")