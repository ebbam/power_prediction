# Import packages
import numpy as np
import random as random
import matplotlib.pyplot as plt
import scipy
import pickle
from tqdm.notebook import tqdm

# load in better objects

from better import better, run_market, plot_returns
def evaluate_markets(market_record):
    mp = np.array(market_record['price_history'])
    tp = np.array(market_record['gen_el'])
    mse = np.mean((mp-tp)**2)
    # cross-correlation isn't normalised, so this isn't a value between -1 and 1
    max_cor = np.max(scipy.signal.correlate(mp, tp, mode='full')[-len(tp):len(tp)]) # only taking the correlations when we have at least 50% overlap

    return mse, max_cor


def run_trial(budget_total,w, N, N_whales, av_budget, parameters,mv):
    whale_budget = budget_total * w
    remaining_budget_av = av_budget*(1-w)*N/(N-N_whales)
    budget_samples = np.random.exponential(remaining_budget_av,N-N_whales)

    non_whale_betters = [better(budget=budget_samples[i]) for i in range(N-N_whales)]
    whale_betters = [better(budget=whale_budget, whale=True, market_valuation=mv) for _ in range(N_whales)]

    parameters.update({'betters': non_whale_betters + whale_betters})

    market_record = run_market(**parameters)
    mse, max_cor = evaluate_markets(market_record)
    return [w, mse, max_cor]

# set seed
np.random.seed(0)

# sampling budgets
N=  1000
N_whales = 1
av_budget = 500
budget_total = av_budget * N

w = 0.1

results = []

n_iter_ = 100

# Set initial input values to the betting market function
parameters = {'n_betters': N, # The number of betting agents
                #'el_outcome': 1, # Q: Ultimate election outcome - assuming we know this to begin with and it does not change over time...for now this is implemented as a random walk of the probability...but should this be 0 or 1 instead? '''
            't_election': 100, # Time until election takes place (ie. time horizon of betting)
            'initial_price': 0.5, # Initial market price (is this equivalent to probability of winning)
            'outcome_uncertainty': 0.1} # This is a measure of how uncertain the true outcome is - ie. the volatility of the random walk election probability

mv = 1

for w in tqdm(np.arange(0.1,1,0.1)):
    for _ in range(n_iter_):
        results.append(run_trial(budget_total,w, N, N_whales, av_budget, parameters,mv))

    r_arr = np.array(results)
    with open(f'whale_{mv}_{N}.pkl', 'wb') as f:
        pickle.dump(r_arr, f)

results = np.array(results)
# modify the simulation to multiprocess
fig, axs = plt.subplots(1,2,figsize=(10,6))
plt.subplots_adjust(wspace=0.4)
axs[0].scatter(x=results[:,0],y=results[:,1],s=3)
sns.lineplot(x=results[:,0],y=results[:,1],c='maroon',ax=axs[0])
axs[0].set_ylabel('MSE')
axs[0].set_xlabel(r'Proportion of budget allocated to whales, $\rho_w$')
plt.suptitle(f'Single whale with internal valuation of {mv}')
axs[1].scatter(x=results[:,0],y=results[:,2],s=3)
sns.lineplot(x=results[:,0],y=results[:,2],c='maroon',ax=axs[1])
axs[1].set_ylabel('Maximum lagged\n cross-correlation')
axs[1].set_xlabel(r'Proportion of budget allocated to whales, $\rho_w$')

plt.savefig(f'whales_{mv}_{N}.pdf', bbox_inches='tight')
