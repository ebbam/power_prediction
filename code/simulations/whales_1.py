# Import packages
import numpy as np
import random as random
import matplotlib.pyplot as plt
import scipy
import pickle
from tqdm.notebook import tqdm
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

# load in bettor objects

from bettor import bettor, run_market, plot_returns
import statsmodels.api as sm

def best_lag_regression(market, outcome, max_lag=30, criterion='pval'):
    """
    Find the lag l that gives the best predictive regression:
        eta_t = beta * m_{t-l} + error_t

    Parameters
    ----------
    market : array-like
        Market price time series (m_t)
    outcome : array-like
        True outcome time series (η_t)
    max_lag : int, optional
        Maximum lag (in time steps) to test
    criterion : {'pval', 'r2'}, optional
        Select best model based on lowest p-value or highest R^2.

    Returns
    -------
    results : dict
        Dictionary containing best lag, beta coefficient, p-value, R^2, and full summary.
    """
    market = np.asarray(market)
    outcome = np.asarray(outcome)
    n = len(outcome)

    best = {'lag': None, 'beta': None, 'pval': np.inf, 'r2': -np.inf, 'model': None}

    for lag in range(1, max_lag + 1):
        y = outcome[lag:]           # η_t
        x = market[:-lag]           # m_{t-l}
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        beta_pval = model.pvalues[1]
        beta_coef = model.params[1]
        r2 = model.rsquared

        if criterion == 'pval':
            if beta_pval < best['pval']:
                best.update({'lag': lag, 'beta': beta_coef, 'pval': beta_pval, 'r2': r2, 'model': model})
        elif criterion == 'r2':
            if r2 > best['r2']:
                best.update({'lag': lag, 'beta': beta_coef, 'pval': beta_pval, 'r2': r2, 'model': model})

    return best

def evaluate_markets(market_record, maxlag=5):
    mp = np.array(market_record['price_history'])
    tp = np.array(market_record['gen_el'])

    # take first differences
    tp_diff = np.diff(tp)
    mp_diff = np.diff(mp)
    ts_data = np.column_stack([mp_diff, tp_diff])

    mse = np.mean((mp-tp)**2)

    best_lag = best_lag_regression(mp, tp)

    # res = grangercausalitytests(ts_data, maxlag=maxlag, addconst=True, verbose=False)

    # # pull p-values from one of the tests at each lag (e.g., ssr_ftest)
    # pvals = {lag: res[lag][0]['ssr_ftest'][1] for lag in res}

    # # decide significance at, say, 5%
    # sig_lags = [lag for lag, p in pvals.items() if p < 0.05]
    # # best_lag = min(sig_lags) if sig_lags else None

    # # get lag with lowest p-value
    # if pvals:
    #     best_lag = min(pvals, key=pvals.get)
    # else:
    #     best_lag = None

    return mse, best_lag

def run_trial(budget_total,w, N, N_whales, av_budget, parameters,mv):
    whale_budget = budget_total * w
    remaining_budget_av = av_budget*(1-w)*N/(N-N_whales)
    budget_samples = np.random.exponential(remaining_budget_av,N-N_whales)

    non_whale_bettors = [bettor(budget=budget_samples[i], expertise=0.95) for i in range(N-N_whales)]
    whale_bettors = [bettor(budget=whale_budget, whale=True, market_valuation=mv) for _ in range(N_whales)]

    parameters.update({'bettors': non_whale_bettors + whale_bettors})

    market_record = run_market(**parameters)
    mse, min_lag_GC = evaluate_markets(market_record)
    return [w, mse, min_lag_GC]

# set seed
np.random.seed(0)

# sampling budgets
N=  100
N_whales = 1
av_budget = 100
budget_total = av_budget * N

w = 0.1

results = []

n_iter_ = 100

# Set initial input values to the betting market function
parameters = {'n_bettors': N, # The number of betting agents
                #'el_outcome': 1, # Q: Ultimate election outcome - assuming we know this to begin with and it does not change over time...for now this is implemented as a random walk of the probability...but should this be 0 or 1 instead? '''
            't_election': 100, # Time until election takes place (ie. time horizon of betting)
            'initial_price': 0.5, # Initial market price (is this equivalent to probability of winning)
            'outcome_uncertainty': 0.1,# This is a measure of how uncertain the true outcome is - ie. the volatility of the random walk election probability
            'market_lambda': 1/50} # this is the step size for the market price update - smaller values mean smaller changes in price

mv = 0.6

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
axs[1].set_ylabel('Most significant lag for prediction')
axs[1].set_xlabel(r'Proportion of budget allocated to whales, $\rho_w$')

plt.savefig(f'whales_{mv}_{N}.pdf', bbox_inches='tight')
