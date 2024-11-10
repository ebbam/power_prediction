import numpy as np
import random as random
import matplotlib.pyplot as plt

## Defining classes

class better:
    # Wordy but this allows us to more easily change the default values of betters we initialise
    def __init__(self, budget = np.random.uniform(100,1000), market_valuation = np.random.normal(0.5,0.001),
                 n_contracts = 0, risk_av = np.random.uniform(0,1), stubbornness = 0,#np.random.uniform(0,1), 
                 expertise = np.random.normal(0.9, 0.01), 
                 bias = 0,
                 whale=False):
        self.budget = budget # Their personal contract valuation - this will ultimately depend on their beliefs and evolve in relation to market activity
        self.market_valuation = market_valuation
                # The number of contracts currently held
        self.n_contracts = n_contracts
                # Risk aversion score (if we indeed try to vary the risk aversion)
        self.risk_av = risk_av
                # Definition (loose): your willingness to change your opinion in the face of contradicting evidence
                # Will ideally be used to process "fact" or "news" - again, loosely defined for now
        self.stubbornness = stubbornness
                # This is the "clarity" with which a better views the true probability
                # Perhaps a different distribution...although this is something we can play with
        self.expertise = 1- expertise
                # Bias in the better's beliefs - this is the expected difference between the true probability and the better's belief
        self.bias = bias
        self.whale = whale
                
    def exp_utility(self, mkt_price, new_c):
        ''' 
        Expected utility function that takes into account the market price, budget, and market valuation of a better to determine the value of any trade (new_c)
        mkt_price: current market price
        new_c: possible number of contracts to trade: negative (positive) new_c = sell (buy)
        Return: Expected utility of a particular trade volume offered to sell or buy
        '''

        return self.market_valuation * self.utility(self.budget - mkt_price*new_c + self.n_contracts + new_c) + ((1-self.market_valuation)*self.utility(self.budget - mkt_price*new_c))
    
    def utility(self, w):
        ''' 
        Function defining utility of wealth according to risk aversion factor
        W: wealth 
        Return: risk-adjusted value of wealth
        '''
        if self.risk_av ==1:
            return np.log(w)
        else:
            return (w**(1-self.risk_av))/(1-self.risk_av)

    def trade(self, m):
        ''' 
        Function maximising the expected utility above to decide how many contracts to buy or sell constrained by budget
        Return: The number of contracts offerd to either sell (negative) or buy (positive)
        '''
        c_range = np.arange((-1*(int(self.budget) + self.n_contracts)), int(self.budget)+1)
        offered_contracts = c_range[np.argmax([self.exp_utility(m, x) for x in c_range])]
        return offered_contracts

    def update_belief(self, true_value): # stubbornness, risk aversion, information
        ''' 
        Individual better updates their belief as a function of their stubbornnes, expertise-adjusted signal of the true election outcome probability, and their current market valuation. 
        Updates internal belief 
        '''

        if not self.whale:
            self.market_valuation += (1-self.stubbornness)*(np.random.normal(true_value, self.expertise) - self.bias - self.market_valuation)
            # ensure value is within range [0,1]
            self.market_valuation = np.clip(self.market_valuation, 0, 1)
    

# Original code for managing order book
def fulfil_orders(buy_orders, sell_orders):
    sum_buy = np.sum(buy_orders)
    sum_sell = -np.sum(sell_orders)

    fulfilled_buy = np.zeros(len(buy_orders))
    fulfilled_sell = np.zeros(len(sell_orders))


    if sum_buy == sum_sell:
        fulfilled_buy = buy_orders
        fulfilled_sell = sell_orders
    
    elif sum_buy > sum_sell:
        fulfilled_sell = sell_orders

        # fulfil buy orders
        shuffled_indices = np.random.permutation(len(buy_orders))
        running_total = 0

        i=-1
        for i in range(np.sum(np.cumsum(buy_orders[shuffled_indices])<sum_sell)):
            j= shuffled_indices[i]
            running_total += buy_orders[j]
            fulfilled_buy[j] = buy_orders[j]

        i+=1
        j = shuffled_indices[i]
        fulfilled_buy[j] = sum_sell - running_total


    elif sum_buy < sum_sell:
        fulfilled_buy = buy_orders

        # fulfil sell orders
        shuffled_indices = np.random.permutation(len(sell_orders))
        running_total = 0
        i=-1
        for i in range(np.sum(np.cumsum(-sell_orders[shuffled_indices])<sum_buy)):
            j= shuffled_indices[i]
            running_total -= sell_orders[j]
            fulfilled_sell[j] = sell_orders[j]
        

        i+=1
        j = shuffled_indices[i]
        fulfilled_sell[j] = -(sum_buy - running_total)

    return fulfilled_buy, fulfilled_sell


def split_orders(all_orders):
    # Define a function which takes in a numpy array of orders and
    # returns two numpy arrays of buy and sell orders alongside a list of indices
    # to reconstruct the original array of orders from the fulfilled buy and sell orders.

    buy_orders = all_orders[all_orders > 0]
    sell_orders = all_orders[all_orders < 0]

    buy_indices = np.where(all_orders > 0)[0]
    sell_indices = np.where(all_orders < 0)[0]

    return buy_orders, sell_orders, buy_indices, sell_indices


def reconstruct_orders(buy_orders, sell_orders, buy_indices, sell_indices, N):
    # Define a function which takes in two numpy arrays of buy and sell orders
    # alongside two lists of indices and returns a single numpy array of orders
    # with the buy and sell orders in the correct place.

    all_orders = np.zeros(N)

    all_orders[buy_indices] = buy_orders
    all_orders[sell_indices] = sell_orders

    return all_orders


def manage_orders(order_book):
    # Define a function which takes in a list of orders and returns a list of fulfilled orders
    # alongside a list of unfilled orders.

    all_orders = np.array(order_book)
    buy_orders, sell_orders, buy_indices, sell_indices = split_orders(all_orders)

    fulfilled_buy, fulfilled_sell = fulfil_orders(buy_orders, sell_orders)

    return reconstruct_orders(fulfilled_buy, fulfilled_sell, buy_indices, sell_indices, len(order_book))


def set_market_price(m, net_supply_demand, total_orders):
    delta = net_supply_demand/(total_orders*10) if total_orders > 0 else 0
    m += delta
    return np.clip(m, 0, 1) # constrain market price to be within [0,1]

# Generate election outcome as a random walk
# Not sure whether this should be a series of probabilities or if it should be a series of 1 and 0
def gen_election(init_price, t_el, sd):
    el = [init_price]
    for k in range(t_el):
        el.append(np.clip(el[k] + np.random.normal(0, sd), 0, 1))
    return el


def run_market(n_betters, t_election, initial_price, outcome_uncertainty, betters):

    # Create true election probability
    gen_el = gen_election(initial_price, t_election, outcome_uncertainty)
    # Initialise betting population
    #betters = [better() for _ in range(n_betters)]
    # Set market price
    mkt_price = initial_price

    # Varioues records to store market price, volume on the market, average better beliefs, and net supply
    price_history = [mkt_price]
    vol_history = [sum(k.n_contracts for k in betters)]
    beliefs = [np.mean([k.market_valuation for k in betters])]
    beliefs_weighted = [np.average([k.market_valuation for k in betters], weights = [k.budget for k in betters])]
    market_pressure = [0]
    net_supply = [0]
    fulfilled_orders_list = []

    for t in range(t_election):

        order_book = [] # Initialise order book for time step t
        for b in betters:
            order_book.append(b.trade(mkt_price)) # add better's order volume to the order_book, positive for a buy order and negative for a sell order
        
        # Resolve order book
        fulfilled_orders = manage_orders(order_book)  # Fulfill buy-sell orders once every better has placed their order
        
        # Traders update portfolios according to fulfilled orders or not
        """ Q: For now, I have separated the process for buyers and sellers here as we do not allow for negative contract holdings...if a seller for example enters into/sells NEW contracts rather than contracts they already hold
            Not sure if this is a problem...we do not currently match each existing contract with a negative contract elsewhere...maybe we need this?
            Additionally, the below only works because of the budget constraint implicit in our trade() function -- correct? Perhaps assert or unit test here or in trade function.  """
        
        """ I don't actually think this is something we need to include - the budget constraint means you can sell contracts you don't have, but you need to put up $1 for each one (max payout). """
        
        for i in range(n_betters):
            betters[i].n_contracts += fulfilled_orders[i]
            betters[i].budget -= fulfilled_orders[i]*mkt_price
            # Buyers
            # if fulfilled_orders[i] > 0:
            #     betters[i].n_contracts += fulfilled_orders[i]
            #     betters[i].budget -= fulfilled_orders[i]*mkt_price

            # # # Sellers 
            # elif fulfilled_orders[i] < 0:
            #     betters[i].n_contracts += max(fulfilled_orders[i], -betters[i].n_contracts) # Ensures no negative contract holdings
            #     betters[i].budget -= fulfilled_orders[i]*mkt_price

        
        net_supply_demand = np.sum(order_book) # Get net supply (buy contracts - sell contracts)
        order_volume = np.sum(np.abs(order_book)) # Get total order volume
        market_pressure.append(net_supply_demand/order_volume if order_volume != 0 else 0) # Record market pressure
        assert(abs(net_supply_demand) <= order_volume)


        """ Q: The market price updating is not working...See last chunk/plot for demonstration of the problem: price converges to one or zero."""
        mkt_price = set_market_price(mkt_price, net_supply_demand, order_volume) #Update market price

        # Update beliefs feeding in the true probability of the election outcome at time t
        for b in betters:
            b.update_belief(gen_el[t]) 

        # Update records
        beliefs.append(np.mean([k.market_valuation for k in betters]))
        beliefs_weighted.append(np.average([k.market_valuation for k in betters], weights = [k.budget for k in betters]))
        price_history.append(mkt_price)
        net_supply.append(net_supply_demand)       
        vol_history.append(sum(np.abs(k.n_contracts) for k in betters)) 

        # to del
        fulfilled_orders_list.append(np.sum(fulfilled_orders))

    record = {'price_history': price_history,
            'beliefs': beliefs,
            'weighted_beliefs': beliefs_weighted,
            'vol_history': vol_history,
            'gen_el': gen_el,
            'net_supply': net_supply[1:],
            'market_pressure': market_pressure}

    return record


def plot_returns(rec, scale_fact = 10, step=False): ## Election Result

    # process market pressure
    mp =  np.array(rec['market_pressure'])[1:]
    upward_pressure = rec['price_history'][:-1] +  ([mp>0] * mp)[0]/scale_fact # positive market pressure
    downward_pressure = rec['price_history'][:-1]  + ([mp<0] * mp)[0]/scale_fact # negative market pressure
    N = len(rec["price_history"])

    fig, ax = plt.subplots(2,figsize=[8,7])
    plt.subplots_adjust(hspace=-5)

    if step:
        ax[0].plot(rec['price_history'], color = "black", label = "Market price", drawstyle = "steps")
        ax[0].fill_between(range(N-1), upward_pressure, rec['price_history'][:-1], color = "darkorange", alpha = 0.5, step = 'pre')
        ax[0].fill_between(range(N-1), downward_pressure, rec['price_history'][:-1], color = "dodgerblue", alpha = 0.5, step = 'pre')


        ax[0].plot(rec['gen_el'], color = "red", label = "Election outcome", drawstyle = "steps")
        ax[0].plot(rec['beliefs'], color = "purple", linestyle = "dotted", label = "Average market valuation of betters", drawstyle = "steps")


        # volume and net supply
        ax[1].plot(rec['vol_history'], color = "red", label = "No. of Contracts", drawstyle = "steps")
        ax[1].plot(rec['net_supply'], color = "blue", linestyle = "dotted", label = "Net Supply", drawstyle = "steps")

    else:
        ax[0].plot(rec['price_history'], color = "black", label = "Market price")
        ax[0].fill_between(range(N-1), upward_pressure, rec['price_history'][:-1], color = "darkorange", alpha = 0.5)
        ax[0].fill_between(range(N-1), downward_pressure, rec['price_history'][:-1], color = "dodgerblue", alpha = 0.5)


        ax[0].plot(rec['gen_el'], color = "red", label = "Election outcome")
        ax[0].plot(rec['beliefs'], color = "purple", linestyle = "dotted", label = "Average market valuation of betters")


        # volume and net supply
        ax[1].plot(rec['vol_history'], color = "red", label = "No. of Contracts")
        ax[1].plot(rec['net_supply'], color = "blue", linestyle = "dotted", label = "Net Supply")

    ax[0].set_ylim(0,1.02)
    ax[0].legend(loc = "lower center", bbox_to_anchor=(0.5, -.3), ncols=3)
    ax[1].legend(loc = "lower center", bbox_to_anchor=(0.5, -0.5), ncols=2)

    ax[0].set_title('Market Price')
    ax[1].set_title('Contract Volume & Net Supply')

    plt.xlabel('Time')
    ax[0].set_ylabel('Result')
    ax[1].set_ylabel('Result')
    plt.xticks(rotation=45)
    fig.tight_layout()

    # Display the plot
    plt.show()