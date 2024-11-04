import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union
from scipy.stats import norm


class SimulationBasedModel:
    def __init__(self, ):
        self.memo = {}


    def possible_prices(self, St, t):
        """Compute possible prices given asset paths St at time t."""
        # Replace this with your actual `possible_prices` function.
         
        quantiles = np.linspace(0,1,t+3)
        return np.quantile(St[:, t], quantiles)[1:-1]
    
    def get_variance(self, St, vt, t):
        quantiles = np.linspace(0,1,t+3)

        variances = pd.Series(vt[:, t], name = 'variance')
        prices_t = pd.Series(St[:, t], name = 'prices')
        price_var = pd.concat([variances, prices_t], axis = 1)

        price_var['quantile'] = pd.qcut(price_var['prices'], quantiles)
        
        return price_var.groupby('quantile')\
            .mean()\
            .variance\
            .iloc[1:]\
            .to_numpy()\
            .reshape(-1,1)



    def compute_option_prices_counting(
        self, 
        St : np.ndarray, 
        vt : Union[float, np.ndarray], 
        g: callable, 
        n, 
        T, 
        K, 
        r,
        v0
    ):
        """Compute option prices using backward induction and quantile-based transitions."""
        dt = T/n
        self.memo = {}

        for t in (range(n - 1, -1, -1)):
            self.memo[t] = np.zeros(t)
            quantiles = np.linspace(0, 1, t + 2)

            # Current time step data
            variances = pd.Series(vt[:, t], name='variance_i')
            prices_t = pd.Series(St[:, t], name='prices_i')
            price_var = pd.concat([variances, prices_t], axis=1)
            price_var['q_i'] = pd.qcut(price_var['prices_i'], quantiles, labels=False)

            # Stopping values at time step t
            stopping_values = g(self.possible_prices(St, t), K)
            if t == n - 1:
                self.memo[t] = stopping_values
                continue

            # Next time step data
            q_next = np.linspace(0, 1, t + 3)
            variances_next = pd.Series(vt[:, t + 1], name='variance_i+1')
            prices_t_next = pd.Series(St[:, t + 1], name='prices_i+1')
            price_var_next = pd.concat([variances_next, prices_t_next], axis=1)
            price_var_next['q_i+1'] = pd.qcut(price_var_next['prices_i+1'], q_next, labels=False)

            # Transition matrix for quantiles
            df = pd.concat([price_var, price_var_next], axis=1)
            transitions = df.groupby(['q_i', 'q_i+1']).size().unstack(fill_value=0).to_numpy()
            probabilities = transitions / transitions.sum(axis=1, keepdims=True)

            # Continuation values calculation
            continuation_values = probabilities @ (self.memo[t + 1] * np.exp(-r * dt))
            self.memo[t] = np.maximum(continuation_values, stopping_values)

        self.option_prices = self.memo[0]
        return self.option_prices[0]
    
    def compute_option_prices_probabilities(
        self, 
        St : np.ndarray, 
        vt : Union[float, np.ndarray], 
        g: callable, 
        n, 
        T, 
        K, 
        r, 
        v0
    ):
        dt = T/n
        self.memo = {}

        for t in (range(n-1, -1, -1)):
            prices = self.possible_prices(St, t)
            if t == 0:
                var = v0
            
            else:
                var = self.get_variance(St, vt, t)

            var = np.mean(vt[:,t])

            # var = np.mean(vt[:,t])
            log_prices = np.log(prices)

            self.memo[t] = np.zeros_like(prices)
            stopping_values = g(prices, K)

            if t == n-1:
                self.memo[t] = stopping_values
                continue

            future_prices = self.possible_prices(St, t+1)
            log_future_prices = np.log(future_prices)
            
            log_future_prices[0:-1] += np.diff(log_future_prices)/2

            probabilities = np.zeros((t+1, t+2))
            matrix = np.tile(log_prices.reshape(t+1, 1), (1, t + 2))

            cdf_values = norm.cdf((log_future_prices-matrix- (r-var/2)*dt)/np.sqrt(dt*var))

            
            probabilities[:, 1:] = np.diff(cdf_values, axis = 1)
            probabilities[:, -1] = 1-cdf_values[:, -1]  
            probabilities[:, 0] = cdf_values[:,0]
            
        
            if (probabilities.sum(axis = 1) == 0).any():
                break
            
            probabilities = probabilities / probabilities.sum(axis = 1, keepdims = True)
        

            continuation_values = probabilities @ (self.memo[t+1]*np.exp(-r*dt))

            # memo[t]= continuation_values
            self.memo[t]= np.maximum(continuation_values, stopping_values)

        return self.memo[0][0]

