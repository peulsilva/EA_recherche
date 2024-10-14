import numpy as np

def price_american_put_binomial(S0, K, r, sigma, T, M):
    """
    Price an American put option using the binomial tree model.

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying asset
    - T: Time to maturity
    - M: Number of time steps

    Returns:
    - Estimated option price
    """
    dt = T / M
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u                        # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability
    discount = np.exp(-r * dt)          # Discount factor

    # Initialize asset prices at maturity
    asset_prices = np.zeros(M + 1)
    for i in range(M + 1):
        asset_prices[i] = S0 * (u ** i) * (d ** (M - i))

    # Initialize option values at maturity (put payoff)
    option_values = np.maximum(K - asset_prices, 0)

    # Work backward through the tree
    for t in range(M - 1, -1, -1):
        for i in range(t + 1):
            asset_prices[i] = S0 * (u ** i) * (d ** (t - i))
            continuation_value = discount * (p * option_values[i + 1] + (1 - p) * option_values[i])
            exercise_value = K - asset_prices[i]
            option_values[i] = max(exercise_value, continuation_value)  # American option can be exercised early
    
    return option_values[0]

