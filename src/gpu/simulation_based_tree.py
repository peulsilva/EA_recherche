import torch
import numpy as np

class SimulationBasedModel:
    def __init__(self, n, K, r, dt, option_type : str, device='cuda'):
        self.n = n
        self.K = K
        self.r = r
        self.dt = dt
        self.device = device
        self.memo = {}

        assert option_type in ['european', 'american']
        self.option_type = option_type

    def compute_quantiles(self, values, num_quantiles):
        """Compute quantile bins using torch."""
        if num_quantiles == 3:
            return torch.ones_like(values, dtype = torch.long), values
        sorted_vals, _ = torch.sort(values)
        quantile_indices = (torch.linspace(0, len(values) - 1, num_quantiles, device=self.device)).long()
        quantile_values = sorted_vals[quantile_indices]
        quantile_values[0] = -float('inf')  # Ensure all values are included
        quantile_values[-1] = float('inf')
        bins = torch.bucketize(values, quantile_values[1:-1], right=False)
        return bins, quantile_values


    def compute_transition_matrix(self, current_bins, next_bins,):
        """Compute the transition matrix based on quantile bins."""
        # Create a 2D histogram of transitions

        num_quantiles_current = current_bins.max()
        num_quantiles_next = next_bins.max()

        # if num_quantiles_current > 1:
        current_bins = current_bins.clamp(max=num_quantiles_current - 1)
        next_bins = next_bins.clamp(max=num_quantiles_next - 1)
        
        # Initialize the transition matrix with the correct dimensions
        
        transition_matrix = torch.zeros((num_quantiles_current, num_quantiles_next), dtype=torch.long, device=current_bins.device)

        # Populate the transition matrix
        transition_matrix.index_put_((current_bins, next_bins), torch.ones_like(current_bins), accumulate=True)

        probabilities = transition_matrix/transition_matrix.sum(dim=1, keepdim=True)
        return probabilities

    def compute_option_prices_counting(self, St, g : callable):
        """Compute option prices with backward induction."""
        St = St.to(self.device)

        M, n = St.shape
        self.memo = {}  # Initialize memoization list


        for t in range(n - 1, -1, -1):

            # Compute quantile bins for current and next time steps
            bins_current, prices_t = self.compute_quantiles(St[:, t], t+3)

            

            if t == n-1:
                self.memo[t] = (g(prices_t[1:-1], self.K))
                continue

            bins_next, prices_t_next = self.compute_quantiles(St[:, t + 1], t+4)


            # Compute transition probabilities

            # return bins_current, bins_next
            probabilities = self.compute_transition_matrix(
                bins_current, 
                bins_next,
            )


            continuation_values = probabilities @ (self.memo[t+1] * torch.exp(torch.tensor(-self.r * self.dt)))

            stopping_values_all = g(prices_t, self.K)
            if self.option_type == 'european':
                self.memo[t] = continuation_values

            elif self.option_type == 'american':
                self.memo[t] = torch.maximum(continuation_values, stopping_values_all)

        # Option price at time 0
        option_price = self.memo[0][0]
        return option_price

    def possible_prices(self, St, t):
        """Compute possible prices given asset paths St at time t."""
        device = St.device
        prices_t = St[:, t]
        quantiles = torch.linspace(0, 1, steps=t + 3, device=device)
        quantile_values = torch.quantile(prices_t, quantiles)
        # Exclude the first and last quantiles
        return quantile_values[1:-1]

    def get_variance(self, St, vt, t):
        device = St.device
        prices_t = St[:, t]
        variances_t = vt[:, t]

        quantiles = torch.linspace(0, 1, steps=t + 3, device=device)
        quantile_values = torch.quantile(prices_t, quantiles)

        # Assign bins to prices
        bins = torch.bucketize(prices_t, quantile_values[1:-1], right=False)

        num_bins = t + 2  # Number of bins is t + 2 since we exclude first and last quantiles
        sum_variances_per_bin = torch.zeros(num_bins, device=device)
        counts_per_bin = torch.zeros(num_bins, device=device)

        # Aggregate variances per bin
        sum_variances_per_bin.scatter_add_(0, bins, variances_t)
        counts_per_bin.scatter_add_(0, bins, torch.ones_like(variances_t))

        # Compute mean variances per bin
        mean_variances_per_bin = sum_variances_per_bin / (counts_per_bin + 1e-8)  # Avoid division by zero

        # Exclude the first bin to mimic iloc[1:]
        mean_variances_per_bin = mean_variances_per_bin[1:]

        # Return as a column vector
        return mean_variances_per_bin.view(-1, 1)

    def torch_norm_cdf(self, x):
        return 0.5 * (1.0 + torch.erf(x / np.sqrt(2)))

    def compute_option_prices_probabilities(
            self,
            St: torch.Tensor,
            vt,
            g: callable,
            n: int,
            T: float,
            K: float,
            r: float,
            v0: float
        ) -> torch.Tensor:
        dt = T / n
        self.memo = {}

        device = St.device  # Ensure computations are on the same device

        for t in range(n - 1, -1, -1):
            prices = self.possible_prices(St, t)  # Should return torch.Tensor
            if t == 0:
                var = v0
            else:
                var = v0

            log_prices = torch.log(prices)

            self.memo[t] = torch.zeros_like(prices, device=device)
            stopping_values = g(prices, torch.tensor(K))

            if t == n - 1:
                self.memo[t] = stopping_values
                continue

            future_prices = self.possible_prices(St, t + 1)
            log_future_prices = torch.log(future_prices)

            # Adjust future log prices
            diff_log_future_prices = torch.diff(log_future_prices)
            log_future_prices_adjusted = log_future_prices.clone()
            log_future_prices_adjusted[:-1] += diff_log_future_prices / 2

            # Create matrix for broadcasting
            matrix = log_prices.view(-1, 1).repeat(1, t + 2)

            # Compute the standardized variable for the normal CDF
            x = (log_future_prices_adjusted - matrix - (r - var / 2) * dt) / torch.sqrt(torch.tensor(dt * var))

            # Compute CDF values
            cdf_values = self.torch_norm_cdf(x)

            # Initialize probabilities
            probabilities = torch.zeros((t + 1, t + 2), device=device)

            # Calculate probabilities
            probabilities[:, 1:] = torch.diff(cdf_values, dim=1)
            probabilities[:, -1] = 1 - cdf_values[:, -1]
            probabilities[:, 0] = cdf_values[:, 0]

            # Check for zero rows
            if (probabilities.sum(dim=1) == 0).any():
                break

            # Normalize probabilities
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

            # Compute continuation values
            continuation_values = probabilities @ (self.memo[t + 1] * torch.exp(torch.tensor(-r * dt)))

            # Update memoization
            if self.option_type == 'european':
                self.memo[t] = continuation_values

            elif self.option_type == 'american':
                self.memo[t] = torch.maximum(continuation_values, stopping_values)

        return self.memo[0][0]