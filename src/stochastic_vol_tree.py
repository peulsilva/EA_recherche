from typing import Any
import numpy as np

class StochasticVolTree:
    """
    A class to represent a stochastic volatility tree for option pricing.

    This class provides methods to calculate various components of a stochastic volatility model,
    such as the drift and volatility terms, as well as the option value based on the tree model.
    """
    memo = {}

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.memo = {}
        return self.compute_option_value(0,0,0,0,0, **kwds)

    def get_XY(self, k, l, m, **kwargs):
        """
        Computes the values of X and Y at a given point in the tree.

        Parameters:
        - k (int): Current time step.
        - l (int): Current state index related to X.
        - m (int): Current state index related to Y.
        - kwargs (dict): Parameters including 'x0', 'y0', 'eta', 'h', and 'rho'.

        Returns:
        - tuple: (Xk, Yk) representing the X and Y coordinates.
        """
        x0, y0, eta, h, rho = kwargs['x0'], kwargs['y0'], kwargs['eta'], kwargs['h'], kwargs['rho']
        Xk = x0 + (2 * l - k) * np.sqrt(eta * h)
        Yk = y0 + (2 * m - k) * np.sqrt(eta * h * (1 - rho ** 2))

        return Xk, Yk

    def sigma(self, x, y, **kwargs):
        """
        Computes the local volatility.

        Parameters:
        - x (float): X value.
        - y (float): Y value.
        - kwargs (dict): Parameters including 'rho'.

        Returns:
        - float: The local volatility value.
        """
        rho = kwargs['rho']
        return np.sqrt(np.maximum(0, y + rho * x))

    def mu_X(self, x, y, **kwargs):
        """
        Computes the drift term for X.

        Parameters:
        - x (float): X value.
        - y (float): Y value.
        - kwargs (dict): Parameters including 'r', 'eta', and 'rho'.

        Returns:
        - float: The drift of X.
        """
        r, eta, rho = kwargs['r'], kwargs['eta'], kwargs['rho']
        return r - 1 / 2 * eta * (y + rho * x)

    def mu_Y(self, x, y, **kwargs):
        """
        Computes the drift term for Y.

        Parameters:
        - x (float): X value.
        - y (float): Y value.
        - kwargs (dict): Parameters including 'r', 'eta', 'rho', 'kappa', and 'theta'.

        Returns:
        - float: The drift of Y.
        """
        r, eta, rho, kappa, theta = kwargs['r'], kwargs['eta'], kwargs['rho'], kwargs['kappa'], kwargs['theta']
        return kappa * theta / eta - rho * r + 1 / 2 * (rho * eta - 2 * kappa) * (y + rho * x)

    def alpha(self, k, l, m, **kwargs):
        """
        Computes the alpha term which adjusts the volatility at a given step.

        Parameters:
        - k (int): Current time step.
        - l (int): Current state index related to X.
        - m (int): Current state index related to Y.
        - kwargs (dict): Parameters including 'A'.

        Returns:
        - float: The alpha adjustment term.
        """
        A = kwargs['A']

        if k == 0:
            return 0

        X_k_1, Y_k_1 = self.get_XY(k - 1, l, m, **kwargs)
        return (self.sigma(X_k_1, Y_k_1, **kwargs) ** 2 - 1) / 2

    def current_payoff(self, k, l, m, eX, g=lambda x, strike: np.maximum(strike - x, 0), **kwargs):
        """
        Computes the current payoff of the option.

        Parameters:
        - k (int): Current time step.
        - l (int): Current state index related to X.
        - m (int): Current state index related to Y.
        - eX (int): Modifier for state transition.
        - g (callable): Payoff function (default is European put option).
        - kwargs (dict): Parameters including 'eta', 'K', and 'h'.

        Returns:
        - float: The payoff value.
        """
        eta = kwargs['eta']
        K, h = kwargs['K'], kwargs['h']
        X_k, _ = self.get_XY(k, l, m, **kwargs)
        S_k = np.exp(X_k + np.sqrt(h * eta) * self.alpha(k, l, m, **kwargs) * eX)

        return g(S_k, K)

    def compute_option_value(self, k, l, m, eX, eY, **kwargs):
        """
        Computes the option value at a given point in the tree using backward induction.

        Parameters:
        - k (int): Current time step.
        - l (int): Current state index related to X.
        - m (int): Current state index related to Y.
        - eX (int): Modifier for state transition in X.
        - eY (int): Modifier for state transition in Y.
        - kwargs (dict): Model parameters.

        Returns:
        - float: The computed option value.
        """
        h = kwargs.get('T') / kwargs.get('N')  # Assuming 'T' is passed through kwargs
        N = kwargs.get('N')

        if (k, l, m, eX, eY) in  self.memo:
            return self.memo[(k, l, m, eX, eY)]

        phi_k_plus_one = 1 + self.alpha(k + 1, l, m, **kwargs)
        phi_k = 1 + self.alpha(k, l, m, **kwargs)

        X_k_minus_one, Y_k_minus_one = self.get_XY(k - 1, l, m, **kwargs)
        X_k_plus_one, Y_k_plus_one = self.get_XY(k + 1, l, m, **kwargs)

        p_k_plus_one = (np.exp(kwargs.get('r') * h + np.sqrt(kwargs.get('eta') * h) * phi_k_plus_one * eX) -
                        np.exp(-np.sqrt(kwargs.get('eta') * h) * phi_k)) / \
                       (np.exp(np.sqrt(kwargs.get('eta') * h) * phi_k_plus_one) -
                        np.exp(-np.sqrt(kwargs.get('eta') * h) * phi_k_plus_one))
        p_k_plus_one = np.clip(p_k_plus_one, 0, 1)

        q_k_plus_one = 1 / 2 + self.alpha(k - 1, l - eX, m - eY, **kwargs) * eY / (2 * phi_k_plus_one) + \
                       np.sqrt(h / (kwargs.get('eta') * (1 - kwargs.get('rho') ** 2))) * \
                       self.mu_Y(X_k_plus_one, Y_k_plus_one, **kwargs) / (2 * phi_k_plus_one)
        q_k_plus_one = np.clip(q_k_plus_one, 0, 1)

        if k == N:
            return self.current_payoff(N, l, m, eX, **kwargs)

        epsilon_V = 0  # continuation value
        for i in [0, 1]:
            for j in [0, 1]:
                epsilon_V += (1 - i + (2 * i - 1) * p_k_plus_one) * (1 - j + (2 * j - 1) * q_k_plus_one) * \
                             self.compute_option_value(k + 1, l + i, m + j, 2 * i - 1, 2 * j - 1, **kwargs)

        self.memo[(k, l, m, eX, eY)] = np.maximum(epsilon_V, self.current_payoff(k, l, m, eX, **kwargs))
        return self.memo[(k, l, m, eX, eY)]