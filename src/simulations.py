import numpy as np
from numba import njit

class PathSimulator:
    def __init__(self) -> None:
        pass

    @classmethod
    def bs(
        self,
        mu: float,
        T : float ,
        m : int ,
        n: int ,
        sigma: float,
        S0, 
    ):
        """
        Simulates multiple trajectories of the underlying asset price using a geometric Brownian motion (GBM) model.

        Parameters:
        mu (float): The drift rate, representing the expected return of the asset.
        T (float): The total time period for the simulation (in years). Default is `T`.
        m (int): The number of trajectories (simulated paths) to generate. Default is `M`.
        n (int): The number of time steps within each trajectory. Default is `N`.
        sigma (float): The volatility of the asset. Default is `sigma`.

        Returns:
        np.ndarray: A 2D array of shape (m, n+1) where each row represents a simulated trajectory of the asset price.
        """
        dt = T/n
        dW = np.random.normal(0, np.sqrt(dt), size=(m,n+1))

        dW[:, 0] = 0

        W = dW.cumsum(axis=1)
        t = np.linspace(0,T, n+1)
        trend = mu - sigma**2/2

        St = S0*np.exp(trend*t + sigma*W)

        return St
    
    @classmethod
    def heston(
        cls,
        T, 
        m,
        n,
        rho,
        kappa,
        theta,
        v0,
        S0,
        eta, 
        r
    ):
        dt = T/n

        brownian = np.random.multivariate_normal(
            [0,0], 
            [[1, rho],[rho,1]], 
            size=(m,n)
        )
        dW, dW_hat= brownian[:,:,0]*np.sqrt(dt), brownian[:,:,1]*np.sqrt(dt)

        return PathSimulator.__generate_trajectories_heston(
            dW,
            dW_hat,
            kappa,
            theta,
            v0, 
            S0,
            eta,
            dt,
            r, 
            n
        )
                
    @staticmethod
    @njit()
    def __generate_trajectories_heston(
        dW, 
        dW_hat, 
        kappa, 
        theta, 
        v0, 
        S0, 
        eta, 
        dt,
        r, 
        N
    ):
        v = np.zeros_like(dW_hat)
        S = np.zeros_like(v)
        v[:,0] = v0
        S[:, 0] = S0

        for t in range(1,N):
            v[:, t] = v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + eta * np.sqrt(np.maximum(v[:, t-1], 0)) * dW[:, t-1]
            
            # Ensure the volatility remains non-negative
            v[:, t] = np.maximum(v[:, t], 0)
            
            # Update price path using the Heston price process
            S[:, t] = S[:, t-1] * np.exp((r - 0.5 * v[:, t-1]) * dt + np.sqrt(v[:, t-1]) * dW_hat[:, t-1])
            

        return S, v