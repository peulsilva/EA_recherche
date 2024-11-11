import numpy as np
from numba import njit
import torch

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
    def bs_gpu(self, mu: float, T: float, m: int, n: int, sigma: float, S0: float, device='cuda'):
        """
        Simulates multiple trajectories of the underlying asset price using a geometric Brownian motion (GBM) model
        on the GPU with PyTorch.

        Parameters:
        mu (float): The drift rate, representing the expected return of the asset.
        T (float): The total time period for the simulation (in years).
        m (int): The number of trajectories (simulated paths) to generate.
        n (int): The number of time steps within each trajectory.
        sigma (float): The volatility of the asset.
        S0 (float): The initial asset price.
        device (str): The device to use for the computation. Default is 'cuda' for GPU.

        Returns:
        torch.Tensor: A 2D tensor of shape (m, n+1) where each row represents a simulated trajectory of the asset price.
        """

        # Move calculations to the specified device (GPU if available)
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

        dt = T / n
        # Generate random increments dW, with shape (m, n+1) on the specified device
        dW = torch.randn((m, n+1), device=device) * torch.sqrt(torch.tensor(dt, device=device))

        # Set the first column of dW to 0
        dW[:, 0] = 0

        # Calculate cumulative sum to get W
        W = torch.cumsum(dW, dim=1)

        # Generate time steps from 0 to T
        t = torch.linspace(0, T, n+1, device=device)

        # Calculate the trend term
        trend = mu - 0.5 * sigma ** 2

        # Compute St using vectorized operations
        St = S0 * torch.exp(trend * t + sigma * W)

        return St

    
    @classmethod
    def heston_gpu(
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
        mean = torch.zeros(2, device='cuda')
        cov = torch.tensor([[1, rho], [rho, 1]], device='cuda')

        # Multivariate normal distribution
        mvn = torch.distributions.MultivariateNormal(mean, cov)

        # Generate samples (size: m x n x 2)
        brownian = mvn.sample((m, n)) * torch.sqrt(torch.tensor(dt, device='cuda'))

        # Extract dW and dW_hat
        dW = brownian[:, :, 0]
        dW_hat = brownian[:, :, 1]

        St, vt = PathSimulator.__generate_trajectories_heston_gpu(
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

        return St, vt

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
    
    @staticmethod
    def __generate_trajectories_heston_gpu(
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
    ):
        m = dW.size(0)
        St = torch.zeros((m, n), device='cuda')
        vt = torch.zeros((m, n), device='cuda')

        # Initial conditions
        St[:, 0] = S0
        vt[:, 0] = v0

        for t in range(1, n):
            # Ensure variance is non-negative
            sqrt_v_prev = torch.sqrt(torch.clamp(vt[:, t - 1], min=0))

            # Variance process
            vt[:, t] = vt[:, t - 1] + kappa * (theta - vt[:, t - 1]) * dt + eta * sqrt_v_prev * dW_hat[:, t - 1]
            vt[:, t] = torch.clamp(vt[:, t], min=0)

            # Price process
            St[:, t] = St[:, t - 1] * torch.exp((r - 0.5 * vt[:, t - 1]) * dt + sqrt_v_prev * dW[:, t - 1])

        return St, vt