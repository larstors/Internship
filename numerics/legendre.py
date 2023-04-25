import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.misc as misc
from numba import njit, jit, parallel
import sys



class transform:
    def __init__(self, lambda_, a, tau, D1, D2, noise="d", pot="m"):
        
        # parameters
        self.a = a
        self.lambda_ = lambda_
        self.tau = tau
        self.D1 = D1
        self.D2 = D2

    
        # characteristic function of noise
        if noise == "g":
            self.phi = self.Phi_gauss
        elif noise == "d":
            self.phi = self.Phi_delta
        elif noise == "e":
            self.phi = self.Phi_exp
        elif noise == "g":
            self.phi = self.Phi_gamma
        elif noise == "t":
            self.phi = self.Phi_truncated

        # potential
        if pot == "m":
            self.potential = self.Pot_mexico
        # TODO add more potentials (skewed, or other?)


    def Pot_mexico(self, x):
        """*mexican* hat potential

        Args:
            x (np.ndarray): position

        Returns:
            np.ndarray: value of potential at positions
        """
        return x**4 / 4 - x**2 / 2

    def Phi_gauss(self, x):
        """characteristic function for gaussian distribution

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return np.exp(x**2 * self.sigma**2 / 2.0)

    def Phi_delta(self, x):
        """characteristic function for symmetric delta distributed

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return np.cosh(x) - 1

    def Phi_exp(self, x):
        """characteristic function for symmetric exponential distribution

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return x**2 / (2 * (1 - x**2))

    def Phi_gamma(self, x):
        """characteristic function for gamma distribution

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return ((1 + x) ** self.b + (1 - x) ** self.b - 2) / (2 * self.b * (self.b - 1))

    def Phi_truncated(self, x):
        """characteristic function for truncated distribution

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return 0.5 * x**2 + self.b * x**4

    def Psi(self, k2, k1=0):

        return self.D1 * k1 ** 2 / 2 + self.D2 * k2 ** 2 * 0.5 + self.lambda_ * self.phi(self.a * k2)

    def Legendre_transform(self, initial, qDot: np.ndarray, q: np.ndarray, yDot: np.ndarray, y: np.ndarray):

        f0 = qDot + misc.derivative(self.potential, q, dx=self.dx) - y
        f1 = self.tau * yDot + y 

        k1 = f0 / self.D1

        k2 = opt.fsolve()

