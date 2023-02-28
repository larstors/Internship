import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import random as rn
import os
import time
import sys

"""
Simply code to produce some visualisations of the path of an OU process with non-Gaussian noise.

"""


def Gaussian_Distribution(variance: float):
    """We always want 0 mean Gaussian

    Args:
        variance (float): variance

    Returns:
        float: random gaussian number with 0 mean and variance=variance
    """
    return np.random.normal(0, variance)


def Laplace_Distribution(scale: float):
    """Laplace distribution centered around 0

    Args:
        scale (float): exponential scale of the exponential part of the laplace distribution

    Returns:
        float: random laplace distributed number with 0 mean
    """
    return np.random.laplace(0, scale)


def Poisson_Shot_Noise(
    pA, tfinal: float, tinitial: float, timestep: float, lam: float, *args
):
    """Creates an array corresponding to the impulses taking place (at discrete times, i.e. multiples of time step width)

    Args:
        pA (function): distribution of amplitude of the pulses
        tfinal (float): Final Time
        tinitial (float): Initial Time
        timestep (float): width of time step
        lam (float): weight of poisson process
    """

    # total number of kicks in the PSN follows a poisson distribution with mean lambda*t
    N = np.random.poisson(lam * (tfinal - tinitial))

    # the random times at which the kicks occur are also drawn from poisson with weight lambda
    T = np.random.poisson(lam, size=N)
    T = np.sort(T) * timestep

    # corresponding amplitudes
    A = np.array([pA(args)[0] for a in range(N)])

    return T, A


def potential(x: np.ndarray, a: float, b: float, c: float):
    """potential for the problem. It has the form ax + bx^2 + cx^4

    Args:
        x (np.ndarray): position
        a (float): parameter for linear term
        b (float): parameter for quadratic term
        c (float): parameter for quartic term

    Returns:
        np.ndarray: potential at position x
    """
    return a * x + b * x**2 + c * x**4


def potential_gradient(x: np.ndarray, a: float, b: float, c: float):
    """gradient of the potential

    Args:
        x (np.ndarray): position
        a (float): parameter for linear term
        b (float): parameter for quadratic term
        c (float): parameter for quartic term

    Returns:
        np.ndarray: gradient of potential at position x
    """
    return a + 2 * b * x + 4 * c * x**3


######################## System Parameters ##########################

# initial value of path
x0 = -1

# lambda for the poisson process

# a, b, c for the potential


x = np.linspace(-1, 1, 1000)

plt.plot(x, potential(x, 0.1, -1, 1))
plt.show()

print(Poisson_Shot_Noise(Gaussian_Distribution, 5, 0, 0.5, 2, 1))
