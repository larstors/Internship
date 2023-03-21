import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy import optimize


q0 = -1
qf = 0
k20 = 0
k2f = 0


class NG_memory:
    def __init__(
        self,
        a=10.0,
        b=2.0,
        sigma=1.0,
        lam=1.0,
        D1=1.0,
        D2=1.0,
        kappa=1.0,
        noise="g",
        pot="m",
        dx=1e-4,
        boundary_cond=[-1, 0, 0, 0],
        number_timestep=30,
        maxtime=1.0,
    ):
        """Initialisation of the class

        Args:
            a (float, optional): rescaled variance of characteristic funtion. Defaults to 10.0.
            b (float, optional): parameter for characteristic function: for details see individual functions. Defaults to 2.0.
            sigma (float, optional): variance for gaussian characteristic function. Defaults to 1.0.
            lam (float, optional): poisson shot noise rate. Defaults to 1.0.
            D1 (float, optional): gaussian noise variance on q. Defaults to 1.0.
            D2 (float, optional): gaussian noise variance on y. Defaults to 1.0.
            kappa (float, optional): inverse OU memory timescale. Defaults to 1.0.
            noise (str, optional): type amplitude distribution, choice between:
                    g for Gaussian,
                    d for delta,
                    e for two-sided exponential,
                    g for gamma,
                    t for truncated.
                    Defaults to "g".
            pot (str, optional): type of potential, choice between:
                    m for mexican hat.
                    Defaults to "m".
            dx (float, optional): discretisation for derivatives. Defaults to 1e-4.
            boundary_cond (list, optional): boundary conditions of form
                    [q_initial, q_final, y_initial, y_final].
                    Defaults to [-1, 0, 0, 0].
            number_timestep (int, optional): number of timesteps for instanton. Defaults to 30.0.
            maxtime (float, optional): maximal time. Defaults to 1.0.
        """
        self.a = a
        self.lamb = lam
        self.D1 = D1
        self.D2 = D2
        self.kappa = kappa
        self.dx = dx

        # noise parameters (if needed)
        self.b = b
        self.sigma = sigma

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

        # system parameters
        self.boundary_cond = boundary_cond
        self.number_timestep = number_timestep
        self.maxtime = maxtime

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


def Pot_mexico(x):
    """*mexican* hat potential

    Args:
        x (np.ndarray): position

    Returns:
        np.ndarray: value of potential at positions
    """
    return x**4 / 4 - x**2 / 2


def Phi_delta(x):
    """characteristic function for symmetric delta distributed

    Args:
        x (float): position at which to evaluate the characteric function

    Returns:
        float: value of characteristic function
    """
    return np.cosh(x) - 1


def F(
    t,
    s,
    a=10.0,
    b=1.2,
    sigma=1.0,
    lamb=0.01,
    D1=1.0,
    D2=1.0,
    kappa=1.0,
    phi=Phi_delta,
    dx=1e-4,
    potential=Pot_mexico,
):
    """System function

    Args:
        t (np.ndarray): time
        s (np.ndarray): system array
        p (list): initial parameters to be determined

    Returns:
        np.ndarray: system equations
    """
    q, k1, = s
    # q coordinate
    dqdt = + k1 * D1 - misc.derivative(potential, q, dx=dx) + lamb * a * misc.derivative(phi, k1 * a, dx=dx)

    # k1
    dk1dt = k1 * misc.derivative(potential, q, dx, n=2)

    # output
    return dqdt, dk1dt


@np.vectorize
def shooting_eval(x, maxtime=10, number_timestep=100):
    sol = solve_ivp(
        F,
        (0, maxtime),
        (-1, x),
        t_eval=np.linspace(0, maxtime, number_timestep),
    )
    q, k1 = sol.y
    return q[-1]


p0 = optimize.newton(shooting_eval, 0.2)


print(p0)
