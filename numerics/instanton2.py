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

    def F(self, t, s, p):
        """System function

        Args:
            t (np.ndarray): time
            s (np.ndarray): system array
            p (list): initial parameters to be determined

        Returns:
            np.ndarray: system equations
        """
        q, y, k1, k2 = s
        # q coordinate
        dqdt = +k1 * self.D1 + y - misc.derivative(self.potential, q, dx=self.dx)

        # y coordinate
        dydt = (
            -self.kappa * y
            + self.D2 * k2
            + self.lamb * self.a * misc.derivative(self.phi, k2 * self.a, dx=self.dx)
        )

        # k1
        dk1dt = k1 * misc.derivative(self.potential, q, self.dx, n=2)

        # k2
        dk2dt = -k1 + self.kappa * k2

        # output
        return dqdt, dydt, dk1dt, dk2dt

    @np.vectorize
    def shooting_eval(p0, self):
        sol = solve_ivp(
            self.F,
            (0, self.maxtime),
            (self.boundary_cond[0], p0[0], p0[1], self.boundary_cond[2]),
            t_eval=np.linspace(0, self.maxtime, self.number_timestep),
        )
        y_num, v = sol.y
        return y_num[-1]

    def optimal_initial(self):
        p0 = optimize.newton(self.shooting_eval, x0=np.array([0.2, 0.2]), args=())
        return p0


test = NG_memory(lam=0, noise="d")
opt = test.optimal_initial()

print(opt)
