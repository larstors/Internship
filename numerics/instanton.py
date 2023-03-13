import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from scipy.optimize import fsolve
from scipy.integrate import solve_bvp


"""
Scipt for finding instanton of a OU-process with non-Gaussian noise

We use the system vector s of the form:

    s = (q, y, k_1, k_2)


"""


class NG_memory:
    def __init__(
        self,
        a=10,
        b=2,
        sigma=1,
        lam=1,
        D1=1,
        D2=1,
        kappa=1,
        noise="g",
        pot="m",
        dx=1e-4,
        boundary_cond=[-1, 0, 0, 0],
        number_timestep=30,
        maxtime=1,
    ) -> None:
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

    def Pot_mexico(self, x) -> float:
        return x**4 / 4 - x**2 / 2

    def Phi_gauss(self, x) -> float:
        return np.exp(x**2 * self.sigma**2 / 2)

    def Phi_delta(self, x) -> float:
        return np.cosh(x) - 1

    def Phi_exp(self, x) -> float:
        return x**2 / (2 * (1 - x**2))

    def Phi_gamma(self, x) -> float:
        return ((1 + x) ** self.b + (1 - x) ** self.b - 2) / (2 * self.b * (self.b - 1))

    def Phi_truncated(self, x) -> float:
        return 0.5 * x**2 + self.b * x**4

    def F(self, t, s, p) -> np.ndarray:
        # q coordinate
        output1 = (
            +s[2] * self.D1 + s[1] - misc.derivative(self.potential, s[0], dx=self.dx)
        )

        # y coordinate
        output2 = (
            -self.kappa * s[1]
            + self.D2 * s[3]
            + self.lamb * self.a * misc.derivative(self.phi, s[3] * self.a, dx=self.dx)
        )

        # k1
        output3 = s[2] * misc.derivative(self.potential, s[0], self.dx, n=2)

        # k2
        output4 = -s[2] + self.kappa * s[3]

        # output
        return np.vstack((output1, output2, output3, output4))

    def Residuals(self, ini, fin, p) -> np.ndarray:
        k = p[0]
        k2 = p[1]
        res = np.zeros(6)
        res[0] = ini[0] - self.boundary_cond[0]
        res[1] = fin[0] - self.boundary_cond[1]
        res[2] = ini[1] - self.boundary_cond[2]
        res[3] = fin[1] - self.boundary_cond[3]

        res[4] = ini[2] - k
        res[5] = ini[3] - k2
        return res

    def instanton(self):
        # time
        t = np.linspace(0, self.maxtime, self.number_timestep)

        # value array
        y = np.zeros((4, t.size))
        # apply initial guess
        y[0, 0] = self.boundary_cond[0]
        y[1, 0] = self.boundary_cond[2]

        result = solve_bvp(self.F, self.Residuals, t, y, p=[0.2, 0.2], verbose=0)

        print(result.message)
        return result


test = NG_memory()

work = test.instanton()

print(work.message)
