import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.misc as misc
from numba import njit, jit
import sys



class transform:
    def __init__(self, lambda_, a, tau, D1, D2, N, noise="d", pot="m", tmax=10):
        
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

        # initial guess length
        self.N = N

        #max time
        self.tmax = tmax

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

    def Opt_Func(self, k2, f1):
        return f1 - misc.derivative(self.Psi, k2)

    def Legendre_transform(self, initial, qDot: np.ndarray, q: np.ndarray, yDot: np.ndarray, y: np.ndarray):

        f0 = qDot + misc.derivative(self.potential, q) - y
        f1 = self.tau * yDot + y 

        if self.D1 != 0:
            k1 = f0 / self.D1
        else:
            k1 = np.zeros_like(f0)

        k2 = opt.fsolve(self.Opt_Func, initial, args=f1)

        return k1, k2

    def MSR_action(self, init_values):
        
        delta_t = self.tmax / self.N

        q = init_values[:self.N]
        y = init_values[self.N:]

        qdot = (q[1:] - q[:-1]) / delta_t
        ydot = (y[1:] - y[:-1]) / delta_t

        k1, k2 = self.Legendre_transform(np.zeros_like(qdot), qdot, q[:-1], ydot, y[:-1])

        S = 0
        
        for i in range(self.N - 1):
            S += k1[i] * (qdot[i] + misc.derivative(self.potential, q[i]) - y[i]) + k2[i] * (self.tau * ydot[i] + y[i]) - self.D1 / 2 * k1[i] ** 2 - self.D2 / 2 * k2[i] ** 2 - self.lambda_ * self.phi(self.a * k2[i])

        return S

    def init_constraint(self, t):
        """Function for initial constraint that q(0) = -1

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for initial constraint
        """
        return t[0] + 1
    
    def final_constraint(self, t):
        """Function for final constraint that q(tf) = 0

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for final constraint
        """
        return t[self.N - 1]

    def minimize(self):
        system = np.zeros(2 * self.N)

        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}]

        optimum = opt.minimize(self.MSR_action, x0=system, constraints=constraint, options={'disp': True})

        return optimum


N = 40

t = transform(0.0, 10, 0.0001, 1, 1, N, "d", "m", 10)
op = t.minimize()

print(op.message)
print(op.x[:N])

time = np.linspace(0, 10, N, endpoint=True)

plt.plot(time, op.x[:N])
plt.xlabel(r"$t$")
plt.ylabel(r"$q$")
plt.show()
