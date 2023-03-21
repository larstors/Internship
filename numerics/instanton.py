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
        # q coordinate
        output1 = (
            s[2] * self.D1 + s[1] - misc.derivative(self.potential, s[0], dx=self.dx)
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

    def Residuals(self, ini, fin, p):
        """Residuals for boundary conditions

        Args:
            ini (np.ndarray): initial values
            fin (np.ndarray): final values
            p (list): initial parameters to be determined

        Returns:
            np.ndarray: residuals
        """
        k = p[0]
        k2 = p[1]
        res = np.zeros(6)
        res[0] = ini[0] - self.boundary_cond[0]
        res[1] = fin[0] - self.boundary_cond[1]
        res[2] = ini[3] - self.boundary_cond[2]
        res[3] = fin[3] - self.boundary_cond[3]

        res[4] = ini[1] - k
        res[5] = ini[2] - k2
        return res

    def instanton(self):
        """Function to recover the instanton of the system given some initial conditions. The method used is the
        boundary value problem solver by scipy, see shooting method

        Returns:
            scipy.integrate._bvp.BVPResult: Result from the integrator
        """
        # time
        t = np.linspace(0, self.maxtime, self.number_timestep)

        # value array
        y = np.zeros((4, t.size))
        # apply initial guess
        y[0, 0] = self.boundary_cond[0]
        y[3, 0] = self.boundary_cond[2]

        result = solve_bvp(self.F, self.Residuals, t, y, p=[0.2, 0.2], verbose=2)

        print(result.message)
        return result, t


class NG_no_memory:
    def __init__(
        self,
        a=10.0,
        b=2.0,
        sigma=1.0,
        lam=1.0,
        D1=1.0,
        noise="g",
        pot="m",
        dx=1e-4,
        boundary_cond=[-1, 0, 0.2],
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
            noise (str, optional): type amplitude distribution, choice between:
                    g for Gaussian,
                    d for delta,
                    e for two-sided exponential,
                    ga for gamma,
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
        elif noise == "ga":
            self.phi = self.Phi_gamma
        elif noise == "t":
            self.phi = self.Phi_truncated

        # potential
        if pot == "m":
            self.potential = self.Pot_mexico
        elif pot == "p":
            self.potential = self.Pot_paper
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

    def Pot_paper(self, x):
        return x**4 - 6 * x**2 - 2 * x + 5

    def Phi_gauss(self, x):
        """characteristic function for gaussian distribution

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return np.exp(-(x**2) * self.sigma**2 / 2.0) - 1

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

        Returns:
            np.ndarray: system equations
        """
        # q coordinate
        output1 = (
            s[1] * self.D1
            - misc.derivative(self.potential, s[0], dx=self.dx)
            + self.lamb * self.a * misc.derivative(self.phi, s[1] * self.a, dx=self.dx)
        )

        # k
        output2 = s[1] * misc.derivative(self.potential, s[0], self.dx, n=2)

        # output
        return np.vstack((output1, output2))

    def Residuals(self, ini, fin, p):
        """Residuals for boundary conditions

        Args:
            ini (np.ndarray): initial values
            fin (np.ndarray): final values

        Returns:
            np.ndarray: residuals
        """
        res = np.zeros(3)
        res[0] = ini[0] - self.boundary_cond[0]
        res[1] = fin[0] - self.boundary_cond[1]
        res[2] = ini[1] - p[0]
        return res

    def guess(self, x):
        return -1 / (1 + np.exp(x - 5))

    def instanton(self):
        """Function to recover the instanton of the system given some initial conditions. The method used is the
        boundary value problem solver by scipy, see shooting method

        Returns:
            scipy.integrate._bvp.BVPResult: Result from the integrator
        """
        # time
        t = np.linspace(0, self.maxtime, self.number_timestep)

        # value array
        y = np.zeros((2, t.size))
        # apply initial guess
        # y[0] = self.guess(t)
        y[0, 0] = self.boundary_cond[0]
        y[1, 0] = self.boundary_cond[2]

        result = solve_bvp(
            self.F,
            self.Residuals,
            t,
            y,
            p=[self.boundary_cond[2]],
            verbose=2,
            max_nodes=3500,
        )

        print(result.message)
        print(result.p)
        return result, t


# mem = NG_memory(lam=1, a=1, maxtime=10, noise="d", number_timestep=30)
# mem_G = NG_memory(lam=0, a=1, maxtime=10, noise="d", number_timestep=30)
# instanton_mem, t = mem.instanton()
# instanton_mem_G, t = mem_G.instanton()

nsteps = 100
boundary = [-1, 0, 0.2]
l = 1
a = 1


nomem_t = NG_no_memory(
    lam=l,
    a=a,
    maxtime=10,
    noise="t",
    number_timestep=nsteps,
    b=0.5,
    pot="m",
    boundary_cond=boundary,
)

nomem_e = NG_no_memory(
    lam=l,
    a=a,
    maxtime=10,
    noise="e",
    number_timestep=nsteps,
    pot="m",
    boundary_cond=boundary,
)

nomem_g = NG_no_memory(
    lam=0,
    a=a,
    maxtime=10,
    noise="g",
    number_timestep=nsteps,
    pot="m",
    boundary_cond=boundary,
)

nomem = NG_no_memory(
    lam=l,
    a=a,
    maxtime=10,
    noise="d",
    number_timestep=nsteps,
    pot="m",
    boundary_cond=boundary,
)

# nomem_ga = NG_no_memory(lam=l, a=a, maxtime=10, noise="ga", number_timestep=nsteps, b=1.2, pot="m", boundary_cond=boundary)

instanton_no_mem, t = nomem.instanton()
instanton_no_mem_t, t = nomem_t.instanton()
instanton_no_mem_e, t = nomem_e.instanton()
instanton_no_mem_g, t = nomem_g.instanton()

# instanton_no_mem_ga, t = nomem_ga.instanton()


# plt.plot(t, instanton_mem.sol(t)[0], label="OU, NG")
# plt.plot(t, instanton_mem_G.sol(t)[0], "--", label="OU, Gaussian")
plt.plot(t, instanton_no_mem.sol(t)[0], label="non-OU, const")
#plt.plot(t, instanton_no_mem_t.sol(t)[0], label="non-OU, trunated")
#plt.plot(t, instanton_no_mem_e.sol(t)[0], label="non-OU, exp")
# plt.plot(t, instanton_no_mem_ga.sol(t)[0], label="non-OU, gamma")

plt.plot(t, instanton_no_mem_g.sol(t)[0], "--", label=r"non-OU, $\lambda=0$")

plt.legend()
plt.grid()
plt.xlabel(r"$t$")
plt.ylabel(r"$q$")


plt.show()







quest = []

for i in range(1, 100, 1):
    xg = 0.01 * i
    nomem.boundary_cond[2] = xg
    solution, t = nomem.instanton()
    quest.append(solution.sol(t)[1, 0])

for i in range(len(quest)):
    print(quest[i])
