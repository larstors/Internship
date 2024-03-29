import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from scipy.optimize import fsolve
from scipy.integrate import solve_bvp

# plt.rcParams.update({"font.size": 8})

"""
Scipt for finding instanton of a OU-process with non-Gaussian noise

We use the system vector s of the form:

    s = (q, y, k_1, k_2)


"""

class noise_and_potential:
    def __init__(self, sigma=1, b=0.5, a=10, scaling=1.0):
        self.sigma = sigma
        self.b = b
        self.a = a
        self.scaling = scaling


    def Pot_mexico(self, x):
        """*mexican* hat potential

        Args:
            x (np.ndarray): position

        Returns:
            np.ndarray: value of potential at positions
        """
        return x**4 / 4 - x**2 / 2

    def dPot_mexico(self, x):
        """*mexican* hat potential

        Args:
            x (np.ndarray): position

        Returns:
            np.ndarray: value of potential at positions
        """
        return x**3 - x
    
    def ddPot_mexico(self, x):
        """*mexican* hat potential

        Args:
            x (np.ndarray): position

        Returns:
            np.ndarray: value of potential at positions
        """
        return 3*x**2 - 1
    
    def Pot_mexico_shift(self, x):
        """*mexican* hat potential

        Args:
            x (np.ndarray): position

        Returns:
            np.ndarray: value of potential at positions
        """
        return self.scaling*(x**4 / 4 - 3*x**3 / 2 + 9*x**2 / 4)

    def dPot_mexico_shift(self, x):
        """*mexican* hat potential

        Args:
            x (np.ndarray): position

        Returns:
            np.ndarray: value of potential at positions
        """
        return self.scaling*(x**3 - 9*x**2 / 2 + 9 * x / 2)

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

    def dPhi_delta(self, x):
        """characteristic function for symmetric delta distributed

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return np.sinh(x)

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

    def Pot_harmonic(self, x):
        return x ** 2 / 2
    
    def dPot_harmonic(self, x):
        return x


class NG_memory(noise_and_potential):
    def __init__(
        self,
        a=10.0,
        b=2.0,
        sigma=1.0,
        lam=0.01,
        D1=1.0,
        D2=1.0,
        tau=1.0,
        noise="g",
        pot="m",
        dx=1e-4,
        boundary_cond=[-1, 0, 0, 0],
        number_timestep=30,
        maxtime=1.0,
        coupling=1.0,
        scaling=1.0
    ):

        noise_and_potential.__init__(self, sigma, b, scaling=scaling)

        """Initialisation of the class

        Args:
            a (float, optional): rescaled variance of characteristic funtion. Defaults to 10.0.
            b (float, optional): parameter for characteristic function: for details see individual functions. Defaults to 2.0.
            sigma (float, optional): variance for gaussian characteristic function. Defaults to 1.0.
            lam (float, optional): poisson shot noise rate. Defaults to 1.0.
            D1 (float, optional): gaussian noise variance on q. Defaults to 1.0.
            D2 (float, optional): gaussian noise variance on y. Defaults to 1.0.
            tau (float, optional): inverse OU memory timescale. Defaults to 1.0.
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
        self.tau = tau
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
            self.dpotential = self.dPot_mexico
            self.ddpotential = self.ddPot_mexico
        if pot == "ms":
            self.potential = self.Pot_mexico_shift
            self.dpotential = self.dPot_mexico_shift
        elif pot == "h":
            self.potential = self.Pot_harmonic
            self.dpotential = self.dPot_harmonic
    

        # system parameters
        self.boundary_cond = boundary_cond
        self.number_timestep = number_timestep
        self.maxtime = maxtime

        # coupling of the y dimension to the q dimension
        self.coupling = coupling

    def F(self, t, s):
        """System function

        Args:
            t (np.ndarray): time
            s (np.ndarray): system array

        Returns:
            np.ndarray: system equations
        """
        # q coordinate
        output1 = (
            s[2] * self.D1
            - self.dpotential(s[0])
            + s[1] * self.coupling
        )

        # y coordinate
        if self.lamb == 0:
            output2 = self.tau ** (-1) * (
            - s[1]
            +  self.D2 * s[3]
            )
        else:
            output2 = self.tau ** (-1) * (
                - s[1]
                +  self.D2 * s[3]
                + self.lamb * self.a * self.dPhi_delta(s[3] * self.a)
            )

        # k1
        output3 = s[2] * self.ddpotential(s[0])

        # k2
        output4 = self.tau ** (-1) * (-s[2] * self.coupling + s[3])

        # output
        return np.vstack((output1, output2, output3, output4))

    def Residuals(self, ini, fin):
        """Residuals for boundary conditions

        Args:
            ini (np.ndarray): initial values
            fin (np.ndarray): final values

        Returns:
            np.ndarray: residuals
        """
        # k = p[0]
        # k2 = p[1]
        res = np.zeros(4)
        res[0] = ini[0] - self.boundary_cond[0]
        res[1] = fin[0] - self.boundary_cond[1]
        res[2] = ini[3] - self.boundary_cond[2]
        res[3] = fin[3] - self.boundary_cond[3]
        # res[3] = ini[1] - 0
        # res[4] = ini[1] - k
        # res[5] = ini[2] - k2
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
        y[0, 0] = self.boundary_cond[0]  # q
        y[3, 0] = self.boundary_cond[2]  # k_2
        y[1, 0] = self.boundary_cond[4]  # y
        y[2, 0] = self.boundary_cond[5]  # k_1

        result = solve_bvp(self.F, self.Residuals, t, y, verbose=2)

        print(result.message)
        # print(result.p)
        # print(self.boundary_cond)
        return result, t

    def action(self, dt, q, y, k1, k2):
        """function to calculate the stochastic action using the Ito formalism

        Args:
            dt (float): time step width
            q (np.ndarray): position
            y (np.ndarray): OU-part of noise correlator
            k1 (np.ndarray): conjugate variable to q
            k2 (np.ndarray): conjugate variable to y

        Returns:
            float: value of (discrete) MSR action of system
        """
        # dimension
        m = len(q) - 1

        # derivatives
        qdot = np.zeros(m)
        ydot = np.zeros(m)

        qdot[:] = (q[1:] - q[:-1]) / dt
        ydot[:] = (y[1:] - y[:-1]) / dt

        if self.lamb == 0:
            S = (
                np.dot(k1[:-1], qdot + misc.derivative(self.potential, q[:-1]) - y[:-1])
                + np.dot(k2[:-1], self.tau * ydot +  y[:-1])
                - self.D1 / 2 * np.dot(k1[:-1], k1[:-1])
                - self.D2 / 2 * np.dot(k2[:-1], k2[:-1])
            )
        else:
        # action
            S = (
                np.dot(k1[:-1], qdot + misc.derivative(self.potential, q[:-1]) - y[:-1])
                + np.dot(k2[:-1], self.tau * ydot +  y[:-1])
                - self.D1 / 2 * np.dot(k1[:-1], k1[:-1])
                - self.D2 / 2 * np.dot(k2[:-1], k2[:-1])
                - self.lamb * np.dot(np.ones(m), self.phi(k2[:-1] * self.a))
            )

        # return the action
        return S * dt


class NG_no_memory(noise_and_potential):
    def __init__(
        self,
        a=10.0,
        b=2.0,
        sigma=1.0,
        lam=0.01,
        D1=1.0,
        D2=1.0,
        tau=1.0,
        noise="g",
        pot="m",
        dx=1e-4,
        boundary_cond=[-1, 0, 0, 0],
        number_timestep=30,
        maxtime=1.0,
        coupling=1.0,
        scaling=1.0
    ):

        noise_and_potential.__init__(self, sigma, b, scaling=scaling)

        """Initialisation of the class

        Args:
            a (float, optional): rescaled variance of characteristic funtion. Defaults to 10.0.
            b (float, optional): parameter for characteristic function: for details see individual functions. Defaults to 2.0.
            sigma (float, optional): variance for gaussian characteristic function. Defaults to 1.0.
            lam (float, optional): poisson shot noise rate. Defaults to 1.0.
            D1 (float, optional): gaussian noise variance on q. Defaults to 1.0.
            D2 (float, optional): gaussian noise variance on y. Defaults to 1.0.
            tau (float, optional): inverse OU memory timescale. Defaults to 1.0.
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
        self.tau = tau
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
            self.dpotential = self.dPot_mexico
            self.ddpotential = self.ddPot_mexico
        if pot == "ms":
            self.potential = self.Pot_mexico_shift
            self.dpotential = self.dPot_mexico_shift
        elif pot == "h":
            self.potential = self.Pot_harmonic
            self.dpotential = self.dPot_harmonic
    

        # system parameters
        self.boundary_cond = boundary_cond
        self.number_timestep = number_timestep
        self.maxtime = maxtime

        # coupling of the y dimension to the q dimension
        self.coupling = coupling

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

    def F(self, t, s):
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
            - self.dpotential(s[0])
            + self.lamb * self.a * self.dPhi_delta(s[1] * self.a)
        )
        
        # k
        output2 = s[1] * self.ddpotential(s[0])

        # output
        return np.vstack((output1, output2))

    def Residuals(self, ini, fin):
        """Residuals for boundary conditions

        Args:
            ini (np.ndarray): initial values
            fin (np.ndarray): final values

        Returns:
            np.ndarray: residuals
        """
        res = np.zeros(2)
        res[0] = ini[0] - self.boundary_cond[0]
        res[1] = fin[0] - self.boundary_cond[1]
        # res[2] = ini[1] - p[0]
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
            verbose=2,
            max_nodes=3500,
        )

        print(result.message)
        # print(result.p)
        return result, t

    def action(self, dt, q, k1):
        """function to calculate the stochastic action using the Ito formalism

        Args:
            dt (float): time step width
            q (np.ndarray): position
            k1 (np.ndarray): conjugate variable to q

        Returns:
            float: (discrete) MSR action of system
        """
        # dimension
        m = len(q) - 1

        # derivatives
        qdot = np.zeros(m)

        qdot[:] = (q[1:] - q[:-1]) / dt

        # action
        S = (
            np.dot(k1[:-1], qdot + misc.derivative(self.potential, q[:-1]))
            - self.D1 / 2 * np.dot(k1[:-1], k1[:-1])
            - self.lamb * np.dot(np.ones(m), self.phi(k1[:-1] * self.a))
        )

        # return the action
        return S * dt


# mem_G = NG_memory(lam=0, a=1, maxtime=10, noise="d", number_timestep=30)
# instanton_mem, t = mem.instanton()
# instanton_mem_G, t = mem_G.instanton()

if __name__ == "__main__":
    nsteps = 500
    boundary = [-1, 0, 0.1]
    l = 0
    a = 1

    # mem = NG_no_memory(lam=0, a=0, maxtime=100, D1=0, noise="d", pot="m", number_timestep=nsteps, boundary_cond=[-1.0, 0, 0.2])
    # inst, t = mem.instanton()


    # plt.plot(t, inst.sol(t)[0])
    # plt.show()
    #S = mem.action(t[1] - t[0], inst.sol(t)[0], inst.sol(t)[1], inst.sol(t)[2], inst.sol(t)[3])

    #print(S)

    

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


    instanton_no_mem, t = nomem.instanton()
    instanton_no_mem_t, t = nomem_t.instanton()
    instanton_no_mem_g, t = nomem_g.instanton()


    # plt.plot(t, instanton_mem.sol(t)[0], label="OU, NG")
    # plt.plot(t, instanton_mem_G.sol(t)[0], "--", label="OU, Gaussian")
    plt.plot(t, instanton_no_mem.sol(t)[0], label="non-OU, const")
    # plt.plot(t, instanton_no_mem_t.sol(t)[0], label="non-OU, trunated")
    # plt.plot(t, instanton_no_mem_e.sol(t)[0], label="non-OU, exp")
    # plt.plot(t, instanton_no_mem_ga.sol(t)[0], label="non-OU, gamma")

    plt.plot(t, instanton_no_mem_g.sol(t)[0], "--", label=r"non-OU, $\lambda=0$")

    plt.legend()
    plt.grid()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$q$")

    print(instanton_no_mem.p)
    plt.show()


    # make array of initial guesses

    
    guesses = np.array([1e-2 * i for i in range(0, 7)])

    solutions = []

    action = []

    fig2 = plt.figure()
    for i in guesses:
        boundary = [-1, 0, i]

        guess_class = NG_no_memory(
            lam=l,
            a=a,
            noise="d",
            boundary_cond=boundary,
            b=0.5,
            pot="m",
            number_timestep=nsteps,
            maxtime=10,
        )
        guess_class_l0 = NG_no_memory(
            lam=0,
            a=a,
            noise="d",
            boundary_cond=boundary,
            pot="m",
            number_timestep=nsteps,
            maxtime=10,
        )

        y, t = guess_class.instanton()
        yl, tl = guess_class_l0.instanton()
        S = guess_class.action(t[1] - t[0], y.sol(t)[0], y.sol(t)[1])
        action.append(S)
        # if (y.p[0] < 0.01 and y.p[0] > 0):
        plt.plot(t, y.sol(t)[0], label="g=%g, p=%g" % (i, y.sol(t)[1, -1]))
        # plt.plot(tl, yl.sol(tl)[0], "--")
        # plt.plot(t, y.sol(t)[1], label="g=%g, p=%g" % (i, y.sol(t)[1, -1]))
        # plt.plot(tl, yl.sol(tl)[1], "--", label="g=%g, p=%g" % (i, yl.sol(tl)[1, -1]))
        # solutions.append(y.p[0])
        #plt.plot(y.sol(t)[0], y.sol(t)[1], label="g=%g, p=%g" % (i, y.sol(t)[1, -1]))
    plt.xlabel(r"$t$")
    plt.ylabel(r"$q$")
    plt.grid()
    plt.legend()
    plt.show()

    fig_action = plt.figure()

    plt.plot(guesses, action, "-x")
    plt.ylabel(r"$S[q, k_1]$")
    plt.xlabel(r"Initial guess for $k_1$")
    plt.grid()
    plt.show()
    """
    No Memory
    Optimal paratmeter for lambda=0 seems to be in the range 0.00267308-0.00412278, so say optimum is at 0.332899
    for lambda = 0.01 and 
        -   delta noise we instead get 0.00546838-0.00699686, so say optimum at 0.00630584
        -   exponential
        -   truncated
        -   gamma
        -   Gaussian
    


    # x = np.argwhere(solutions < )

    # solutions = np.array(solutions)
    # solutions = solutions[solutions<1]

    # fig3 = plt.figure()
    # plt.hist(solutions, bins=10)
    # plt.show()


    # fig4 = plt.figure()
    # guess_class = NG_no_memory(lam=l, a=a, noise="e", boundary_cond=[-1, 0, 0.06], pot="m", number_timestep=nsteps, maxtime=10)
    # guess_class_l0 = NG_no_memory(lam=0, a=a, noise="d", boundary_cond=boundary, pot="m", number_timestep=nsteps, maxtime=10)
    # y, t = guess_class.instanton()
    # plt.plot(t, y.sol(t)[0], label="g=%g, p=%g" % (i, y.p))
    # plt.show()


    
    # ######################################### MEMORY #############################################

    so if I see this correctly, we need to introduce a 2d grid search instead of a 1D, which sounds a lot more inefficient
    


    # TODO look for more efficient methods


    nsteps = 100
    l = 0.01
    a = 10

    wow = 4

    guess1 = np.array([-1.6 * 0 for i in range(wow)])
    guess2 = np.array([1e-1 * i for i in range(wow)])

    sol_nm = NG_no_memory(a=a, lam=0, pot="m", number_timestep=200, maxtime=10)
    y_nm, t_nm = sol_nm.instanton()

    action = []

    fig5, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(15, 15))
    # plt.tight_layout()
    for i in guess1:
        for j in guess2:
            boundary = [-1.0, 0, 0, 0, i, j]
            sol = NG_memory(
                lam=l,
                a=a,
                tau=0.01,
                maxtime=10,
                noise="d",
                b=0.5,
                number_timestep=200,
                pot="m",
                boundary_cond=boundary,
                coupling=1.0,
            )

            print(boundary)

            y, t = sol.instanton()
            ax[0, 0].plot(t, y.sol(t)[0], label=r"$y=%g, k_1=%g$" % (i, j))
            ax[0, 1].plot(t, y.sol(t)[1])
            ax[1, 0].plot(t, y.sol(t)[2])
            ax[1, 1].plot(t, y.sol(t)[3])

            S = sol.action(t[1] - t[0], y.sol(t)[0], y.sol(t)[1], y.sol(t)[2], y.sol(t)[3])
            action.append(S)

    ax[0, 0].plot(t_nm, y_nm.sol(t)[0], label="AP-traj")

    ax[0, 0].set_ylabel(r"$q$")
    ax[0, 1].set_ylabel(r"$y$")
    ax[1, 0].set_ylabel(r"$k_1$")
    ax[1, 1].set_ylabel(r"$k_2$")
    ax[0, 0].grid()
    ax[0, 1].grid()
    ax[1, 0].grid()
    ax[1, 1].grid()

    ax[1, 0].set_xlabel(r"$t$")
    ax[1, 1].set_xlabel(r"$t$")
    ax[0, 0].legend(loc="lower right")

    plt.savefig("OU_traj.pdf", dpi=500, bbox_inches="tight")


    S = np.asarray(action).reshape(wow, wow)


    fig_action = plt.figure()
    plt.imshow(S)
    cbar = plt.colorbar(
        orientation="vertical",
    )
    cbar.ax.set_ylabel(r"$S[q, y, k_1, k_2]$", rotation=270)
    plt.xlabel(r"Initial guess for $y$")
    plt.ylabel(r"Initial guess for $k_1$")
    plt.grid()
    plt.savefig("OU_action.pdf", dpi=500, bbox_inches="tight")

    """