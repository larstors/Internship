import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.misc as misc
from numba import njit, jit
import sys

from instanton import NG_memory, NG_no_memory

class noise_and_potential:
    def __init__(self, sigma=1, b=0.5):
        self.sigma = sigma
        self.b = b

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


class transform(noise_and_potential):
    def __init__(self, lambda_, a, tau, D1, D2, N, noise="d", pot="m", tmax=10, sigma=1, b=1):
        noise_and_potential.__init__(self, sigma, b)
        # parameters
        self.a = a
        self.lambda_ = lambda_
        self.tau = tau
        self.D1 = D1
        self.D2 = D2

    
        # characteristic function of noise
        if noise == "g":
            self.phi = noise_and_potential.Phi_gauss
        elif noise == "d":
            self.phi = noise_and_potential.Phi_delta
        elif noise == "e":
            self.phi = noise_and_potential.Phi_exp
        elif noise == "g":
            self.phi = noise_and_potential.Phi_gamma
        elif noise == "t":
            self.phi = noise_and_potential.Phi_truncated

        # potential
        if pot == "m":
            self.potential = noise_and_potential.Pot_mexico

        # initial guess length
        self.N = N

        #max time
        self.tmax = tmax

    def Psi(self, k2, k1=0):

        return self.D1 * k1 ** 2 / 2 + self.D2 * k2 ** 2 * 0.5 + self.lambda_ * self.phi(self.a * k2)

    def Opt_Func(self, k2, f1):
        return f1 - misc.derivative(self.Psi, k2)

    def Legendre_transform(self, initial, qDot: np.ndarray, q: np.ndarray, yDot: np.ndarray, y: np.ndarray):

        f0 = qDot + self.dPhi_delta(q) - y
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

    def init_constraint_conjugate(self, t):
        """Function for initial constraint that k2(0) = -1

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for initial constraint
        """
        return t[3 * self.N] 
    
    def final_constraint_conjugate(self, t):
        """Function for final constraint that k2(tf) = 0

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for final constraint
        """
        return t[-1]

    def minimize(self):
        system = np.zeros(2 * self.N)

        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}]

        optimum = opt.minimize(self.MSR_action, x0=system, constraints=constraint, options={'disp': True})

        return optimum

    def action(self, init_values):
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

        q = init_values[:self.N]
        y = init_values[self.N:2*self.N]
        k1 = init_values[2*self.N:3*self.N]
        k2 = init_values[3*self.N:]

        dt = self.tmax / self.N
        # dimension
        m = len(q) - 1

        # derivatives
        qdot = np.zeros(m)
        ydot = np.zeros(m)

        qdot[:] = (q[1:] - q[:-1]) / dt
        ydot[:] = (y[1:] - y[:-1]) / dt

        # action
        S = (
            np.dot(k1[:-1], qdot + misc.derivative(self.potential, q[:-1]) - y[:-1])
            + np.dot(k2[:-1], self.tau * ydot +  y[:-1])
            - self.D1 / 2 * np.dot(k1[:-1], k1[:-1])
            - self.D2 / 2 * np.dot(k2[:-1], k2[:-1])
            - self.lambda_ * np.dot(np.ones(m), self.phi(k2[:-1] * self.a))
        )

        # return the action
        return S
    
    def minimize_full(self):
        system = np.zeros(4 * self.N)

        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}, {"type":"eq", "fun":self.init_constraint_conjugate}, {"type":"eq", "fun":self.final_constraint_conjugate}]

        optimum = opt.minimize(self.action, x0=system, constraints=constraint, options={'disp': True})

        return optimum

class transform_nomemory:
    def __init__(self, lambda_, a, D, N, noise="d", pot="m", tmax=10):
        
        # parameters
        self.a = a
        self.lambda_ = lambda_
        self.D = D
    
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

    def dPot_mexico(self, x):
        """*mexican* hat potential

        Args:
            x (np.ndarray): position

        Returns:
            np.ndarray: value of potential at positions
        """
        return x**3 - x

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

    def Psi(self, k):

        return  self.D * k ** 2 * 0.5 #+ self.lambda_ * self.phi(self.a * k)

    def Opt_Func(self, k, f):
        if self.lambda_ == 0:
            return f - self.D * k
        else:
            return f - self.D * k - self.lambda_ * self.a * self.dPhi_delta(self.a * k) 

    def Legendre_transform(self, initial, qDot: np.ndarray, q: np.ndarray):

        f = qDot + self.dPot_mexico(q) 

        k = opt.fsolve(self.Opt_Func, initial, args=f)

        return k

    def MSR_action(self, init_values):
        
        delta_t = self.tmax / self.N

        q = init_values[: self.N]

        qdot = (q[1:] - q[:-1]) / delta_t

        k = self.Legendre_transform(-np.ones_like(qdot)*1e-4, qdot, q[:-1])

        S = 0
        
        for i in range(self.N - 1):
            if self.lambda_ == 0:
                S += k[i] * (qdot[i] + self.dPot_mexico(q[i])) - self.D / 2 * k[i] ** 2
            else:
                S += k[i] * (qdot[i] + self.dPot_mexico(q[i])) - self.D / 2 * k[i] ** 2 - self.lambda_ * self.phi(self.a * k[i])
        return S * delta_t

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
        return t[self.N-1]

    def minimize(self):
        system = np.zeros(self.N)#np.linspace(-1, 0, 2 * self.N)

        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}]

        optimum = opt.minimize(self.MSR_action, x0=system, constraints=constraint, options={'disp': True, 'maxiter': 400})

        return optimum

    def minimize_full(self):
        system = np.linspace(-1, 0, self.N)  

        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}]

        optimum = opt.minimize(self.OM_action, x0=system, constraints=constraint, options={'disp': True, 'maxiter': 400})

        return optimum

    def action(self, init_values):
        """function to calculate the stochastic action using the Ito formalism

        Args:
            dt (float): time step width
            q (np.ndarray): position
            k1 (np.ndarray): conjugate variable to q

        Returns:
            float: (discrete) MSR action of system
        """
        q = init_values[:self.N]
        k1 = init_values[self.N:]
        dt = self.tmax / self.N
        # dimension
        m = len(q) - 1

        # derivatives
        qdot = np.zeros(m)

        qdot[:] = (q[1:] - q[:-1]) / dt

        # action
        S = 0
        for i in range(m):
            S += k1[i] * (qdot[i] + self.dPot_mexico(q[i])) - self.D / 2 * k1[i] ** 2

        # return the action
        return S * dt

    def OM_action(self, init_values):
        """function to calculate the stochastic action using the Ito formalism

        Args:
            dt (float): time step width
            q (np.ndarray): position
            k1 (np.ndarray): conjugate variable to q

        Returns:
            float: (discrete) MSR action of system
        """
        q = init_values[:self.N]
        dt = self.tmax / self.N
        # dimension
        m = len(q) - 1

        # derivatives
        qdot = np.zeros(m)

        qdot[:] = (q[1:] - q[:-1]) / dt

        # action
        S = 0

        for i in range(m):
            S += self.D**(-1) * (qdot[i] + self.dPot_mexico(q[i])) ** 2 / 2

        # return the action
        return S * dt

    

#if __name__ == "main":


# t = transform(0.0, 10, 0.0001, 1, 1, N, "d", "m", 10)
# op = t.minimize()
# #op1 = t.minimize_full()

# print(op.message)
# #print(op.x[:N])

# time = np.linspace(0, 10, N, endpoint=True)

# fig = plt.figure()
# plt.plot(time, op.x[:N])

# #plt.plot(time, op1.x[:N])
# plt.xlabel(r"$t$")
# plt.ylabel(r"$q$")
# plt.show()
N = 100
tmax=10
dt = tmax/N

# t = transform_nomemory(lambda_=0, a=10, D=1, N=N, noise="d", pot="m", tmax=tmax)
# op = t.minimize_full()
# res = t.minimize()

# t = transform_nomemory(lambda_=0.01, a=10, D=1, N=N, noise="d", pot="m", tmax=tmax)
# mem = t.minimize()

# time = np.linspace(0, tmax, N, endpoint=True)

# fig2, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
# ax[0].plot(time, op.x[:N], label="OM minimization")
# ax[0].plot(time, res.x[:N], label=r"MSR Legendre minimization, $\lambda=0$")
# ax[0].plot(time, mem.x[:N], label=r"MSR Legendre minimization, $\lambda=0.01$")
# ax[0].set_xlabel(r"$t$")
# ax[0].set_ylabel(r"$q$")
# ax[0].set_title("Comparison of minimizations")
# ax[0].legend()
# ax[0].grid()

# ax[1].plot(op.x[:N-1], (op.x[1:N] - op.x[:N-1])/dt, label="OM minimization")
# ax[1].plot(res.x[:N-1], (res.x[1:N] - res.x[:N-1])/dt, label=r"MSR Legendre minimization, $\lambda=0$")
# ax[1].plot(mem.x[:N-1], (mem.x[1:N] - mem.x[:N-1])/dt, label=r"MSR Legendre minimization, $\lambda=0.01$")
# ax[1].set_ylabel(r"$\frac{q(t+dt)-q(t)}{dt}$")
# ax[1].set_xlabel(r"$q$")
# ax[1].set_title("Comparison of minimizations")
# #ax[1].legend()
# ax[1].grid()




# plt.savefig("minimization_comparison_phase.pdf", dpi=500, bbox_inches="tight")


# print(op.message)
# print("The minimization of the markovian OM yields\t", op.fun, "\nThe fraction with the difference in potential (should yield 1) is\t", op.fun / (2*(t.Pot_mexico(0) - t.Pot_mexico(-1))))
# print("Doing Legendre transform yields\t", res.fun, "\nAnd the fraction (again 1?) yields", res.fun/(2*(t.Pot_mexico(0) - t.Pot_mexico(-1))))




lam = np.logspace(start=-8, stop=2, base=10, num=20)

S_norm_OM  = []
S_norm_MSR = []

for l in lam:
    S_G = 2 * (0 - (-1)) / (1 + l * 10 ** 2)
    t = transform_nomemory(lambda_=l, a=10, D=1, N=N, noise="d", pot="m", tmax=tmax)
    #op = t.minimize_full()
    res = t.minimize()
    #S_norm_OM.append(op.fun/S_G)
    S_norm_MSR.append(res.fun/S_G)

fig3 = plt.figure()
#plt.plot(lam, S_norm_OM, label="OM")
plt.plot(lam, S_norm_MSR, label="MSR")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$S_\mathrm{norm}$")
plt.grid()
plt.legend()
plt.savefig("S_norm.pdf", dpi=500, bbox_inches="tight")
