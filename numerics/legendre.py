import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.misc as misc
from numba import njit, jit
import sys
import math
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
#import torch
from instanton import NG_memory, NG_no_memory

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
    
    def dPhi_exp(self, x):
        return x / ((1 - x ** 2) ** 2)

    def Phi_gamma(self, x):
        """characteristic function for gamma distribution

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return ((1 + x) ** self.b + (1 - x) ** self.b - 2) / (2 * self.b * (self.b - 1))

    def dPhi_gamma(self, x):
        return ((1 + x) ** (self.b-1) + (1 - x) ** (self.b-1) - 2) / (2 * self.b * (self.b - 1))


    def Phi_truncated(self, x):
        """characteristic function for truncated distribution

        Args:
            x (float): position at which to evaluate the characteric function

        Returns:
            float: value of characteristic function
        """
        return 0.5 * x**2 + self.b * x**4
    
    def dPhi_truncated(self, x):
        return x * 4 * self.b * x ** 3

    def Pot_harmonic(self, x):
        return x ** 2 / 2
    
    def dPot_harmonic(self, x):
        return x

    def init_constraint(self, t):
        """Function for initial constraint that q(0) = -1

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for initial constraint
        """
        return t[0] -self.const_i
    
    def final_constraint(self, t):
        """Function for final constraint that q(tf) = 0

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for final constraint
        """
        return t[self.N - 1] - self.const_f

    def velocity_constraint(self, t):
        return t[self.N - 1] - t[self.N - 2]
    def velocity_constraint_initial(self, t):
        return t[1] - t[0]
    
class transform(noise_and_potential):
    def __init__(self, lambda_, a, tau, D1, D2, N, noise="d", pot="m", tmax=10, sigma=1, b=1, const_i=-1, const_f=0, scaling=1.0):
        noise_and_potential.__init__(self, sigma, b, scaling=scaling)
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
            self.dpotential = self.dPot_mexico
        if pot == "ms":
            self.potential = self.Pot_mexico_shift
            self.dpotential = self.dPot_mexico_shift
        elif pot == "h":
            self.potential = self.Pot_harmonic
            self.dpotential = self.dPot_harmonic

        # initial guess length
        self.N = N

        #max time
        self.tmax = tmax
        
        # constraints
        self.const_i = const_i
        self.const_f = const_f


    def Opt_Func(self, k2, f1):
        """Function for calculating Legendre transform numerically

        Args:
            k2 (np.ndarray): array to solve for
            f1 (np.ndarray): input

        Returns:
            np.ndarray: values for equation
        """
        if self.lambda_ == 0:
            return f1 - self.D2 * k2
        else:
            return f1 - self.D2 * k2 - self.lambda_ * self.a  * self.dPhi_delta(self.a * k2)

    def Legendre_transform(self, initial, qDot: np.ndarray, q: np.ndarray, yDot: np.ndarray, y: np.ndarray):

        f0 = qDot + self.dpotential(q) - y


        
        if self.tau == 0:
            f1 = y
        else:
            f1 = self.tau * yDot + y 

        if self.D1 != 0:
            k1 = f0 / self.D1
        else:
            k1 = np.zeros_like(f0)

        k2 = opt.fsolve(self.Opt_Func, initial, args=f1)

        return k1, k2

    def MSR_action(self, init_values, prop=(-1e-6)):
        
        delta_t = self.tmax / self.N

        q = init_values[:self.N]

        qdot = (q[1:] - q[:-1]) / delta_t
        y = qdot + self.dpotential(q[:-1])
        ydot = (y[1:] - y[:-1]) / delta_t

        k1, k2 = self.Legendre_transform(np.ones_like(ydot)*prop, qdot[:-1], q[:-2], ydot, y[:-1])

        S = 0
        
        for i in range(self.N - 2):
            if self.lambda_ != 0:
                S -= self.lambda_ * self.phi(self.a * k2[i])
            S +=  k2[i] * (self.tau * ydot[i] + y[i]) - self.D2 / 2 * k2[i] ** 2 # + k1[i] * (qdot[i] + self.dpotential(q[i]) - y[i]) - self.D1 / 2 * k1[i] ** 2 

        return S * delta_t

    def init_constraint(self, t):
        """Function for initial constraint that q(0) = -1

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for initial constraint
        """
        return t[0] -self.const_i
    
    def final_constraint(self, t):
        """Function for final constraint that q(tf) = 0

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for final constraint
        """
        return t[self.N - 1] - self.const_f

    def velocity_constraint(self, t):
        return t[self.N - 1] - t[self.N - 2]
    def velocity_constraint_initial(self, t):
        return t[1] - t[0]

    def minimize(self, in_cond = np.zeros(2 * 100), a=-1e-4, maxiter=1000):
        system = np.zeros(2*self.N)
        if self.tau>0:
            lam0 = min(1, 1/self.tau)
            lam_1 = min(2, 1/self.tau)
            lam = (lam_1 + lam0) / 2
            system[:self.N] = np.linspace(self.const_i, self.const_f, self.N) #in_cond
            system[self.N:] = -lam * system[:self.N] * (1 + system[:self.N])
        else:
            lam0 = 1
            lam_1 = 2
            lam = (lam_1 + lam0) / 2
            system[:self.N] = np.linspace(self.const_i, self.const_f, self.N) #in_cond
            system[self.N:] = -lam * system[:self.N] * (1 + system[:self.N])
            
        if self.lambda_ > 10 and False:
            constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}, {"type":"eq", "fun":self.velocity_constraint}]
        elif self.potential == self.Pot_harmonic or self.potential == self.Pot_mexico_shift or self.potential == self.Pot_mexico:
            constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}, {"type":"eq", "fun":self.velocity_constraint}, {"type":"eq", "fun":self.velocity_constraint_initial}]
        else:
            constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}]

        optimum = opt.minimize(self.MSR_action, x0=system, args=(a), constraints=constraint, options={'disp': True,  'maxiter': maxiter})

        return optimum

class TransformLargeTau(noise_and_potential):
    def __init__(self, lambda_, a, tau, D1, D2, N, noise="d", pot="m", tmax=10, sigma=1, b=1, const_i=-1, const_f=0, scaling=1.0):
        noise_and_potential.__init__(self, sigma, b, scaling=scaling)
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
            self.dpotential = self.dPot_mexico
        if pot == "ms":
            self.potential = self.Pot_mexico_shift
            self.dpotential = self.dPot_mexico_shift
        elif pot == "h":
            self.potential = self.Pot_harmonic
            self.dpotential = self.dPot_harmonic

        # initial guess length
        self.N = N

        #max time
        self.tmax = tmax
        
        # constraints
        self.const_i = const_i
        self.const_f = const_f

    def Opt_Func(self, k2, f1):
        return f1 - self.D2 * k2 - self.lambda_ * self.a  * self.dPhi_delta(self.a * k2)

    def Legendre_transform(self, initial, qddot: np.ndarray, qDot: np.ndarray, q: np.ndarray):

        if self.lambda_ > 0:
            f1 = (1 / self.tau * (qddot + qDot[:-1]) + self.ddPot_mexico(q[:-2]) * qDot[:-1] + self.dPot_mexico(q[:-2]))
            k2 = opt.fsolve(self.Opt_Func, initial, args=f1)
        else:
            k2 = 1/self.D2 * (1 / self.tau * (qddot + qDot[:-1]) + self.ddPot_mexico(q[:-2]) * qDot[:-1] + self.dPot_mexico(q[:-2]))
        return k2

    def MSR_action_position_space(self):

        S = 0
        # optimize wrt v = qdot

        for i in range(self.N-1):
            vprime = (v[i+1]-v[i])/dq
            S+= 1 / self.tau * (vprime + 1) + self.ddPot_mexico(q[i])


        return S

    def MSR_action(self, init_values, guess):
        delta_t = self.tmax = self.N
        q = init_values
        qdot = (q[1:] - q[:-1]) / delta_t
        qddot = (qdot[1:] - qdot[:-1]) / delta_t

        k2 = self.Legendre_transform(guess, qddot, qdot, q)

        S = 0
        for i in range(self.N-2):
            x = q[i]#(q[i+2] + 2*q[i+1] + q[i]) / 4
            v = (qdot[i+1]+qdot[i])/2

            S += k2[i] * (1 / self.tau * (qddot[i] + v) + self.ddPot_mexico(x) * v + self.dPot_mexico(x) - self.D2 / 2 * k2[i])
            if self.lambda_ > 0:
                S -= self.lambda_ * self.Phi_delta(self.a * k2[i])

        return delta_t * self.tau * S

    def minimize(self, in_cond, guess, a=-1e-4, maxiter=1000):
        system = np.zeros(self.N)
        lam0 = min(1, 1/self.tau)
        lam_1 = min(2, 1/self.tau)
        lam = (lam_1 + lam0) / 2
        system = np.linspace(self.const_i, self.const_f, self.N) #in_cond

        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}, {"type":"eq", "fun":self.velocity_constraint}, {"type":"eq", "fun":self.velocity_constraint_initial}]

        optimum = opt.minimize(self.MSR_action, x0=system, args=(guess), constraints=constraint, options={'disp': True,  'maxiter': maxiter})

        return optimum


class transform_adjust(noise_and_potential):
    def __init__(self, lambda_, a, tau, D1, D2, N, noise="d", pot="m", tmax=10, sigma=1, b=1, const_i=-1, const_f=0, scaling=1.0):
        noise_and_potential.__init__(self, sigma, b, scaling=scaling)
        # parameters
        self.a = a
        self.lambda_ = lambda_
        self.tau = tau
        self.D1 = D1
        self.D2 = D2
        print("lambda \t\t %e\n a \t\t %f \n tau \t\t %f \n D \t\t %f \n tmax \t\t %f\n N \t\t %d\n" % (self.lambda_, self.a, self.tau, self.D2, tmax, N))
        # characteristic function of noise
        if noise == "g":
            self.phi = self.Phi_gauss
        elif noise == "d":
            print("Noise: symmetric delta at +- 1")
            self.phi = self.Phi_delta
            self.dphi = self.dPhi_delta
        elif noise == "e":
            print("Noise: symmetric exponential with characteristic scale 1")
            self.phi = self.Phi_exp
            self.dphi = self.dPhi_exp
        elif noise == "g":
            print("Noise: gamma distributed with scale %f" % self.b)
            self.phi = self.Phi_gamma
            self.dphi = self.dPhi_gamma
        elif noise == "t":
            print("Noise: truncated coefficient %f" % self.b)
            self.phi = self.Phi_truncated
            self.dphi = self.dPhi_truncated

        # potential
        if pot == "m":
            print("Potential: quartic potential q^4/4 - q^2/2\n")
            self.potential = self.Pot_mexico
            self.dpotential = self.dPot_mexico
        if pot == "ms":
            self.potential = self.Pot_mexico_shift
            self.dpotential = self.dPot_mexico_shift
        elif pot == "h":
            self.potential = self.Pot_harmonic
            self.dpotential = self.dPot_harmonic

        # initial guess length
        self.N = N

        #max time
        self.tmax = tmax
        
        # constraints
        self.const_i = const_i
        self.const_f = const_f

        # setting up look-up table for Legendre Transform
        # dummy for m_2 = tau * ydot + y
        m_2 = np.linspace(-2000, 2000, 500)
        initial = np.ones_like(m_2) * (1e-4)
        #initial = self.solver(m_2=m_2, initial=initial)
        
        # function values
        #Leg = self.solver(m_2=m_2, initial=initial, lambda_=self.lambda_, a=self.a)
        Leg = self.solver2(m_2=m_2, initial=initial, lambda_=self.lambda_, a=self.a)
    
        # set up interpolation
        self.Legendre = interp1d(m_2, Leg, kind="linear")

    def solver(self, m_2, initial, lambda_=0.01, a=10):

        def Eq(k, m, lam, a):
            return m - self.D2 * k - lambda_ * a * self.dphi(a * k)
        
        k = opt.root(Eq, initial, args=(m_2, lambda_, a), method="lm").x

        return k
    
    def solver2(self, m_2, initial, lambda_=0.01, a=10, b=1/2):
        def Eq2(k, m, a, lambda_):
            return  k * m- self.D2/2 * k**2 - lambda_ * (np.cosh(a*k) - 1)

        k = np.linspace(-200, 200, 4000)

        K = []
        for M in m_2:
            e = Eq2(k, M, a, lambda_)
            f = np.argmax(e)
            K.append(k[f])
    

        return K

    def Legendre_transform(self, yDot: np.ndarray, y: np.ndarray):
        f1 = self.tau * yDot + y 
        k2 = self.Legendre(f1) #opt.fsolve(self.Opt_Func, initial, args=f1)
        return k2

    def MSR_action(self, init_values):
        
        delta_t = self.tmax / self.N

        q = init_values[:self.N]
        qdot = (q[1:] - q[:-1]) / delta_t
        y = qdot + self.dpotential(q[:-1])
        ydot = (y[1:] - y[:-1]) / delta_t
        k2 = self.Legendre_transform(ydot, y[:-1])

        S = 0
        
        for i in range(self.N - 2):
            if self.lambda_ != 0:
                S -= self.lambda_ * self.phi(self.a * k2[i])
            S +=  k2[i] * (self.tau * ydot[i] + y[i]) - self.D2 / 2 * k2[i] ** 2 
        return S * delta_t

    def init_constraint(self, t):
        """Function for initial constraint that q(0) = -1

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for initial constraint
        """
        return t[0] - self.const_i
    
    def final_constraint(self, t):
        """Function for final constraint that q(tf) = 0

        Args:
            t (np.ndarray): array of q and y, so has 2N dimensions, where q=t[:N] and y=t[N:]

        Returns:
            float: equation for final constraint
        """
        return t[self.N - 1] - self.const_f

    def velocity_constraint(self, t):
        return t[self.N - 1] - t[self.N - 2]
    def velocity_constraint_initial(self, t):
        return t[1] - t[0]

    def minimize(self, in_cond = np.zeros(2 * 100), maxiter=1000):
        system = np.zeros(self.N)
        if in_cond[0] == 0:
            system[:self.N] = np.linspace(self.const_i, self.const_f, self.N) #in_cond
        else:
            system = in_cond
        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}, {"type":"eq", "fun":self.velocity_constraint}, {"type":"eq", "fun":self.velocity_constraint_initial}]

        optimum = opt.minimize(self.MSR_action, x0=system, constraints=constraint, options={'disp': True,  'maxiter': maxiter})

        return optimum

class transform_adjust_2(noise_and_potential):
    def __init__(self, lambda_, a, tau, D1, D2, N, noise="d", pot="m", tmax=10, sigma=1, b=1, const_i=-1, const_f=0, scaling=1.0):
        noise_and_potential.__init__(self, sigma, b, scaling=scaling)
        # parameters
        self.a = a
        self.lambda_ = lambda_
        self.tau = tau
        self.D1 = D1
        self.D2 = D2
        print("lambda \t|\t %e\n a \t|\t %f \n tau \t|\t %f \n D \t|\t %f \n tmax \t|\t %f\n N \t|\t %d\n" % (self.lambda_, self.a, self.tau, self.D2, tmax, N))
        # characteristic function of noise

        print(const_f, const_i)
        if noise == "d":
            print("Noise: symmetric delta at +- 1")
            self.phi = self.Phi_delta
            self.dphi = self.dPhi_delta
        elif noise == "e":
            print("Noise: symmetric exponential with characteristic scale 1")
            self.phi = self.Phi_exp
            self.dphi = self.dPhi_exp
        elif noise == "g":
            print("Noise: gamma distributed with scale %f" % self.b)
            self.phi = self.Phi_gamma
            self.dphi = self.dPhi_gamma
        elif noise == "t":
            print("Noise: truncated coefficient %f" % self.b)
            self.phi = self.Phi_truncated
            self.dphi = self.dPhi_truncated

        # potential
        if pot == "m":
            print("Potential: quartic potential q^4/4 - q^2/2\n")
            self.potential = self.Pot_mexico
            self.dpotential = self.dPot_mexico
        if pot == "ms":
            self.potential = self.Pot_mexico_shift
            self.dpotential = self.dPot_mexico_shift
        elif pot == "h":
            self.potential = self.Pot_harmonic
            self.dpotential = self.dPot_harmonic

        # initial guess length
        self.N = N

        #max time
        self.tmax = tmax
        
        # constraints
        self.const_i = const_i
        self.const_f = const_f

        # setting up look-up table for Legendre Transform
        # dummy for m_2 = tau * ydot + y
        m_2 = np.linspace(-40000, 40000, 40000)
        initial = np.ones_like(m_2) * (1e-4)
        #initial = self.solver(m_2=m_2, initial=initial)
        
        # function values
        #Leg = self.solver(m_2=m_2, initial=initial, lambda_=self.lambda_, a=self.a)
        Leg = self.solver2(m_2=m_2, initial=initial, lambda_=self.lambda_, a=self.a, noise=noise)
    
        # set up interpolation
        self.Legendre = interp1d(m_2, Leg, kind="linear")



    def solver(self, m_2, initial, lambda_=0.01, a=10):

        def Eq(k, m, lam, a):
            return m - self.D2 * k - lambda_ * a * self.dphi(a * k)
        
        k = opt.root(Eq, initial, args=(m_2, lambda_, a), method="lm").x

        return k
    
    def solver2(self, m_2, initial, lambda_=0.01, a=10, b=1/2, noise="d"):
        def Eq2(k, m, a, lambda_):
            #print(k, np.cosh(a*k))
            return  k * m- self.D2/2 * k**2 - lambda_ * self.phi(a*k)#(np.cosh(a*k) - 1)

        k = np.linspace(-40, 40, 60000)
        if noise == "g" or noise == "e":
            k = np.linspace(-1/a + 1e-5, 1/a - 1e-5, 60000)
        
        K = []
        for M in m_2:
            e = Eq2(k, M, a, lambda_)
            f = np.argmax(e)
            K.append(k[f])
    

        return K

    def Legendre_transform(self, yDot: np.ndarray, y: np.ndarray):
        f1 = self.tau * yDot + y 
        k2 = self.Legendre(f1) #opt.fsolve(self.Opt_Func, initial, args=f1)
        return k2

    def MSR_action(self, q):
        
        delta_t = self.tmax / self.N

        q[0] = self.const_i
        q[1] = self.const_i
        q[-2] = self.const_f
        q[-1] = self.const_f
        y = np.zeros(self.N-1)
        ydot = np.zeros(self.N-1)
        qdot = (q[1:] - q[:-1]) / delta_t
        y[:] = qdot + self.dpotential(q[:-1])
        ydot[:-2] = (y[1:-1] - y[:-2]) / delta_t
        #print(max(np.abs(self.tau * ydot[:] + y[:])))
        k2 = self.Legendre(self.tau * ydot[:] + y[:])

        #karra = np.linspace(-10, 10, 100)
        #k2 = [karra[np.argmax([k*y[i] - self.D2/2 * k**2 - self.lambda_*self.phi(self.a*k) for k in karra])] for i in range(len(y))]
        
        S = 0
        
        for i in range(self.N-1):
            S +=  k2[i] * (self.tau * ydot[i] + y[i]) - self.D2 / 2 * k2[i] ** 2 - self.lambda_ * self.phi(self.a * k2[i]) 
        return S * delta_t


    def minimize(self, in_cond = np.zeros(2 * 100), maxiter=1000):
        system = np.linspace(self.const_i, self.const_f, self.N) #in_cond
        
        optimum = opt.minimize(self.MSR_action, x0=system, options={'disp': True})

        return optimum



class NG_memory(noise_and_potential):
    def __init__(
        self,
        a=10.0,
        b=2.0,
        sigma=1.0,
        lam=0.01,
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
        self.D2 = D2
        self.tau = tau

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
        # qdot coordinate
        output1 = s[0]
        # qddot
        output2 = 1/self.tau * ((- self.tau * self.ddPot_mexico(s[0]) + 1)*s[1] - self.dPot_mexico(s[0]) + self.D2*s[2] + self.a*self.lamb * self.dPhi_delta(self.a*s[2]))


        # k2dot
        output3 = s[2]
        # k2ddot
        output4 = 1/self.tau * ((1 + self.tau * self.ddPot_mexico(s[0])) * s[3] - self.ddPot_mexico(s[0])*s[2])

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
        res[2] = ini[1] - self.boundary_cond[2]
        res[3] = fin[1] - self.boundary_cond[3]
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

    def MSR_action(self, init_values, a):
        
        delta_t = self.tmax / self.N

        q = init_values[: self.N]

        qdot = (q[1:] - q[:-1]) / delta_t

        k = self.Legendre_transform(np.ones(self.N-1) * a, qdot, q[:-1])

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

    def minimize(self, in_cond = np.zeros(100), a=-1e-4):
        system = in_cond

        constraint = [{"type":"eq", "fun":self.init_constraint}, {"type":"eq", "fun":self.final_constraint}]

        optimum = opt.minimize(self.MSR_action, x0=system, args=(a), constraints=constraint, options={'disp': True, 'maxiter': 400})

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
    
    

if __name__ == "__main__":


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
        t = transform_nomemory(lambda_=l, a=10, D=1, N=N, noise="d", pot="m", tmax=tmax)
        S_G = 2 * (t.Pot_mexico(0) - t.Pot_mexico(-1)) / (1 + l * 10 ** 2)

        #op = t.minimize_full()
        res = t.minimize()
        #S_norm_OM.append(op.fun/S_G)
        S_norm_MSR.append(res.fun/S_G)

    fig3 = plt.figure()
    #plt.plot(lam, S_norm_OM, label="OM")
    plt.plot(lam, S_norm_MSR, label="MSR")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$S_\mathrm{norm}$")
    plt.xscale("log")
    plt.grid()
    plt.legend()
    plt.axis([min(lam), max(lam), -0.1, 1.1])
    plt.savefig("S_norm.pdf", dpi=500, bbox_inches="tight")
