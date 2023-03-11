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


def Phi(x):
    return np.cosh(x) - 1


def Phi_exp(x):
    return x**2 / (2 * (1 - x**2))


def Phi_gamma(x, alpha=2):
    return ((1 + x) ** alpha + (1 - x) ** alpha - 2) / (2 * alpha * (alpha - 1))


def Phi_truncated(x, b=1):
    return 0.5 * x**2 + b * x**4


def F0(t, s, p, phi=Phi, kappa=1, a=10, lamb=0, D1=1, D2=1):

    # q coordinate
    output1 = +s[2] * D1 + s[1] - misc.derivative(potential, s[0], dx=1e-4)

    # y coordinate
    output2 = (
        -kappa * s[1] + D2 * s[3] + lamb * a * misc.derivative(phi, s[3] * a, dx=1e-4)
    )

    # k1
    output3 = s[2] * misc.derivative(potential, s[0], 1e-4, n=2)

    # k2
    output4 = -s[2] + kappa * s[3]

    # output
    return np.vstack((output1, output2, output3, output4))


def F(t, s, p, phi=Phi, kappa=1, a=10, lamb=1, D1=1, D2=1):

    # q coordinate
    output1 = +s[2] * D1 + s[1] - misc.derivative(potential, s[0], dx=1e-4)

    # y coordinate
    output2 = (
        -kappa * s[1] + D2 * s[3] + lamb * a * misc.derivative(phi, s[3] * a, dx=1e-4)
    )

    # k1
    output3 = s[2] * misc.derivative(potential, s[0], 1e-4, n=2)

    # k2
    output4 = -s[2] + kappa * s[3]

    # output
    return np.vstack((output1, output2, output3, output4))


def potential(q):
    return q**4 / 4 - q**2 / 2


def residuals(ini, fin, p):
    k = p[0]
    k2 = p[1]
    res = np.zeros(6)
    res[0] = ini[0] - qi
    res[1] = fin[0] - qf
    res[2] = ini[1] - yi
    res[3] = fin[1] - yf

    res[4] = ini[2] - k
    res[5] = ini[3] - k2
    return res


# initial and final conditions
qi = -1
qf = 0
yi = 0
yf = 0


# time
t = np.linspace(0, 1, 30)


# value array
y = np.zeros((4, t.size))
# apply initial guess
y[0, 0] = qi
y[1, 0] = yi

result = solve_bvp(F, residuals, t, y, p=[0.2, 0.2], verbose=0)
result0 = solve_bvp(F0, residuals, t, y, p=[0.2, 0.2], verbose=0)

print(result.message)
print(result.p)
print(result0.message)
print(result0.p)


fig = plt.figure()
plt.plot(t, result.sol(t)[0], "o-", label=r"$\phi(x) = \cosh{x}-1$")
plt.plot(t, result0.sol(t)[0], "o-", label=r"$\lambda = 0$")
plt.xlabel(r"$t$")
plt.ylabel(r"$q$")
plt.grid()
plt.legend()
plt.show()
