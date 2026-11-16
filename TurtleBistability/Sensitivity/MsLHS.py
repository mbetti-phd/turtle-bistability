from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyrates import CircuitTemplate, NodeTemplate, OperatorTemplate
from pycobi import ODESystem
from scipy.integrate import solve_ivp
import sympy as sp
from scipy.stats import qmc
from scipy.signal import find_peaks

def a2 (A,J,P,mu,p):

    beta = p['betta']
    k = p['k_pred']
    phi = p['phi']
    v = p['v']
    delta = p['delta']
    c = p['c']
    K = p['K_sex']
    n = p['n']
    r = p['r']
    m = p['m']

    a2 = (-(v * (-(((delta + m) * mu + delta * m) * v - phi * (mu + m)) * k ** 2 * P ** 2 - 2 * (((delta + m) * mu + delta * m) * v - phi * (mu + m)) * k * P + ((beta - delta - m) * mu + m * (beta - delta)) * v + phi * (mu + m)) * J ** 2 + (-2 * (((delta + m) * mu + delta * m) * v - phi * (mu + m) / 2) * k ** 2 * P ** 2 - 4 * (((delta + m) * mu + delta * m) * v - phi * (mu + m) / 2) * k * P + ((2 * beta - 2 * delta - 2 * m) * mu + 2 * m * (beta - delta)) * v + phi * (mu + m)) * J - k ** 2 * r * (mu + delta) * P ** 3 - (((delta + m) * mu + delta * m) * k + 2 * r * (mu + delta)) * k * P ** 2 + (((-2 * delta - 2 * m) * mu - 2 * delta * m) * k - r * (mu - beta + delta)) * P + (beta - delta - m) * mu + m * (beta - delta)) * n ** 2 * A ** 4 - 2 * (v * (-(((delta + m) * mu + delta * m) * v - phi * (mu + m)) * k ** 2 * P ** 2 - 2 * (((delta + m) * mu + delta * m) * v - phi * (mu + m)) * k * P + ((beta - delta - m) * mu + m * (beta - delta)) * v + phi * (mu + m)) * J ** 2 + (-2 * (((delta + m) * mu + delta * m) * v - phi * (mu + m) / 2) * k ** 2 * P ** 2 - 4 * (((delta + m) * mu + delta * m) * v - phi * (mu + m) / 2) * k * P + ((2 * beta - 2 * delta - 2 * m) * mu + 2 * m * (beta - delta)) * v + phi * (mu + m)) * J - k ** 2 * r * (mu + delta) * P ** 3 - (((delta + m) * mu + delta * m) * k + 2 * r * (mu + delta)) * k * P ** 2 + (((-2 * delta - 2 * m) * mu - 2 * delta * m) * k - r * (mu - beta + delta)) * P + (beta - delta - m) * mu + m * (beta - delta)) * K ** 2 * n * A ** 2 - 2 * K ** 2 * c * m * (P * k + 1) ** 2 * (J * v + 1) ** 2 * A - (v * (-(((delta + m) * mu + delta * m) * v - phi * (mu + m)) * k ** 2 * P ** 2 - 2 * (((delta + m) * mu + delta * m) * v - phi * (mu + m)) * k * P + ((beta - delta - m) * mu + m * (beta - delta)) * v + phi * (mu + m)) * J ** 2 + (-2 * (((delta + m) * mu + delta * m) * v - phi * (mu + m) / 2) * k ** 2 * P ** 2 - 4 * (((delta + m) * mu + delta * m) * v - phi * (mu + m) / 2) * k * P + ((2 * beta - 2 * delta - 2 * m) * mu + 2 * m * (beta - delta)) * v + phi * (mu + m)) * J - k ** 2 * r * (mu + delta) * P ** 3 - (((delta + m) * mu + delta * m) * k + 2 * r * (mu + delta)) * k * P ** 2 + (((-2 * delta - 2 * m) * mu - 2 * delta * m) * k - r * (mu - beta + delta)) * P + (beta - delta - m) * mu + m * (beta - delta)) * K ** 4) / (A ** 2 * n + K ** 2) ** 2 / (J * v + 1) ** 2 / (P * k + 1) ** 2

    return a2

def turtle_ode(t, y, p):
    P, J, A = y
    betta = p['betta']
    k_pred = p['k_pred']
    phi = p['phi']
    v = p['v']
    delta = p['delta']
    c = p['c']
    K_sex = p['K_sex']
    n = p['n']
    r = p['r']
    m = p['m']
    mu_mod = p['mu_mod']
    
    dP = (betta * P) / (1.0 + k_pred * P) + (phi * J / (1.0 + v * J)) * P - delta * P
    dJ = (c / (K_sex**2)) * (A**2 / (1.0 + n * (A / K_sex)**2)) - (r * J / (1.0 + v * J)) * P - m * J
    dA = m * J - mu_mod * A
    
    return [dP, dJ, dA]


def get_steady_state(params):
    params_ivp = params

    y0 = [13.77, 1.55, 10.33] 

    t_span = (0, 1000)  
    t_eval = np.linspace(0, 1000, 10000)


    sol = solve_ivp(turtle_ode, t_span, y0, args=(params_ivp,), t_eval=t_eval, method='RK45')


    P_ss = sol.y[0, -1]
    J_ss = sol.y[1, -1]
    A_ss = sol.y[2, -1]

    return P_ss, J_ss, A_ss

def M(xi,p):

    beta = p['betta']
    k = p['k_pred']
    phi = p['phi']
    v = p['v']
    delta = p['delta']
    c = p['c']
    K = p['K_sex']
    n = p['n']
    r = p['r']
    m = p['m']
    mu_mod = p['mu_mod']

    return (-n * v * xi ** 2 + c * m - (((v * (beta - delta) + phi) * xi + m * (beta - delta)) / ((delta * v - phi) * xi + delta * m) / k * n * r - c * v + m * n) * xi) / K ** 2 / (r * ((v * (beta - delta) + phi) * xi + m * (beta - delta)) / ((delta * v - phi) * xi + delta * m) / k + v * xi + m) * xi

def Mp0(p):

    beta = p['betta']
    k = p['k_pred']
    phi = p['phi']
    v = p['v']
    delta = p['delta']
    c = p['c']
    K = p['K_sex']
    n = p['n']
    r = p['r']
    m = p['m']
    mu_mod = p['mu_mod']

    return c*delta*k*m/((r*(beta-delta)+delta*k*m)*K**2)
PARAMETERS = {
    "n": 1.0,
    "K_sex": 1.0,  # renamed from K: Fortran treats K and k as the same symbol
    "betta": 2.0, # renamed from beta: AUTO has internal beta parameter ##0.62
    "k_pred": 1.0,  # renamed from k to avoid collision with K in Fortran
    "r": 2.0,
    "delta": 0.5,
    "c": 17.0,
    "m": 1.0 / 15.0,
    "phi": 0.6,
    "v": 1.0,
    "mu_mod": 0.01
}


lhc_params = {}

for key in PARAMETERS:

    lhc_params[key] = [1e-3*PARAMETERS[key],2*PARAMETERS[key]]

# Sampling
n_samples = 20000
sampler = qmc.LatinHypercube(d=len(lhc_params),seed=42)
sample = sampler.random(n=n_samples)

# Scale samples to bounds
l_bounds = [v[0] for v in lhc_params.values()]
u_bounds = [v[1] for v in lhc_params.values()]
sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

lhc_dict = {
    'c':    [],
    'betta': [],
    'delta':[],
    'K_sex':    [],
    'phi':  [],
    'r':    [],
    'k_pred':    [],
    'mu_mod':   [],
    'v':    [],
    'n':    [],
    'm':    []
}

Ms = []
Msfull = []
xis = []

for s in sample_scaled:
    i = 0
    p = PARAMETERS.copy()
    for key in PARAMETERS:
        p[key] = s[i]
        i += 1
    if p['betta']-p['delta']<=0 or p['phi']-p['delta']*p['v']<=0:
        #print("BAD")
        continue

    xi_T = p['delta']*p['m']/(p['phi']-p['delta']*p['v'])
    xi = np.linspace(0,xi_T,500000)

    y = M(xi,p)


    peaks, properties = find_peaks(y, height=0)

    if len(peaks)>1:
        print(peaks)
        print(xi_T)
        print(y[1])
        print('Derivative',Mp0(p))
        print(p)
        sns.lineplot(x=xi,y=y)
        plt.show()
    elif len(peaks)==0:
        print(y[1])
        print('Derivative',Mp0(p))
        print(peaks)
        print(p)
        sns.lineplot(x=xi,y=y)
        plt.show()

    Ms.append(len(peaks))
    Msfull.append(y)
    xis.append(xi)

print(len(Ms))
sns.lineplot(x=range(len(Ms)),y=Ms,linewidth=2)
plt.xlabel(r'Sample')
plt.ylabel(r'Number of peaks')
plt.show()

max_v = 0
max_xi = 0

for k in range(12):

    import random

    r = random.randint(0,len(Ms)-1)

    Mplot = Msfull[r][::10]
    xiplot = xis[r][::10]

    print(np.nanmax(Mplot))
    fn = np.nanargmax(Mplot<0)
    fn_xi = xiplot[fn]

    if np.nanmax(Mplot) > max_v:

        max_v = np.nanmax(Mplot)

    if fn_xi > max_xi:

        max_xi = fn_xi

    sns.lineplot(x=xiplot,y=Mplot)

plt.xlabel(r'$\xi$')
plt.ylabel(r'$M(\xi)$')
plt.yscale('log')
plt.ylim([1e-7,1.25*max_v])
plt.show()




