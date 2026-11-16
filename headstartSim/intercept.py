import numpy as np
from scipy.integrate import solve_ivp
from headstart_sim import *

def persists(J0, A0, mu, P0, T=8000.0):
    sol = solve_ivp(rhs, (0,T), [P0,J0,A0], args=(mu,), method="LSODA", rtol=1e-9, atol=1e-13)
    return sol.y[2,-1] > 1e-2

def bisect(fn, lo, hi, it=40):
    # fn(lo)=False, fn(hi)=True ; geometric bisection
    assert not fn(lo) and fn(hi), (fn(lo), fn(hi))
    for _ in range(it):
        mid = np.sqrt(lo*hi)
        if fn(mid): hi = mid
        else: lo = mid
    return hi

rows=[]
for frac in [0.05,0.3, 0.75, 0.9]:
    mu = frac*mu0
    C = kappa**2*K**2*mu/c
    for P0f in [0.5, 1.0, 2.0]:
        P0 = P0f*PP
        Jstar = bisect(lambda J: persists(J, 1e-6, mu, P0), 1e-3, 1e4)    # juvenile-only release
        Astar = bisect(lambda A: persists(1e-6, A, mu, P0), 1e-4, 1e3)    # adult-only release
        rows.append((frac, P0f, Astar, kappa, Jstar, C, Jstar/Astar))
        print(f"mu/mu0={frac:.2f} P0/PP={P0f:.1f}  A*={Astar:.4f} (lin {C/kappa:.4f})  "
              f"J*={Jstar:.3f} (lin {C:.3f})  J*/A*={Jstar/Astar:.1f}", flush=True)
np.save("intercepts.npy", np.array(rows))