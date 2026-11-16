import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg")

p = dict(n=1, K=1, beta=0.56, k=1, r=2, delta=.5, c=17, m=1/15, phi=.6, v=1)
n,K,beta,k,r,d,c,m,phi,v = [p[s] for s in 'n K beta k r delta c m phi v'.split()]

PP    = (beta-d)/(k*d)
PC = []
kappa = 1 + r*PP/m
xiT   = d*m/(phi - v*d)

def Hxi(xi): 
    return ((v*(beta-d)+phi)*xi + m*(beta-d))/(((v*d-phi)*xi + d*m)*k)
def M(xi):
    H = Hxi(xi)
    return (c*m - n*v*xi**2 - (H*n*r - c*v + m*n)*xi)/(K**2*(r*H + v*xi + m))*xi

xs = np.linspace(1e-9, xiT*(1-1e-9), 400001)
Ms = M(xs); xistar = xs[Ms.argmax()]; mu0 = np.sqrt(Ms.max())

def coexistence(mu):
    f = lambda x: M(x) - mu**2
    x1 = brentq(f, 1e-12, xistar); x2 = brentq(f, xistar, xiT*(1-1e-9))
    return [(Hxi(x), x/m, x/mu) for x in (x1, x2)]          # (P,J,A)

def rhs(t, y, mu):
    P, J, A = y
    sat = J/(1+v*J)
    return [beta*P/(1+k*P) + phi*sat*P - d*P,
            c*A**2/(K**2 + n*A**2) - r*sat*P - m*J,
            m*J - mu*A]

def outcome(J0, A0, mu, T=6000.0):
    sol = solve_ivp(rhs, (0, T), [PP, J0, A0], args=(mu,), method="LSODA",
                    rtol=1e-8, atol=1e-12)
    return sol.y[2, -1], sol.y[1, -1]


def Amax_of(mu):
    """Smaller prey-only equilibrium E_L^-; real only if mu <= c/(2*K*sqrt(n))."""
    disc = c**2 - 4*K**2*n*mu**2
    return (c - np.sqrt(disc))/(2*n*mu) if disc >= 0 else None

def classify(J0, A0, mu, P0=None, T=20000.0, W=0.25):
    """Return ('collapse', t_enter) if the trajectory enters B_A (proved forward-invariant,
    all trajectories there -> E_P), else ('persist', (Amin, Amax_late)) over the last W*T."""
    P0 = PP if P0 is None else P0
    Am = Amax_of(mu)
    g = lambda t, y, mu: min(Am - y[2], mu*y[2] - m*y[1])      # > 0 inside B_A
    if g(0, (P0, J0, A0), mu) > 0:                              # starts inside B_A: event would never fire
        return "collapse", 0.0
    g.terminal, g.direction = True, 1
    sol = solve_ivp(rhs, (0, T), [P0, J0, A0], args=(mu,), method="LSODA",
                    rtol=1e-9, atol=1e-13, events=g, dense_output=True)
    if sol.status == 1:
        return "collapse", sol.t_events[0][0]
    A = sol.sol(np.linspace((1-W)*T, T, 4000))[2]
    return "persist", (A.min(), A.max())

def run(mu, ng=150, Jr=(1e-3, 15), Ar=(1e-3, 2),P0=None):
    Jg = np.logspace(np.log10(Jr[0]), np.log10(Jr[1]), ng)
    Ag = np.logspace(np.log10(Ar[0]), np.log10(Ar[1]), ng)
    S = np.zeros((ng, ng))                      # rows: J0, cols: A0
    for i, J0 in enumerate(Jg):
        for j, A0 in enumerate(Ag):
            S[i, j] = 1.0 if classify(J0, A0, mu,P0)[0] == 'persist' else 0.0
    return Jg, Ag, S

if __name__ == "__main__":
    mus = [0.05*mu0,0.3*mu0, 0.75*mu0,0.9*mu0]
    print("mu0 =", mu0, "kappa =", kappa, "PP =", PP)
    results = {}
    for mu in mus:
        eq = coexistence(mu)
        Amax = (c - np.sqrt(c**2 - 4*K**2*n*mu**2))/(2*n*mu)
        C = kappa**2*K**2*mu/c
        print(f"\nmu={mu:.4f}: E_C1 (P,J,A)={eq[0]}, E_C2={eq[1]}")
        print(f"  A_max={Amax:.4g}, approx A_C1=kappa*K^2*mu/c={kappa*K**2*mu/c:.4g}, C={C:.4g}")
        if beta == 2:
            Jg, Ag, S = run(mu,Jr=(1e-3, C*1.1), Ar=(1e-3, max(eq[0][2],C/kappa*1.1)),P0=eq[0][0])
        else:
            if mu in mus[0:2]:
                Jg, Ag, S = run(mu,Jr=(1e-3, C*2), Ar=(1e-3, max(2*eq[0][2],C/kappa*2)),P0=eq[0][0]) 
            else:
                Jg, Ag, S = run(mu,Jr=(1e-3, C*10), Ar=(1e-3, max(2*eq[0][2],C/kappa*2)),P0=eq[0][0])
        results[mu] = (eq[0][0],Jg, Ag, S, eq, Amax, C)
        np.save(f"S_mu{mu:.4f}.npy", S)
    import pickle; pickle.dump(results, open(f"results{p['beta']}.pkl", "wb"))