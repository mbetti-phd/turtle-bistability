import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, t as tdist
from scipy.optimize import brentq
from scipy.stats import qmc

def quartic_coeffs(p):
    mu, k, v, n, phi, delta = p['mu'], p['k'], p['v'], p['n'], p['phi'], p['delta']
    c, r, beta, K, m        = p['c'], p['r'], p['beta'], p['K'], p['m']

    AA = delta * k * mu ** 3 * n * v ** 2 - k * mu ** 3 * n * phi * v
    BB = (-c * delta * k * mu ** 2 * v ** 2 + 2 * delta * k * m * mu ** 2 * n * v + 
          beta * mu ** 2 * n * r * v + c * k * mu ** 2 * phi * v - 
          delta * mu ** 2 * n * r * v - k * m * mu ** 2 * n * phi + mu ** 2 * n * phi * r)
    CC = (K ** 2 * delta * k * mu ** 3 * v ** 2 - K ** 2 * k * mu ** 3 * phi * v - 
          2 * c * delta * k * m * mu * v + delta * k * m ** 2 * mu * n + 
          beta * m * mu * n * r + c * k * m * mu * phi - delta * m * mu * n * r)
    DD = (2 * K ** 2 * delta * k * m * mu ** 2 * v + K ** 2 * beta * mu ** 2 * r * v - 
          K ** 2 * delta * mu ** 2 * r * v - K ** 2 * k * m * mu ** 2 * phi + 
          K ** 2 * mu ** 2 * phi * r - c * delta * k * m ** 2)
    EE = K ** 2 * delta * k * m ** 2 * mu + K ** 2 * beta * mu * r * m - K ** 2 * delta * mu * r * m

    return AA, BB, CC, DD, EE

def quartic_discriminant(a, b, c, d, e):
    # The Δ for ax^4 + bx^3 + cx^2 + dx + e
    return (256*a**3*e**3 - 192*a**2*b*d*e**2 - 128*a**2*c**2*e**2 + 
            144*a**2*c*d**2*e - 27*a**2*d**4 + 144*a*b**2*c*e**2 - 
            6*a*b**2*d**2*e - 80*a*b*c**2*d*e + 18*a*b*c*d**3 + 
            16*a*c**4*e - 4*a*c**3*d**2 - 27*b**4*e**2 + 
            18*b**3*c*d*e - 4*b**3*d**3 - 4*b**2*c**3*e + b**2*c**2*d**2)

def disc_from_params(p):
    a, b, c, d, e = quartic_coeffs(p)
    if any(np.isnan(x) for x in [a, b, c, d, e]): return np.nan
    return quartic_discriminant(a, b, c, d, e)

def find_mu0(p, mu_lo=1e-4, mu_hi=10.0, n_bracket=400):
    mus = np.linspace(mu_lo, mu_hi, n_bracket)
    prev_val, prev_mu = np.nan, mus[0]
    for mu_val in mus:
        pp = dict(p); pp['mu'] = mu_val
        cur_val = disc_from_params(pp)
        if not np.isnan(prev_val) and not np.isnan(cur_val):
            if prev_val * cur_val < 0:
                try:
                    return brentq(lambda m: disc_from_params({**p, 'mu': m}), prev_mu, mu_val)
                except: pass
        prev_val, prev_mu = cur_val, mu_val
    return np.nan

def partial_rank_correlation(X, y):
    """Computes PRCC for each column in X relative to y."""
    X_rank = np.apply_along_axis(rankdata, 0, X)
    y_rank = rankdata(y)
    
    # Combined matrix for correlation
    C = np.column_stack([X_rank, y_rank])
    rho = np.corrcoef(C, rowvar=False)
    
    # Matrix Inversion to get Partial Correlations
    try:
        prec = np.linalg.inv(rho)
    except np.linalg.inv.LinAlgError:
        # Fallback if matrix is singular (usually due to redundant parameters)
        print("WARNING: Singular Matrix")
        prec = np.linalg.pinv(rho)
        
    diag = np.diag(prec)
    prcc_vals = -prec[:-1, -1] / np.sqrt(diag[:-1] * diag[-1])
    return prcc_vals

# --- 3. Sampling and Execution ---

params_default = {'c': 17, 'beta': 2, 'delta': 0.5, 'K': 1, 'phi': 0.6, 
                  'r': 2, 'k': 1, 'v': 1, 'n': 1, 'm': 1/15}
vary_params = ['c', 'beta', 'delta', 'K', 'phi', 'r', 'k', 'v', 'n', 'm']
name_params = ['c', r'$\beta$', r'$\delta$', 'K', r'$\phi$', 'r', 'k', 'v', 'n', 'm']

N = 12000 # Number of samples

# Seeded Latin hypercube sampling (same generator as MsLHS.py)
SEED = 42
sampler = qmc.LatinHypercube(d=len(vary_params), seed=SEED)
lhd = sampler.random(N)

# Scale samples to +/- 20% of baseline values
l_bounds = [1e-3 * params_default[name] for name in vary_params]
u_bounds = [2 * params_default[name] for name in vary_params]
samples_scaled = qmc.scale(lhd, l_bounds, u_bounds)

X_samples = []
y_outputs = []

print(f"Sampling {N} points with phi > v*delta constraint (seed={SEED})...")
N_excluded = 0
for i in range(N):
    p_sample = {name: samples_scaled[i, j] for j, name in enumerate(vary_params)}
    
    if p_sample['phi'] > (p_sample['v'] * p_sample['delta']) and p_sample['beta']>p_sample['delta']:
        res = find_mu0(p_sample)
        if not np.isnan(res):
            N_excluded += 1
            X_samples.append([p_sample[name] for name in vary_params])
            y_outputs.append(res)

X_samples = np.array(X_samples)
y_outputs = np.array(y_outputs)

n_eff = len(y_outputs)
print(f"Accepted {n_eff}/{N} samples ({100*n_eff/N:.1f}%)")

# Run PRCC
results = partial_rank_correlation(X_samples, y_outputs)

# --- 4. Output

dof = n_eff - len(vary_params) - 2
tvals = results * np.sqrt(dof / (1.0 - results**2))
pvals = 2.0 * (1.0 - tdist.cdf(np.abs(tvals), dof))

# Fisher-z confidence intervals
z = 0.5 * np.log((1.0 + results) / (1.0 - results))
halfwidth = 1.96 / np.sqrt(n_eff - 3)
ci_lo = np.tanh(z - halfwidth)
ci_hi = np.tanh(z + halfwidth)

print(f"Effective sample size: {n_eff}, dof = {dof}")
for name, val, p, lo, hi in zip(vary_params, results, pvals, ci_lo, ci_hi):
    star = "*" if p < 0.05 else " "
    print(f"{name:<8}: PRCC={val:+.4f}  p={p:.3e}{star}  95% CI [{lo:+.4f}, {hi:+.4f}]")

import matplotlib.pyplot as plt

sorted_indices = np.argsort(np.abs(results))[::-1]
sorted_params = [vary_params[i] for i in sorted_indices]
sorted_values = [results[i] for i in sorted_indices]

colors = ["#f96240" if v >= 0 else "#3ba3ff" for v in results]

plt.figure(figsize=(10, 6))
bars = plt.bar(name_params, results, color=colors, edgecolor='black', linewidth=0.5)

for j, (bar, lo, hi, p) in enumerate(zip(bars, ci_lo, ci_hi, pvals)):
    if lo <= 0 <= hi:
        bar.set_alpha(0.35)
        bar.set_hatch('//')

plt.axhline(0, color='black', linewidth=0.8)
plt.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
plt.axhline(-0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Labels and Title
plt.ylabel('PRCC Value', fontsize=14)
plt.title(r'Partial Rank Correlation Coefficients (PRCC) for $\mu_0$',
          fontsize=14, pad=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(-1.1, 1.1) # PRCC is always between -1 and 1
plt.grid(axis='y', linestyle=':', alpha=0.6)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.02 if yval > 0 else -0.05), 
             f'{yval:+.3f}', ha='center', va='bottom' if yval > 0 else 'top', fontsize=10)

x_pos = np.arange(len(name_params))
yerr = np.vstack([
    results - ci_lo,   
    ci_hi - results
])
plt.errorbar(x_pos, results, yerr=yerr, fmt='none', capsize=5,
             ecolor='black', elinewidth=1.2, capthick=1.2, zorder=5)
plt.tight_layout()
plt.show()
