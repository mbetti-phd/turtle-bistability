import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings('ignore')

# ── Default parameters
params_default = {
    'c':    17,
    'beta': 2,
    'delta':1/2,
    'K':    1,
    's':    1/15,
    'phi':  0.6,
    'r':    2,
    'k':    1,
    'mu':   1/39,
    'v':    1,
    'n':    1,
    'm':    1/15
}

# ── Equilibrium quantities
def q_poly_coeffs(p):
    mu, k, v, n, phi, delta = p['mu'], p['k'], p['v'], p['n'], p['phi'], p['delta']
    c, r, beta, K, m        = p['c'], p['r'], p['beta'], p['K'], p['m']

    a4 = -mu**3 * k * v * n * (phi - v*delta)
    a3 = mu**2 * (k*c*v*(phi - v*delta)
                  + n*r*(v*(beta - delta) + phi)
                  + k*m*n*(2*v*delta - phi))
    a2 = mu * (delta*k*m**2*n
               + m*n*r*(beta - delta)
               - K**2*k*mu**2*v*(phi - v*delta)
               - c*k*m*(2*v*delta - phi))
    a1 = (K**2*mu**2*r*v*(beta - delta)
          + K**2*k*m*mu**2*(2*v*delta - phi)
          + K**2*mu**2*phi*r
          - c*delta*k*m**2)
    a0 = K**2 * m * mu * (k*m*delta + r*(beta - delta))
    return [a4, a3, a2, a1, a0]


def find_AC(p):
    mu, phi, v, delta, m = p['mu'], p['phi'], p['v'], p['delta'], p['m']
    denom = phi - v*delta
    if abs(denom) < 1e-15:
        return np.nan
    A_max = m * delta / (mu * denom)
    coeffs = q_poly_coeffs(p)
    roots  = np.roots(coeffs)
    cands  = [r.real for r in roots
              if abs(r.imag) < 1e-8 and r.real > 0 and r.real < A_max]
    return max(cands) if cands else np.nan


def JC(AC, p):
    return p['mu'] / p['m'] * AC


def PC(AC, p):
    mu, m, beta, v, delta, phi, k = (p['mu'], p['m'], p['beta'],
                                      p['v'], p['delta'], p['phi'], p['k'])
    num = AC*(beta*v - v*delta + phi)*mu + m*(beta - delta)
    den = k*(delta*m - AC*(phi - v*delta)*mu)
    return np.nan if abs(den) < 1e-15 else num / den


# ── Cubic coefficients p, q, r
def quartic_coeffs(p):

    mu   = p['mu'];  k = p['k'];  v = p['v'];  n = p['n']
    phi  = p['phi']; r = p['r'];  beta = p['beta']
    K    = p['K'];   m = p['m'];  c = p['c'];  delta = p['delta']

    AA = delta * k * mu ** 3 * n * v ** 2 - k * mu ** 3 * n * phi * v
    BB = -c * delta * k * mu ** 2 * v ** 2 + 2 * delta * k * m * mu ** 2 * n * v + beta * mu ** 2 * n * r * v + c * k * mu ** 2 * phi * v - delta * mu ** 2 * n * r * v - k * m * mu ** 2 * n * phi + mu ** 2 * n * phi * r
    CC = K ** 2 * delta * k * mu ** 3 * v ** 2 - K ** 2 * k * mu ** 3 * phi * v - 2 * c * delta * k * m * mu * v + delta * k * m ** 2 * mu * n + beta * m * mu * n * r + c * k * m * mu * phi - delta * m * mu * n * r
    DD = 2 * K ** 2 * delta * k * m * mu ** 2 * v + K ** 2 * beta * mu ** 2 * r * v - K ** 2 * delta * mu ** 2 * r * v - K ** 2 * k * m * mu ** 2 * phi + K ** 2 * mu ** 2 * phi * r - c * delta * k * m ** 2
    EE = K ** 2 * delta * k * m ** 2 * mu + K ** 2 * beta * mu * r * m - K ** 2 * delta * mu * r * m

    return AA,BB,CC,DD,EE
# ── Discriminant of monic cubic  λ³ + p2 λ² + p1 λ + p0 ─────────────────
def discriminant(a,b,c,d,e):
    # The Δ for ax^4 + bx^3 + cx^2 + dx + e
    return (256*a**3*e**3 - 192*a**2*b*d*e**2 - 128*a**2*c**2*e**2 + 
            144*a**2*c*d**2*e - 27*a**2*d**4 + 144*a*b**2*c*e**2 - 
            6*a*b**2*d**2*e - 80*a*b*c**2*d*e + 18*a*b*c*d**3 + 
            16*a*c**4*e - 4*a*c**3*d**2 - 27*b**4*e**2 + 
            18*b**3*c*d*e - 4*b**3*d**3 - 4*b**2*c**3*e + b**2*c**2*d**2)


def disc_from_params(p):
    a,b,c,d,e = quartic_coeffs(p)
    if any(np.isnan(x) for x in [a,b,c,d,e]):
        return np.nan
    return discriminant(a,b,c,d,e)


# ── Find μ₀ (discriminant = 0)
def find_mu0(p, mu_lo=1e-4, mu_hi=10.0, n_bracket=300):
    """
    Coarse scan then Brent refinement — returns first sign-change root of Δ(μ).
    """
    mus = np.linspace(mu_lo, mu_hi, n_bracket)
    prev_val, prev_mu = np.nan, mus[0]
    for mu_val in mus:
        pp = dict(p); pp['mu'] = mu_val
        cur_val = disc_from_params(pp)
        if not np.isnan(prev_val) and not np.isnan(cur_val):
            if prev_val * cur_val < 0:
                try:
                    def f(mv, _p=p):
                        pp2 = dict(_p); pp2['mu'] = mv
                        return disc_from_params(pp2)
                    return brentq(f, prev_mu, mu_val, xtol=1e-10, rtol=1e-9)
                except Exception:
                    pass
        prev_val, prev_mu = cur_val, mu_val
    return np.nan



mu0_default = find_mu0(params_default)

# ── FNSI on μ₀
vary_params = ['c','beta','delta','K','phi','r','k','v','n','m']

def sensitivity_mu0(param_name, p, h_rel=1e-3):

    p0_val = p[param_name]
    mu0    = find_mu0(p)
    if np.isnan(mu0) or abs(mu0) < 1e-20 or abs(p0_val) < 1e-20:
        return np.nan

    h = abs(p0_val) * h_rel

    p_hi = dict(p); p_hi[param_name] = p0_val + h
    p_lo = dict(p); p_lo[param_name] = max(p0_val - h, 1e-12)

    mu0_hi = find_mu0(p_hi)
    mu0_lo = find_mu0(p_lo)

    if np.isnan(mu0_hi) or np.isnan(mu0_lo):
        return np.nan

    dmu0_dtheta = (mu0_hi - mu0_lo) / (2*h)
    return dmu0_dtheta * p0_val / mu0


print("\nComputing FNSI on μ₀ for each parameter...")
SI = {}
for param in vary_params:
    SI[param] = sensitivity_mu0(param, params_default)
    print(f"  {param:<8} Υ = {SI[param]:+.4f}")

# ── Parameter sweeps for μ₀
N_scan = 60

# ── Aesthetics 
BG      = "#ffffff"; FG = "#000000"
ACCENT  = "#f78166"   # μ₀ curve colour
GRID_C  = "#cccccc"
BAR_POS = "#f78166"; BAR_NEG = "#79c0ff"
ZERO_C  = "#8b949e"

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': "#000000", 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG, 'text.color': FG,
    'grid.color': GRID_C, 'grid.linewidth': 0.6,
    #'font.family': 'monospace',
    'axes.titlesize': 14, 'axes.labelsize': 14,
    'xtick.labelsize': 14, 'ytick.labelsize': 14,
})

latex_labels = {
    'c': '$c$', 'beta': r'$\beta$', 'delta': r'$\delta$',
    'K': '$K$', 'phi': r'$\phi$',  'r': '$r$',
    'k': '$k$', 'v': '$v$',        'n': '$n$', 'm': '$m$',
}

n_params = len(vary_params)
n_cols   = 5
n_rows   = int(np.ceil(n_params / n_cols))

fig = plt.figure(figsize=(16, 4 + 3.0*n_rows), facecolor=BG)
gs  = gridspec.GridSpec(2, 1, figure=fig,
                        height_ratios=[1.3, n_rows*1.0], hspace=0.50)

# ── Bar chart
ax_bar = plt.axes()
x      = np.arange(n_params)
vals   = [SI[p] if not np.isnan(SI[p]) else 0 for p in vary_params]
colors = [BAR_POS if v >= 0 else BAR_NEG for v in vals]

bars = ax_bar.bar(x, vals, 0.6, color=colors,
                  edgecolor="#0d1117", linewidth=0.4, zorder=3)
ax_bar.axhline(0,  color=ZERO_C, linewidth=0.8, zorder=2)
ax_bar.axhline( 1, color=ZERO_C, linewidth=0.5, linestyle='--', zorder=2)
ax_bar.axhline(-1, color=ZERO_C, linewidth=0.5, linestyle='--', zorder=2)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels([latex_labels[p] for p in vary_params], fontsize=14)
ax_bar.set_ylabel(r'$\Upsilon^{\mu_0}_\theta$', fontsize=14)
ax_bar.yaxis.set_minor_locator(AutoMinorLocator())
ax_bar.grid(axis='y', which='both', zorder=1)
ax_bar.set_xlim(-0.6, n_params - 0.4)

for bar, val in zip(bars, vals):
    if abs(val) > 0.01:
        offset = 0.04 if val >= 0 else -0.12
        ax_bar.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + offset,
                    f"{val:+.3f}", ha='center', va='bottom',
                    fontsize=12, color=FG)
plt.show()
# ── Sweep subplots 
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG,
    'axes.edgecolor': "#000000", 'axes.labelcolor': FG,
    'xtick.color': FG, 'ytick.color': FG, 'text.color': FG,
    'grid.color': GRID_C, 'grid.linewidth': 0.6,
    'axes.titlesize': 14, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
})
fig = plt.figure()
gs_inner = gridspec.GridSpec(n_cols, n_rows,
                                             hspace=0.5, wspace=0.5)

for idx, param in enumerate(vary_params):
    row, col = divmod(idx, n_cols)
    ax = fig.add_subplot(gs_inner[col, row])

    p0    = params_default[param]
    p_min = 0#1e-8 if p0 > 0 else -2*abs(p0)
    p_max = 5 * p0 if p0 > 0 else 1e-6
    sweep = np.linspace(p_min, p_max, N_scan)

    mu0_vals = []
    for pval in sweep:
        pp = dict(params_default); pp[param] = pval
        mu0_vals.append(find_mu0(pp))
    mu0_vals = np.array(mu0_vals, dtype=float)

    ax.plot(sweep, mu0_vals, color=ACCENT, linewidth=1.8, zorder=3)
    ax.fill_between(sweep, mu0_vals,facecolor="none", edgecolor=ACCENT, alpha=0.4,hatch='///') 
    ax.axvline(p0,         color="xkcd:bright blue",  linewidth=2, linestyle=':', zorder=2)
    ax.axhline(mu0_default,color="xkcd:bright blue",  linewidth=2, linestyle='--', zorder=2)
    ax.scatter([p0], [mu0_default], color=ACCENT, s=30, zorder=5)

    ax.set_xlabel(latex_labels[param], fontsize=12)
    ax.set_ylabel(r'$\mu_0$', fontsize=9)
    mu_lim = np.nan_to_num(mu0_vals,nan=-1,copy=True,posinf=-1, neginf=-1)
    ax.set_ylim([-1e-3,np.max(mu_lim*1.05)])
    ax.set_xlim([-1e-3,0.97*p_max])
    if np.max(mu_lim)>1:
        ax.set_ylim([1e-3,np.max(mu_lim*1.05)])
    if param=='beta':
        ax.set_xlim([params_default['delta'],p_max])
    elif param == 'delta':
        ax.set_xlim([p_min,params_default['beta']])
    elif param == 'phi':
        ax.set_xlim([params_default['v']*params_default['delta'],p_max])
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    finite = mu0_vals[np.isfinite(mu0_vals)]
    if len(finite) > 1:
        q2, q98 = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
        mg = max(0.05*(q98-q2), 1e-4)

# hide spare panels
for idx in range(n_params, n_rows*n_cols):
    row, col = divmod(idx, n_cols)
    fig.add_subplot(gs_inner[row, col]).set_visible(False)

plt.show()

