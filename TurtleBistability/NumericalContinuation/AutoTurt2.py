
# Before running:
# 1. Install AUTO-07p and PyCoBi using the official instructions.
# 2. Set auto_dir to your local AUTO-07p installation directory.

from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from pyrates import CircuitTemplate, NodeTemplate, OperatorTemplate
from pycobi import ODESystem
from scipy.integrate import solve_ivp
import sympy as sp

def a2 (A,J,P,mu,p):
    print('%%%%%')
    print(A)
    print(J)
    print(P)
    print(mu)

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

auto_dir = str(Path("~/auto-07p").expanduser())  # EDIT THIS PATH

AUTO_DIR = Path(os.environ.get("AUTO07P_DIR", "~/auto-07p")).expanduser()
WORKING_DIR = Path(__file__).parent.resolve() # fresh directory avoids stale generated Fortran files
WORKING_DIR.mkdir(parents=True, exist_ok=True)

MU_MIN = 1.0e-4
MU_MAX = 0.6

PARAMETERS = {
    "n": 1.0,
    "K_sex": 1.0,  # renamed from K: Fortran treats K and k as the same symbol
    "betta": 0.615, # renamed from beta: AUTO has internal beta parameter ##0.62
    "k_pred": 1.0,  # renamed from k to avoid collision with K in Fortran
    "r": 2.0,
    "delta": 0.5,
    "c": 17.0,
    "m": 1.0 / 15.0,
    "phi": 0.6,
    "v": 1.0,
    "mu_mod": 0.01
}

P_ss, J_ss, A_ss = get_steady_state(PARAMETERS)



INITIAL_STATE = {
    "P": f"variable({P_ss})",
    "J": f"variable({J_ss})",
    "A": f"variable({A_ss})",
}

equations = [
    "P' = betta * P / (1.0 + k_pred * P) "
    "+ phi * J / (1.0 + v * J) * P - delta * P",

    "J' = c / (K_sex**2) * A**2 / (1.0 + n * (A / K_sex)**2) "
    "- r * J / (1.0 + v * J) * P - m * J",

    "A' = m * J - mu_mod * A",
]

operator = OperatorTemplate(
    name="turtle_op",
    equations=equations,
    variables={**INITIAL_STATE, **PARAMETERS},
)
node = NodeTemplate(name="turtle_node", operators=[operator])
circuit = CircuitTemplate(name="turtle_model", nodes={"p": node})

ode = ODESystem.from_template(
    template=circuit,
    working_dir=str(WORKING_DIR),
    auto_dir=str(AUTO_DIR),
    init_cont=True,
    init_kwargs={"NMX": 20000, "NPR": 10000},
    analytical_jacobian=True,
    auto_constants=("ivp", "eq", "lc"),
)

try:
    
    print("Running equilibrium continuation...")
    
    solutions, continuation = ode.run(
        origin=0,
        starting_point="EP2",
        name="mu_equilibria",
        c="eq",
        ICP="p/turtle_op/mu_mod",
        bidirectional=False,
        get_stability=False, #This doesn't want to work for some reason
        get_eigenvals=False, #Neither this; stability computed manually below.
        ISP=2,  # Detect Hopf bifurcations
        ILP=1,  # Detect Limit Points (folds)
        ISW=1,
        RL0=MU_MIN,
        RL1=MU_MAX,
        NMX=25000,
        NPR=50,
        DS=1.0e-7,
        DSMIN=1.0e-8,
        DSMAX=2.0e-3,
        IADS=1,
        ITMX=40,
        ITNW=12,
        NWTN=12,
        EPSL=1.0e-7,
        EPSU=1.0e-7,
        EPSS=1.0e-7,
        THL={},
        THU={},
        UZR={},
        STOP={}
        )

    hopf_rows = solutions[solutions['bifurcation'] == 'HB']
    sd_rows = solutions[solutions['bifurcation'] == 'LP']

    hp = hopf_rows['mu_mod']
    lp = sd_rows['mu_mod']

    print("Detected Hopf points:\n", hopf_rows[['mu_mod', 'bifurcation', 'bifurcation_index']])

    if not hopf_rows.empty:
        print(hopf_rows.index[0])

        target_hb_label = int(hopf_rows.index[0])

        lc_solutions, lc_continuation = ode.run(
            origin=1,
            starting_point=target_hb_label,  
            name="hopf_limit_cycles",
            c="lc",
            ICP=["p/turtle_op/mu_mod", "PERIOD"],
            get_stability=True, #Now it works?
            get_eigenvals=True,
            IPS=2,
            ISW=-1,
            NMX=25000,
            NPR=10,
            DS=1e-4,
            DSMIN=1.0e-10,
            DSMAX=2.0e-2,
            IADS=1,
            ITMX=40,
            ITNW=12,
            NWTN=12,
            EPSL=1.0e-7,
            EPSU=1.0e-7,
            EPSS=1.0e-7,
            THL={},
            THU={},
            UZR={},
            NTST = 64,     # More mesh points for accurate orbit integration
            NCOL= 6,      # More collocation points
            STOP={}
        )
        print("Limit cycle continuation completed successfully.")
    else:
        print("No Hopf bifurcation points detected on this branch.") 


    P_sym, J_sym, A_sym = sp.symbols('P J A')
    mu_mod_sym, betta, k_pred, phi, v, delta = sp.symbols('mu_mod betta k_pred phi v delta')
    c, K_sex, n, r, m = sp.symbols('c K_sex n r m')

    f1 = betta * P_sym / (1.0 + k_pred * P_sym) + phi * J_sym / (1.0 + v * J_sym) * P_sym - delta * P_sym
    f2 = c / (K_sex**2) * A_sym**2 / (1.0 + n * (A_sym / K_sex)**2) - r * J_sym / (1.0 + v * J_sym) * P_sym - m * J_sym
    f3 = m * J_sym - mu_mod_sym * A_sym

    state_vars = [P_sym, J_sym, A_sym]
    jacobian_matrix = sp.Matrix([f1, f2, f3]).jacobian(state_vars)

    param_keys = ['betta', 'k_pred', 'phi', 'v', 'delta', 'c', 'K_sex', 'n', 'r', 'm', 'mu_mod']
    all_vars = state_vars + [mu_mod_sym] + [betta, k_pred, phi, v, delta, c, K_sex, n, r, m]
    jacobian_func = sp.lambdify(all_vars, jacobian_matrix, 'numpy')

    results, vmap = ode.extract(["mu_mod", "P", "J", "A"], cont="mu_equilibria")

    mu_vals = results['mu_mod'].to_numpy().ravel()
    P_vals = results['P'].to_numpy().ravel()
    J_vals = results['J'].to_numpy().ravel()
    A_vals = results['A'].to_numpy().ravel()

    fixed_params = {
        'betta': PARAMETERS['betta'],
        'k_pred': PARAMETERS['k_pred'],
        'phi': PARAMETERS['phi'],
        'v': PARAMETERS['v'],
        'delta': PARAMETERS['delta'],
        'c': PARAMETERS['c'],
        'K_sex': PARAMETERS['K_sex'],
        'n': PARAMETERS['n'],
        'r': PARAMETERS['r'],
        'm': PARAMETERS['m'],
    }

    # 5. Loop through each point on the branch to calculate stability
    stability_states = [] # True for stable, False for unstable

    for p_val, j_val, a_val, mu_val in zip(P_vals, J_vals, A_vals, mu_vals):
        # Evaluate numerical Jacobian at this specific point
        J_num = jacobian_func(p_val, j_val, a_val, mu_val, 
                            fixed_params['betta'], fixed_params['k_pred'], 
                            fixed_params['phi'], fixed_params['v'], 
                            fixed_params['delta'], fixed_params['c'], 
                            fixed_params['K_sex'], fixed_params['n'], 
                            fixed_params['r'], fixed_params['m'])
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(J_num)
        
        # Check if all real parts are negative
        is_stable = all(np.real(eigenvals) < 0)
        stability_states.append(is_stable)

    stability_states = np.array(stability_states, dtype=bool)

    plt.figure(figsize=(8, 5))

    # 1. Extract limit cycle continuation data
    try:
        lc_results, lc_vmap = ode.extract(["mu_mod", "A",'stability'], cont="hopf_limit_cycles")

        mu_lc = lc_results['mu_mod'].to_numpy().ravel()

        #Switched to A for clarity (A=0 is an attractor)
        P_col0 = lc_results[('A', 0)].to_numpy().ravel()
        P_col1 = lc_results[('A', 1)].to_numpy().ravel()
        
        P_lc_min = np.minimum(P_col0, P_col1)
        P_lc_max = np.maximum(P_col0, P_col1)

        stability = lc_results['stability'].to_numpy().ravel()
        stable_mask = (stability == True) | (stability == 1)

        print(lc_results["stability"].value_counts(dropna=False))

        plt.plot(mu_lc[stable_mask], P_lc_max[stable_mask], color='green', label='Stable Limit Cycle')
        plt.plot(mu_lc[stable_mask], P_lc_min[stable_mask], color='green')

        plt.plot(mu_lc[~stable_mask], P_lc_max[~stable_mask], color='orange', linestyle='--', label='Unstable Limit Cycle')
        plt.plot(mu_lc[~stable_mask], P_lc_min[~stable_mask], color='orange', linestyle='--')
        plt.axvline(hp.values[0],np.min(A_vals),np.max(A_vals),linestyle='--',linewidth=2)
    except KeyError as e:
        print(e)

    # 6. Plot with color-coding (Stable = Blue, Unstable = Red)
    plt.scatter(mu_vals[stability_states], A_vals[stability_states], color='blue', s=10, label='Stable')
    plt.scatter(mu_vals[~stability_states], A_vals[~stability_states], color='red', s=10, label='Unstable')

    plt.scatter(np.linspace(0,np.max(mu_vals)*1.1,len(mu_vals)), np.zeros((len(mu_vals,))), color='blue', s=10)
    ax = plt.gca()
    bounds = [0.6, 0.58, 0.35, 0.35] 
    #Uncomment next lines for inset. Will cause error if there is no limit cycle
    '''
    inset_ax = ax.inset_axes(bounds)
    inset_ax.scatter(mu_vals[stability_states], A_vals[stability_states], color='blue', s=10, label='Stable')
    inset_ax.scatter(mu_vals[~stability_states], A_vals[~stability_states], color='red', s=10, label='Unstable')
    inset_ax.plot(mu_lc[stable_mask], P_lc_max[stable_mask], color='green', label='Stable Limit Cycle')
    inset_ax.plot(mu_lc[stable_mask], P_lc_min[stable_mask], color='green')
    inset_ax.set_xlim(0.173,0.177)
    inset_ax.set_ylim(0,0.8)
    #'''
    plt.axvline(lp.values[0],np.min(A_vals),np.max(A_vals),linestyle='--',linewidth=2)

    plt.xlabel(r"$\mu$")
    plt.ylabel("A")
    plt.title(fr"$\beta = {PARAMETERS['betta']}$")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    fold = sd_rows.iloc[0]  # Verify this is the coexistence LP

    P_fold = fold['P']
    J_fold = fold['J']
    A_fold = fold['A']
    mu_fold = fold['mu_mod']

    print('a2',a2(A_fold[0],J_fold[0],P_fold[0],mu_fold[0],PARAMETERS))

    print("SUCCESS: PyCoBi ran a continuation and generated a plot.")

    print("results keys:        ", list(ode.results.keys()))
    print("auto_solutions keys: ", list(ode.auto_solutions.keys()))
    print("continuations keys:  ", list(ode.continuations.keys()))

    # Peek at each result entry
    for k, v in ode.results.items():
        print(f"\n=== results[{k}] ===")
        print(type(v))
        print(v if not hasattr(v, 'columns') else list(v.columns))

    import numpy as np
    import matplotlib.pyplot as plt

        # Minimal period detection without relying on PyCoBi extract
    lc_raw = ode.results.get('hopf_limit_cycles', {})
    print(lc_raw)

    if hasattr(lc_raw, 'columns'):
        cols = list(lc_raw.columns)
        print("All columns:", cols)
        
        # Try common period column names
        possible_period_cols = ['PERIOD', 'Period', 'per', 'period', 'PAR(11)']
        for pc in possible_period_cols:
            matching = [c for c in cols if pc.lower() in c.lower()]
            if matching:
                print(f"Possible period column(s): {matching}")
                for mc in matching:
                    vals = lc_raw[mc].to_numpy()
                    if len(vals) > 0:
                        print(f"  {mc}: min={vals.min()}, max={vals.max()}")

    # If still nothing, check the .out or .dat files from AUTO
    auto_out_files = list(WORKING_DIR.glob("hopf_*"))
    print("AUTO output files:", auto_out_files)


    # 1. Extract limit cycle data with period
    lc_results, lc_vmap = ode.extract(["mu_mod", "PAR(11)", "P", "J", "A"], 
                                    cont="hopf_limit_cycles")

    mu_lc = lc_results['mu_mod'].to_numpy().ravel()
    period_lc = lc_results['PAR(11)'].to_numpy().ravel()

    # Extract cycle bounds
    P_col0 = lc_results[('P', 0)].to_numpy().ravel()
    P_col1 = lc_results[('P', 1)].to_numpy().ravel()
    P_max = np.maximum(P_col0, P_col1)
    P_min = np.minimum(P_col0, P_col1)

    # 2. Plot period vs. μ (log scale if diverging)
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(mu_lc, period_lc, 'o-', linewidth=1)
    ax1.set_xlabel(r'$\mu$')
    ax1.set_ylabel('Period')
    ax1.set_title('Period vs. μ')
    ax1.grid(True)

    plt.tight_layout()
    plt.show()

    # 3. Check if period increases monotonically toward branch end
    print("Period progression (last 10 points):")
    print("μ:", mu_lc[-10:])
    print("Period:", period_lc[-10:])

    # 4. Check if cycle envelope approaches equilibrium value
    eq_mu = mu_vals  # From equilibrium continuation
    eq_P = P_vals

    # Find nearest equilibrium point to each LC point
    for i in range(len(mu_lc)-1, max(0, len(mu_lc)-10), -1):
        mu_target = mu_lc[i]
        idx_eq = np.argmin(np.abs(eq_mu - mu_target))
        P_eq = eq_P[idx_eq]
        P_cycle_avg = (P_max[i] + P_min[i]) / 2
        
        print(f"μ={mu_target:.4f}: LC center P={P_cycle_avg:.2f}, EQ P={P_eq:.2f}, diff={abs(P_cycle_avg-P_eq):.2f}")

    # 5. Calculate period divergence rate
    # Homoclinic: period ~ -ln|μ - μ_homo| / λ_saddle
    # Fold of cycles: period stays finite
    finite_period = np.all(np.isfinite(period_lc))
    max_period = np.max(period_lc)

    print(f"\nFinite period throughout: {finite_period}")
    print(f"Maximum period: {max_period:.2f}")
    print(f"Period at branch end: {period_lc[-1]:.2f}")

    # Strong indicator of homoclinic: period > 10× typical timescale
    typical_timescale = 1 / fixed_params['delta']  # Rough estimate
    print(f"Ratio to system timescale: {max_period / typical_timescale:.1f}×")

    print('a2',a2(A_fold[0],J_fold[0],P_fold[0],mu_fold[0],PARAMETERS))

finally:
    ode.close_session(clear_files=False)