import pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from headstart_sim import *

beta = 0.56
R = pickle.load(open(f"results{beta}.pkl","rb"))
for (mu, (Pc, Jg, Ag, S, eq, Amax, C)) in  R.items():
    fig, ax = plt.subplots(1,1)
    print(mu, mu0)
    persist = S > 0.5          # 1 = never entered B_A
    AA, JJ = np.meshgrid(Ag, Jg)
    ax.contourf(AA, JJ, persist.astype(float), levels=[-.5,.5,1.5],
                colors=["#f4c9c4", "#c6e2f5"], alpha=.9)
    ax.contour(AA, JJ, persist.astype(float), levels=[.5], colors="k", linewidths=1.6)
    # first-order separatrix  J + kappa*A = C
    Aline = np.logspace(np.log10(Ag[0]), np.log10(C/kappa), 400)
    ax.plot(Aline, C - kappa*Aline, "--", color="crimson", lw=2,
            label=r"Eq. (84)")
    rows = np.load("intercepts.npy")
    sel = rows[(np.isclose(rows[:,0], mu/mu0)) & (rows[:,1]==1.0)][0]
    #ax.plot(sel[2], Jg[0], "D", ms=8, color="k", clip_on=False, zorder=5, label="boundary on axes")
    #ax.plot(Ag[0], sel[4], "D", ms=8, color="k", clip_on=False, zorder=5)
    ax.plot(eq[0][2], eq[0][1], "o", ms=9, mfc="w", mec="k", label=r"$E_{C,1}$ (saddle)")
    ax.plot(eq[1][2], eq[1][1], "s", ms=9, color="navy", label=r"$E_{C,2}$ (stable)")
    #ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0, np.max(AA)*1.1); ax.set_ylim(0,np.max(JJ)*1.1)
    ax.set_xlabel(r"initial adults $A_0$"); 
    #ax.set_title(f"({lab}) $\\mu={mu:.3f}$")
    ax.set_ylabel(r"initial juveniles $J_0$")
    ax.legend(loc="upper right", fontsize=8, framealpha=.95)
    kk = "\kappa_{\mathrm{eff}}"
    ax.text(0.8, 0.8, 
            rf"""$\kappa={sel[3]:.0f}$
${kk}={sel[6]:.0f}$""", 
            fontsize=12,
            verticalalignment='top',
            transform=ax.transAxes,
            ha='left',
            multialignment='left', 
            )
    import matplotlib.patches as mp
    fig.legend(handles=[mp.Patch(color="#c6e2f5", label=r"Convergence to $E_C$ (or limit cycle)"),
                        mp.Patch(color="#f4c9c4", label="Convergence to $E_P$")],
            loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(f"headstarting_basin_beta056_mu{mu:.3f}_mu0{mu0:0.3f}.png", dpi=200)
    plt.close(fig)

# diagnostics: where does the numerical boundary cross the A0 axis (J0 tiny) and J0 axis (A0 tiny)?
for mu,(Pc,Jg,Ag,S,eq,Amax,C) in R.items():
    pers = S>1e-2
    print(f"\nmu/mu0={mu/mu0:.2f}; C={C:.3f}, C/kappa={C/kappa:.4f}")
    # smallest A0 persisting for each J0 row
    for i in [0, 10, 20, 25, 30, 35, 40, len(Jg)-1]:
        js = np.where(pers[i])[0]
        print(f"  J0={Jg[i]:.3g}: min persisting A0 = {Ag[js[0]] if len(js) else None}")
    for j in [0, 10, 20, len(Ag)-1]:
        is_ = np.where(pers[:,j])[0]
        print(f"  A0={Ag[j]:.3g}: min persisting J0 = {Jg[is_[0]] if len(is_) else None}")
    print("  persisting fraction", pers.mean(), " finalA min among persisting", S[pers].min())