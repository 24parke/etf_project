"""
Simulated GBM — VIX-driven stochastic volatility
=================================================
SIM1 : S&P-like 1× stock,  daily σ = VIX / 100 / sqrt(252)
SIM3 : leveraged stock,     daily arith return = multiplier × SIM1 arith return

Runs 30 simulated paths per multiplier (2×, 3×, 4×) with a fixed VIX band
config [60%, 100%, 160%, 250%] and reports mean ± 95% CI of final P&L.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats

import archive.main as m
from archive.main import ETFHedgePortfolio

# ── Fixed VIX band config ─────────────────────────────────────────────────────
# VIX_BANDS = [(15, 0.06), (20, .1), (30, .2), (float("inf"), .4)]
VIX_BANDS = [(15, 1.00), (20, 1.6), (30, 2.5), (float("inf"), 4)]


# ── Parameters ────────────────────────────────────────────────────────────────
N_PATHS     = 30
MULTIPLIERS = [2, 3, 4]
MU          = 0.25         # SOXX-like annual drift (~25% historically)
DT          = 1 / 252
S1_0        = 14.61         # real SOXX start → shares_under ≈ 205
S2_0        = 0.48          # real SOXL start → shares_triple ≈ -2083
START, END  = "2010-06-09", "2026-07-01"
SEED        = 42


# ── Subclass: inject simulated prices, skip yfinance download ─────────────────
class SimulatedPortfolio(ETFHedgePortfolio):
    def __init__(self, prices_df, multiplier, long_notional=3_000.0):
        self.underlying     = "SIM1"
        self.triple         = "SIM3"
        self.long_notional  = long_notional
        self.short_notional = -long_notional / multiplier  # start beta-neutral for any leverage
        self.leverage       = float(multiplier)
        self.prices         = prices_df
        self.results        = None


# ── Download real VIX once ────────────────────────────────────────────────────
print("Downloading VIX data...")
vix_raw    = yf.download("^VIX", start=START, end=END, auto_adjust=False, progress=False)
vix_series = vix_raw["Close"].dropna()
dates      = [d.strftime("%Y-%m-%d") for d in vix_series.index]
vix_vals   = np.asarray(vix_series).flatten().astype(float)   # guaranteed 1-D
n_days     = len(vix_vals)

# SOXX is ~1.8× more volatile than the broad market; scale VIX accordingly
sig1_annual = 1.8 * vix_vals / 100.0
sig1_daily  = sig1_annual / np.sqrt(252)
mu_adj      = (MU - 0.5 * sig1_annual ** 2) * DT   # Ito correction


# ── GBM simulation ────────────────────────────────────────────────────────────
def simulate_paths(multiplier):
    rng = np.random.default_rng(SEED)
    eps = rng.standard_normal((N_PATHS, n_days))

    s1 = np.empty((N_PATHS, n_days))
    s2 = np.empty((N_PATHS, n_days))
    s1[:, 0] = S1_0
    s2[:, 0] = S2_0

    for t in range(1, n_days):
        log_r1   = mu_adj[t] + sig1_daily[t] * eps[:, t]
        r1_arith = np.expm1(log_r1)
        s1[:, t] = s1[:, t - 1] * np.exp(log_r1)
        s2[:, t] = np.maximum(s2[:, t - 1] * (1.0 + multiplier * r1_arith), 1e-6)

    return s1, s2


# ── Run strategy across all paths for one multiplier ─────────────────────────
LONG_NOTIONAL = 3_000.0

def run_experiment(multiplier):
    """Returns (N_PATHS,) raw P&L and (N_PATHS,) P&L normalised by long_notional."""
    s1_paths, s2_paths = simulate_paths(multiplier)
    m.VIX_BANDS = VIX_BANDS
    pnl = np.zeros(N_PATHS)

    for path in range(N_PATHS):
        prices_df = pd.DataFrame({
            "under":  s1_paths[path],
            "triple": s2_paths[path],
            "vix":    vix_vals,
        }, index=dates)
        port = SimulatedPortfolio(prices_df, multiplier=multiplier)
        r = port.run_strategy()
        pnl[path] = r["cumulative_pl"].iloc[-1]

        if (path + 1) % 10 == 0:
            print(f"  path {path + 1}/{N_PATHS} done")

    return pnl, pnl / LONG_NOTIONAL


# ── 95% CI via t-distribution ─────────────────────────────────────────────────
def ci95(arr):
    n      = len(arr)
    mean   = arr.mean()
    se     = arr.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return mean, mean - t_crit * se, mean + t_crit * se


# ── Figure 1: sample paths + VIX overlay ─────────────────────────────────────
print("\nPlotting sample paths...")
fig1, axes1 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig1.suptitle("Sample simulated path (path 1 of 30) with VIX overlay", fontsize=12)

for ax, mult in zip(axes1, MULTIPLIERS):
    s1, s2 = simulate_paths(mult)
    ax2 = ax.twinx()
    ax.plot(s1[0], lw=1.0, color="steelblue", label="1× SIM1")
    ax.plot(s2[0], lw=1.0, color="tomato",    label=f"{mult}× SIM3")
    ax2.plot(vix_vals, lw=0.6, color="gray", alpha=0.55, label="VIX")
    ax.set_ylabel("Simulated price ($)")
    ax2.set_ylabel("VIX", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title(f"{mult}× volatility multiplier")
    ax.grid(True, alpha=0.3)

step = max(1, n_days // 12)
ticks = range(0, n_days, step)
axes1[-1].set_xticks(list(ticks))
axes1[-1].set_xticklabels([dates[t] for t in ticks], rotation=40, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig("sim_paths.png", dpi=150)
print("Saved → sim_paths.png")


# ── Run all multipliers ───────────────────────────────────────────────────────
all_pnl        = {}
all_norm_pnl   = {}
all_stats      = {}
all_norm_stats = {}
for mult in MULTIPLIERS:
    print(f"\n{mult}× — running {N_PATHS} paths...")
    pnl, norm_pnl       = run_experiment(mult)
    all_pnl[mult]       = pnl
    all_norm_pnl[mult]  = norm_pnl
    all_stats[mult]     = ci95(pnl)
    all_norm_stats[mult]= ci95(norm_pnl)
    mean, lo, hi        = all_stats[mult]
    nmean, nlo, nhi     = all_norm_stats[mult]
    print(f"  raw:        mean=${mean:,.0f}  95% CI=[${lo:,.0f}, ${hi:,.0f}]")
    print(f"  normalised: mean={nmean:.3f}x  95% CI=[{nlo:.3f}x, {nhi:.3f}x]")


# ── Figure 2: raw P&L and normalised P&L side by side ────────────────────────
fig2, (ax_raw, ax_norm) = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
fig2.suptitle(
    f"Simulated GBM (VIX stochastic vol) — Mean P&L ± 95% CI\n"
    f"VIX bands: {[b[1] for b in VIX_BANDS]}  |  n={N_PATHS} paths  |  16-year horizon",
    fontsize=11,
)
x      = np.arange(len(MULTIPLIERS))
xlabels = [f"{m_}× leverage" for m_ in MULTIPLIERS]

def draw_bars(ax, stats_dict, ylabel, fmt):
    means = np.array([stats_dict[m_][0] for m_ in MULTIPLIERS])
    ci_lo = np.array([stats_dict[m_][1] for m_ in MULTIPLIERS])
    ci_hi = np.array([stats_dict[m_][2] for m_ in MULTIPLIERS])
    colors = ["steelblue" if v >= 0 else "tomato" for v in means]
    ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.6, alpha=0.85, zorder=2)
    ax.errorbar(x, means, yerr=[means - ci_lo, ci_hi - means],
                fmt="none", color="black", capsize=8, linewidth=1.4, zorder=3)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.4, zorder=0)
    y_span = max(ci_hi.max() - ci_lo.min(), 1e-9)
    for xi, v in zip(x, means):
        ax.text(xi, v + y_span * 0.02, fmt(v), ha="center", va="bottom", fontsize=10)

draw_bars(ax_raw,  all_stats,      "Cumulative P&L ($)",          lambda v: f"${v:,.0f}")
draw_bars(ax_norm, all_norm_stats, "P&L / long_notional ($3,000)", lambda v: f"{v:.3f}x")

ax_norm.set_xticks(x)
ax_norm.set_xticklabels(xlabels, fontsize=11)
plt.tight_layout()
plt.savefig("sim_sweep.png", dpi=150)
print("\nSaved → sim_sweep.png")
plt.show()
