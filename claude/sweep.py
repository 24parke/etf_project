import matplotlib.pyplot as plt
import archive.main as m
from archive.main import ETFHedgePortfolio

# Sliding window over an exponentially-scaled base sequence.
# Tests 1-2 match the examples: [.01,.03,.06,.10] and [.03,.06,.10,.20]
BASE = [0.01, 0.03, 0.06, 0.10, 0.20, 0.35, 0.60, 1.00, 1.60, 2.50, 4.00, 6.00, 9.00]
VIX_CEILINGS = [15, 20, 30, float("inf")]

TESTS = [list(zip(VIX_CEILINGS, BASE[i:i+4])) for i in range(10)]

print("Downloading data (once)...")
port = ETFHedgePortfolio("SOXX", "SOXL", "2010-06-09", "2026-07-01")

pnls = []
labels = []

for idx, bands in enumerate(TESTS):
    m.VIX_BANDS = bands
    results = port.run_strategy()
    pl = results["cumulative_pl"].iloc[-1]
    pcts = [f"{b[1]:.0%}" for b in bands]
    label = f"T{idx+1}\n{pcts[0]}|{pcts[1]}|{pcts[2]}|{pcts[3]}"
    pnls.append(pl)
    labels.append(label)
    print(f"Test {idx+1:2d}  thresholds={[b[1] for b in bands]}  P&L=${pl:>10,.2f}")

fig, ax = plt.subplots(figsize=(14, 6))
colors = ["green" if p >= 0 else "red" for p in pnls]
bars = ax.bar(range(len(pnls)), pnls, color=colors, edgecolor="black", linewidth=0.5)

for bar, val in zip(bars, pnls):
    ypos = val + max(pnls) * 0.01 if val >= 0 else val - max(pnls) * 0.03
    ax.text(bar.get_x() + bar.get_width() / 2, ypos,
            f"${val:,.0f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Cumulative P&L ($)")
ax.set_title("SOXX / SOXL — Cumulative P&L vs VIX Band Threshold (10 configurations)")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.grid(True, axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("sweep.png", dpi=150)
print("\nChart saved → sweep.png")
plt.show()
