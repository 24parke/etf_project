"""
ETF Hedge Portfolio — VIX-driven beta rebalancing
==================================================
Strategy:
  - Long underlying 1× ETF (e.g. SOXX)
  - Short leveraged 3× ETF (e.g. SOXL)
  - Beta exposure = long_value + short_value × 3
  - Rebalance daily when |beta_exposure| exceeds a VIX-scaled threshold

Pairs to try:
  SOXX / SOXL   (semiconductors)
  QQQ  / TQQQ   (Nasdaq)
  SPY  / SPXL   (S&P 500)
  DIA  / UDOW   (Dow)
  XLI  / DUSL   (industrials)
  XLF  / FAS    (financials)
  XLE  / ERX    (energy)
  IWM  / TNA    (small-cap)
  UNG  / BOYL   (natural gas)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# ---------------------------------------------------------------------------
# VIX → rebalance threshold (fraction of long leg value)
# ---------------------------------------------------------------------------
# VIX_BANDS = [(15, 0.01), (20, 0.03), (30, 0.06), (float("inf"), 0.10)]
# VIX_BANDS = [(15, 0.03), (20, 0.06), (30, 0.10), (float("inf"), 0.20)]
VIX_BANDS = [(15, 1.0), (20, 1.6), (30, 2.5), (float("inf"), 4.0)]

def vix_to_threshold(vix: float) -> float:
    for ceiling, threshold in VIX_BANDS:
        if vix <= ceiling:
            return threshold


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class ETFHedgePortfolio:
    """
    Parameters
    ----------
    underlying      : ticker of the 1× ETF
    triple          : ticker of the 3× leveraged ETF
    start / end     : date strings "YYYY-MM-DD"
    long_notional   : starting dollar value of the long leg  (default $3 000)
    short_notional  : starting dollar value of the short leg (default -$1 000)
    """

    def __init__(
        self,
        underlying: str,
        triple: str,
        start: str,
        end: str,
        long_notional: float = 3_000.0,
        short_notional: float = -1_000.0,
        leverage: float = 3.0,
    ) -> None:
        self.underlying = underlying
        self.triple = triple
        self.long_notional = long_notional
        self.short_notional = short_notional
        self.leverage = leverage

        self.prices = self._download(start, end)
        self.results: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def _download(self, start: str, end: str) -> pd.DataFrame:
        tickers = [self.underlying, self.triple, "^VIX"]
        raw = yf.download(
            tickers, start=start, end=end,
            group_by="ticker", auto_adjust=False, progress=False,
        )
        prices = pd.DataFrame({
            "under": raw[self.underlying]["Close"],
            "triple": raw[self.triple]["Close"],
            "vix":   raw["^VIX"]["Close"],
        }).dropna()
        prices.index = prices.index.strftime("%Y-%m-%d")
        return prices

    # ------------------------------------------------------------------
    # Vectorized return summary (no path dependence)
    # ------------------------------------------------------------------
    def return_summary(self) -> pd.DataFrame:
        u = self.prices["under"]
        t = self.prices["triple"]

        pct_u      = u.pct_change().fillna(0)
        pct_t      = t.pct_change().fillna(0)
        pct_ideal  = pct_u * 3          # 3× with no compounding drag

        cumret = lambda s: (1 + s).cumprod() * 100

        return pd.DataFrame({
            f"{self.underlying}_pct":        pct_u   * 100,
            f"{self.triple}_pct":            pct_t   * 100,
            "ideal_3x_pct":                  pct_ideal * 100,
            f"{self.underlying}_cumulative": cumret(pct_u),
            f"{self.triple}_cumulative":     cumret(pct_t),
            "ideal_3x_cumulative":           cumret(pct_ideal),
        }, index=self.prices.index)

    # ------------------------------------------------------------------
    # Path-dependent strategy simulation
    # ------------------------------------------------------------------
    def run_strategy(self) -> pd.DataFrame:
        """
        Simulate the VIX-threshold beta rebalancing strategy day by day.

        Because each day's trades change the share count carried into the
        next day, the loop is inherently sequential. We use pre-allocated
        numpy arrays to avoid per-row DataFrame overhead.
        """
        under_px  = self.prices["under"].values
        triple_px = self.prices["triple"].values
        vix       = self.prices["vix"].values
        n = len(under_px)

        # Pre-allocate output arrays
        long_val   = np.empty(n)
        short_val  = np.empty(n)
        total_val  = np.empty(n)
        beta_exp   = np.empty(n)
        beta_pct   = np.empty(n)
        cum_cash   = np.empty(n)   # cumulative cash from/to rebalancing trades
        daily_pl   = np.empty(n)
        cum_pl     = np.empty(n)
        actions    = [""] * n      # human-readable trade log
        sh_under   = np.empty(n)   # shares of underlying held each day
        sh_triple  = np.empty(n)   # shares of triple held each day

        # Mutable simulation state
        shares_under  =  self.long_notional  / under_px[0]
        shares_triple =  self.short_notional / triple_px[0]
        cash          =  0.0                  # net cash flow from trades

        for i in range(n):
            # ── mark to market ────────────────────────────────────────
            lv = shares_under  * under_px[i]
            sv = shares_triple * triple_px[i]
            tv = lv + sv
            be = lv + sv * self.leverage

            # ── rebalance (skip day 0 — no prior position yet) ────────
            action = "hold"
            if i > 0:
                threshold = vix_to_threshold(vix[i]) * lv
                # threshold = .05*lv
                be_before = be

                if be > threshold:
                    # Add to short leg to reduce positive beta
                    delta = be / self.leverage / triple_px[i]
                    shares_triple -= delta
                    cash          -= delta * triple_px[i]
                    sv = shares_triple * triple_px[i]
                    tv = lv + sv
                    be = lv + sv * self.leverage
                    action = (
                        f"sold {delta:.4f} {self.triple} @ ${triple_px[i]:.2f}"
                        f" | beta {be_before:+.2f} → {be:+.2f}"
                    )
                elif be < -threshold:
                    # Reduce short leg to eliminate negative beta
                    delta = -be / self.leverage / triple_px[i]
                    shares_triple += delta
                    cash          += delta * triple_px[i]
                    sv = shares_triple * triple_px[i]
                    tv = lv + sv
                    be = lv + sv * self.leverage
                    action = (
                        f"bought {delta:.4f} {self.triple} @ ${triple_px[i]:.2f}"
                        f" | beta {be_before:+.2f} → {be:+.2f}"
                    )

            # ── record ────────────────────────────────────────────────
            long_val[i]  = lv
            short_val[i] = sv
            total_val[i] = tv
            beta_exp[i]  = be
            beta_pct[i]  = be / lv if lv != 0 else 0.0
            cum_cash[i]  = cash
            actions[i]   = action
            sh_under[i]  = shares_under
            sh_triple[i] = shares_triple

            if i == 0:
                daily_pl[i] = 0.0
                cum_pl[i]   = 0.0
                actions[i]  = "inception"
            else:
                cash_delta  = cum_cash[i] - cum_cash[i - 1]
                daily_pl[i] = total_val[i] - total_val[i - 1] - cash_delta
                cum_pl[i]   = cum_pl[i - 1] + daily_pl[i]

        self.results = pd.DataFrame({
            f"long_{self.underlying}":  long_val,
            f"short_{self.triple}":     short_val,
            "portfolio_total":           total_val,
            "beta_exposure":             beta_exp,
            "beta_exposure_pct":         beta_pct,
            "cumulative_cash":           cum_cash,
            "daily_pl":                  daily_pl,
            "cumulative_pl":             cum_pl,
            f"shares_{self.underlying}": sh_under,
            f"shares_{self.triple}":     sh_triple,
            "action":                    actions,
        }, index=self.prices.index)

        return self.results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self, save_path: str = "portfolio.png") -> None:
        if self.results is None:
            raise RuntimeError("Call run_strategy() first.")

        r = self.results
        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        fig.suptitle(f"{self.underlying} / {self.triple} hedge portfolio", fontsize=13)

        # ── leg values & total ────────────────────────────────────────
        ax = axes[0]
        ax.plot(r[f"long_{self.underlying}"],  label=f"Long  {self.underlying}", lw=0.85)
        ax.plot(r[f"short_{self.triple}"],     label=f"Short {self.triple}",     lw=0.85)
        ax.plot(r["portfolio_total"],           label="Total",                    lw=1.2)
        ax.set_ylabel("Value ($)")
        ax.legend(); ax.grid(True)

        # ── beta exposure ─────────────────────────────────────────────
        ax = axes[1]
        ax.plot(r["beta_exposure"], lw=0.85, color="purple", label="Beta exposure")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_ylabel("Beta exposure ($)")
        ax.legend(); ax.grid(True)

        # ── cumulative P&L ────────────────────────────────────────────
        ax = axes[2]
        ax.plot(r["cumulative_pl"], lw=0.85, color="green", label="Cumulative P&L")
        ax.axhline(0, color="black", lw=0.6, ls="--")
        ax.set_ylabel("P&L ($)")
        ax.legend(); ax.grid(True)

        # Sparse date labels on x-axis
        step = max(1, len(r) // 12)
        ticks = range(0, len(r), step)
        for a in axes:
            a.set_xticks(list(ticks))
            a.set_xticklabels([r.index[t] for t in ticks], rotation=40, ha="right", fontsize=7)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Chart saved → {save_path}")
        plt.show()

    def save_csv(self, path: str = "portfolio.csv") -> None:
        if self.results is None:
            raise RuntimeError("Call run_strategy() first.")
        self.results.round(2).to_csv(path)
        print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = ETFHedgePortfolio(
        underlying="SOXX",
        triple="SOXL",
        start="2010-06-09",
        end="2026-07-01",
    )

    results = port.run_strategy()

    total_pl = results["cumulative_pl"].iloc[-1]
    print(f"Total P&L : ${total_pl:>12,.2f}")

    port.save_csv("portfolio.csv")
    port.plot("portfolio.png")