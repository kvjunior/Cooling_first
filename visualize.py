#!/usr/bin/env python3
# =============================================================================
# visualize.py
# -----------------------------------------------------------------------------
# Stage 6. Turns the CSV/parquet outputs of run.py into publication-ready
# figures (vector PDF for LaTeX + 300-dpi PNG preview). Dependency-light:
# matplotlib + pandas + numpy only, Agg backend (no display required).
#
# Figures
#   fig1_theta_curve     theta(SUHII) DML curve with CI band, the data-driven
#                        kink, and the Hansen threshold (gamma + CI) overlaid.
#                        The headline "thermally gated dividend" figure.
#   fig2_regime_dynamics share of cities above gamma per year + mean SUHII trend
#                        ("warming erodes the dividend").
#   fig3_morans          global Moran's I by year (spatial clustering).
#   fig4_sdm_effects      LeSage-Pace direct / indirect / total effects.
#   fig5_regime_slopes   regime-specific slope of the regime regressor(s) with
#                        95% CIs (synergy vs antagonism regime).
#   fig6_city_map        YREB cities by mean SUHII and final-year regime.
#
# Each figure is built only if its input file exists, so the script degrades
# gracefully if a stage was skipped.
#
# Usage
#   python visualize.py --config config.yaml --outdir outputs [--gamma 1.20]
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")                                # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("visualize")

# Colorblind-safe palette.
C_PRIMARY = "#2166ac"     # blue  (estimates)
C_ACCENT = "#b2182b"      # red   (thresholds / antagonism)
C_FILL = "#92c5de"        # light blue (confidence band)
C_GREEN = "#1b7837"
C_GRAY = "#6e6e6e"


def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
    })


def _save(fig, name: str, figdir: str) -> None:
    os.makedirs(figdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"))
    plt.close(fig)
    LOGGER.info("wrote figure %s.(pdf|png)", name)


def _read(path: str, **kw) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        LOGGER.warning("skipping: %s not found", path)
        return None
    return pd.read_csv(path, **kw)


# =============================================================================
# Fig 1 -- the headline: theta(SUHII) with the kink and the Hansen threshold
# =============================================================================
def fig1_theta_curve(theta: pd.DataFrame, thr_est: Optional[pd.DataFrame],
                     gamma_cli: Optional[float], figdir: str) -> None:
    x = theta["suhii"].to_numpy()
    th = theta["theta"].to_numpy()
    d = np.gradient(th, x)
    kink = float(x[int(np.argmin(d))])               # steepest decline

    has_overlap = "overlap" in theta.columns
    if has_overlap:
        fig, (ax, axo) = plt.subplots(
            2, 1, figsize=(6.4, 5.2), sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(0.0, color=C_GRAY, lw=1.0, ls=":", zorder=1)
    ax.fill_between(x, theta["ci_lo"], theta["ci_hi"], color=C_FILL,
                    alpha=0.5, lw=0, label="95% CI", zorder=2)
    ax.plot(x, th, color=C_PRIMARY, label=r"$\hat\theta(\mathrm{SUHII})$ (DML)", zorder=3)

    # Hansen threshold overlay (prefer the LR CI; fall back to alias columns).
    gamma = None
    if thr_est is not None and len(thr_est):
        gamma = float(thr_est["gamma"].iloc[0])
        clo = thr_est.get("lr_ci_lo", thr_est.get("ci_lo")).iloc[0]
        chi = thr_est.get("lr_ci_hi", thr_est.get("ci_hi")).iloc[0]
        if np.isfinite(clo) and np.isfinite(chi):
            ax.axvspan(clo, chi, color=C_ACCENT, alpha=0.10, zorder=1,
                       label="Hansen 95% CI")
    elif gamma_cli is not None:
        gamma = gamma_cli
    if gamma is not None:
        ax.axvline(gamma, color=C_ACCENT, lw=1.6, ls="--",
                   label=fr"Hansen threshold $\hat\gamma={gamma:.2f}$", zorder=4)
    ax.axvline(kink, color=C_GREEN, lw=1.4, ls="-.",
               label=fr"DML kink $={kink:.2f}$", zorder=4)

    ax.set_ylabel(r"Effect of digital productivity, $\theta$")
    ax.set_title("The digital dividend is thermally gated")
    ax.annotate("synergy regime", xy=(x[0], th[0]), xytext=(0.06, 0.92),
                textcoords="axes fraction", color=C_PRIMARY, fontsize=10)
    ax.annotate("antagonism regime", xy=(x[-1], th[-1]), xytext=(0.62, 0.16),
                textcoords="axes fraction", color=C_ACCENT, fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    if has_overlap:
        # Local-overlap diagnostic: where E[V^2|SUHII] is small, theta is weakly
        # identified -- shade the lowest-overlap decile so readers see it.
        ov = theta["overlap"].to_numpy()
        axo.fill_between(x, 0, ov, color=C_GRAY, alpha=0.35, lw=0)
        axo.plot(x, ov, color=C_GRAY, lw=1.2)
        thr_ov = np.quantile(ov, 0.10)
        weak = ov <= thr_ov
        axo.fill_between(x, 0, ov, where=weak, color=C_ACCENT, alpha=0.30, lw=0)
        axo.set_ylabel(r"$E[V^2\,|\,\mathrm{SUHII}]$", fontsize=9)
        axo.set_xlabel("Surface urban heat island intensity (SUHII, $^\\circ$C)")
        axo.set_ylim(bottom=0)
        axo.set_title("treatment overlap (red = weak local identification)",
                      fontsize=8.5, loc="left", color=C_GRAY)
    else:
        ax.set_xlabel("Surface urban heat island intensity (SUHII, $^\\circ$C)")
    _save(fig, "fig1_theta_curve", figdir)


# =============================================================================
# Fig 2 -- regime crossing over time
# =============================================================================
def fig2_regime_dynamics(dyn: pd.DataFrame, gamma: Optional[float], figdir: str) -> None:
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.bar(dyn["year"], dyn["share_above"] * 100, color=C_FILL,
            edgecolor=C_PRIMARY, width=0.6, label="cities above $\\hat\\gamma$ (%)")
    ax1.set_ylabel("Share of cities above threshold (%)", color=C_PRIMARY)
    ax1.tick_params(axis="y", labelcolor=C_PRIMARY)
    ax1.set_xlabel("Year")
    ax1.set_ylim(0, 100)
    ax1.grid(False)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(dyn["year"], dyn["mean_suhii"], color=C_ACCENT, marker="o",
             ms=4, label="mean SUHII")
    if gamma is not None:
        ax2.axhline(gamma, color=C_GRAY, ls="--", lw=1.2)
    ax2.set_ylabel("Mean SUHII ($^\\circ$C)", color=C_ACCENT)
    ax2.tick_params(axis="y", labelcolor=C_ACCENT)
    ax2.grid(False)

    ax1.set_title("Warming pushes cities across the threshold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    _save(fig, "fig2_regime_dynamics", figdir)


# =============================================================================
# Fig 3 -- Moran's I by year
# =============================================================================
def fig3_morans(moran: pd.DataFrame, figdir: str) -> None:
    sig = moran["p_value"] < 0.05
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhline(0.0, color=C_GRAY, ls=":", lw=1.0)
    ax.plot(moran["year"], moran["I"], color=C_PRIMARY, zorder=2)
    ax.scatter(moran["year"][sig], moran["I"][sig], color=C_PRIMARY,
               s=42, zorder=3, label="significant (p<0.05)")
    ax.scatter(moran["year"][~sig], moran["I"][~sig], facecolors="white",
               edgecolors=C_PRIMARY, s=42, zorder=3, label="n.s.")
    ax.set_xlabel("Year")
    ax.set_ylabel("Global Moran's $I$")
    ax.set_title("Spatial clustering of the DP-CEE coupling")
    ax.legend(loc="best", fontsize=9)
    _save(fig, "fig3_morans", figdir)


# =============================================================================
# Fig 4 -- SDM LeSage-Pace effects
# =============================================================================
def fig4_sdm_effects(eff: pd.DataFrame, figdir: str) -> None:
    eff = eff.copy()
    regs = eff.index.tolist()
    xpos = np.arange(len(regs))
    w = 0.26
    kinds = [("direct", C_PRIMARY, "direct"),
             ("indirect", C_GREEN, "indirect (spillover)"),
             ("total", C_ACCENT, "total")]
    has_se = all(f"{k}_se" in eff.columns for k, _, _ in kinds)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axhline(0.0, color=C_GRAY, lw=0.9)
    for j, (k, color, label) in enumerate(kinds):
        off = (j - 1) * w
        vals = eff[k].to_numpy()
        if has_se:
            errs = 1.96 * eff[f"{k}_se"].to_numpy()
            ax.bar(xpos + off, vals, w, label=label, color=color,
                   yerr=errs, capsize=3, error_kw={"lw": 1.0, "ecolor": C_GRAY})
            # Significance star above bars whose 95% CI excludes zero.
            if f"{k}_p" in eff.columns:
                for xp, v, e, pv in zip(xpos + off, vals, errs, eff[f"{k}_p"]):
                    if np.isfinite(pv) and pv < 0.05:
                        ax.annotate("*", (xp, v + np.sign(v) * (e + 0.01)),
                                    ha="center", va="bottom" if v >= 0 else "top",
                                    fontsize=12, color=color)
        else:
            ax.bar(xpos + off, vals, w, label=label, color=color)
    ax.set_xticks(xpos)
    ax.set_xticklabels(regs, rotation=30, ha="right")
    ax.set_ylabel("Effect on the outcome")
    title = "Spatial Durbin effects decomposition"
    if has_se:
        title += " (95% CI; * p<0.05)"
    ax.set_title(title)
    ax.legend(fontsize=9)
    _save(fig, "fig4_sdm_effects", figdir)


# =============================================================================
# Fig 5 -- regime-specific slopes (synergy vs antagonism)
# =============================================================================
def fig5_regime_slopes(coef: pd.DataFrame, figdir: str) -> None:
    rows = coef[coef.index.str.contains("__regime")]
    if rows.empty:
        LOGGER.warning("skipping fig5: no regime coefficients (no threshold adopted)")
        return
    labels, slopes, errs = [], [], []
    for name, r in rows.iterrows():
        reg, regime = name.split("__regime")
        labels.append(f"{reg}\nregime {regime}")
        slopes.append(r["coef"])
        errs.append(1.96 * r["se"])
    xpos = np.arange(len(labels))
    colors = [C_PRIMARY if i == 0 else C_ACCENT for i in range(len(labels))]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.axhline(0.0, color=C_GRAY, lw=0.9, ls=":")
    ax.errorbar(xpos, slopes, yerr=errs, fmt="o", ms=7, capsize=5,
                color="black", ecolor=C_GRAY, zorder=2)
    for i, (xp, s) in enumerate(zip(xpos, slopes)):
        ax.scatter(xp, s, color=colors[i], s=60, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Regime-specific slope (95% CI)")
    ax.set_title("Digital-productivity effect by heat regime")
    _save(fig, "fig5_regime_slopes", figdir)


# =============================================================================
# Fig 6 -- city SUHII map (schematic; lon/lat scatter)
# =============================================================================
def fig6_city_map(panel: pd.DataFrame, gamma: Optional[float], cfg: dict,
                  figdir: str) -> None:
    pcfg = cfg["panel"]
    q = cfg["heat"]["threshold_var"]
    last = panel[panel[pcfg["time_col"]] == panel[pcfg["time_col"]].max()]
    agg = (panel.groupby(pcfg["id_col"])
           .agg(lon=("lon", "first"), lat=("lat", "first"),
                suhii=(q, "mean")).reset_index())
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    sc = ax.scatter(agg["lon"], agg["lat"], c=agg["suhii"], cmap="YlOrRd",
                    s=42, edgecolors=C_GRAY, linewidths=0.4)
    if gamma is not None:
        hot = last[last[q] > gamma]
        hot = agg[agg[pcfg["id_col"]].isin(hot[pcfg["id_col"]])]
        ax.scatter(hot["lon"], hot["lat"], facecolors="none",
                   edgecolors=C_ACCENT, s=110, linewidths=1.3,
                   label=fr"above $\hat\gamma$ in final year")
        ax.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("Mean SUHII ($^\\circ$C)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("YREB cities by urban heat intensity")
    ax.grid(alpha=0.2)
    _save(fig, "fig6_city_map", figdir)


# =============================================================================
# Orchestration
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication figures.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--figdir", default=None)
    parser.add_argument("--gamma", type=float, default=None,
                        help="override SUHII threshold for overlays")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    _setup_style()

    outdir = args.outdir or cfg["paths"]["outputs_dir"]
    figdir = args.figdir or cfg["paths"].get("figures_dir", os.path.join(outdir, "figures"))

    thr_est = _read(os.path.join(outdir, "threshold_estimates.csv"))
    gamma = args.gamma
    if gamma is None and thr_est is not None and len(thr_est):
        gamma = float(thr_est["gamma"].iloc[0])

    theta = _read(os.path.join(outdir, "causal_theta_curve.csv"))
    if theta is not None:
        fig1_theta_curve(theta, thr_est, args.gamma, figdir)

    dyn = _read(os.path.join(outdir, "dynamics_by_year.csv"))
    if dyn is not None:
        fig2_regime_dynamics(dyn, gamma, figdir)

    moran = _read(os.path.join(outdir, "morans_i_by_year.csv"))
    if moran is not None:
        fig3_morans(moran, figdir)

    eff = _read(os.path.join(outdir, "sdm_effects.csv"), index_col=0)
    if eff is not None:
        fig4_sdm_effects(eff, figdir)

    coef = _read(os.path.join(outdir, "threshold_coefficients.csv"), index_col=0)
    if coef is not None:
        fig5_regime_slopes(coef, figdir)

    try:
        panel = pd.read_parquet(cfg["paths"]["panel_file"])
        fig6_city_map(panel, gamma, cfg, figdir)
    except Exception as exc:                          # pragma: no cover
        LOGGER.warning("skipping fig6 (map): %s", exc)

    LOGGER.info("all figures written to %s", figdir)


if __name__ == "__main__":
    main()
