#!/usr/bin/env python3
# =============================================================================
# run.py
# -----------------------------------------------------------------------------
# One-command orchestrator for the full analysis behind:
#   "Cooling first, decarbonizing second? Urban heat island thresholds in the
#    digital productivity-carbon efficiency coupling across China's YREB"
#
# Chains every stage in order and threads results between them:
#   1. data_pipeline      build the balanced 110 x 10 panel (real or --synthetic)
#   2. measurement        CEE (super-SBM), DP (entropy), CCD
#   3. threshold_models   two-way FE benchmark + Hansen panel threshold (-> gamma)
#   4. causal_ml          DML theta(SUHII): the data-driven kink
#   5. spatial_mechanism  SDM + Moran's I + cooling-tax mediation + dynamics,
#                         using the gamma estimated in stage 3
#
# The estimated SUHII threshold from stage 3 is passed automatically into the
# stage-5 warming dynamics, so the whole paper reproduces with a single command.
#
# Usage
#   python run.py --config config.yaml --synthetic        # runnable end-to-end now
#   python run.py --config config.yaml                     # once real data is in data/raw
#   python run.py --config config.yaml --synthetic --quick # fast smoke test (few bootstraps)
#
# All tables and summaries are written under paths.outputs_dir.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import yaml

import data_pipeline as dp
import measurement as ms
import threshold_models as th
import causal_ml as cm
import spatial_mechanism as sp

LOGGER = logging.getLogger("run")


def _banner(stage: str) -> float:
    LOGGER.info("=" * 70)
    LOGGER.info("STAGE: %s", stage)
    LOGGER.info("=" * 70)
    return time.time()


def _done(t0: float) -> None:
    LOGGER.info("... done in %.1fs", time.time() - t0)


def _apply_quick(cfg: dict) -> None:
    """Shrink bootstrap reps for a fast smoke test (does not affect real runs)."""
    cfg["threshold"]["bootstrap_reps"] = 49
    cfg["mediation"]["bootstrap_reps"] = 99
    LOGGER.warning("QUICK mode: bootstrap reps reduced (threshold=49, mediation=99)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full YREB analysis.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate a synthetic panel instead of reading data/raw")
    parser.add_argument("--quick", action="store_true",
                        help="reduce bootstrap replications for a fast smoke test")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    dp.setup_logging(cfg["paths"].get("log_file"))
    if args.quick:
        _apply_quick(cfg)

    outdir = args.outdir or cfg["paths"]["outputs_dir"]
    os.makedirs(outdir, exist_ok=True)
    panel_path = cfg["paths"]["panel_file"]
    meta_path = cfg["paths"]["metadata_file"]

    # ----- Stage 1: data ----------------------------------------------------
    t0 = _banner("1/5 data_pipeline -> balanced panel")
    metadata, panel = dp.build(cfg, synthetic=args.synthetic)
    os.makedirs(os.path.dirname(panel_path) or ".", exist_ok=True)
    metadata.to_csv(meta_path, index=False)
    _done(t0)

    # ----- Stage 2: measurement --------------------------------------------
    t0 = _banner("2/5 measurement -> CEE, DP, CCD")
    panel, weights = ms.compute_constructs(panel, cfg)
    panel = th.prepare_columns(panel, cfg)            # adds lngdp for the models
    panel.to_parquet(panel_path, index=False)
    (pd.Series(weights, name="weight").rename_axis("indicator")
     .to_csv(os.path.join(outdir, "dp_weights.csv")))
    _done(t0)

    # ----- Stage 3: benchmark + Hansen threshold ---------------------------
    t0 = _banner("3/5 threshold_models -> FE benchmark + panel threshold")
    bench = th.fe_benchmark(panel, cfg["benchmark"], cfg["panel"])
    bench.to_csv(os.path.join(outdir, "benchmark_fe.csv"))
    thr = th.fit_panel_threshold(panel, cfg["threshold"], cfg["panel"],
                                 seed=cfg.get("seed", 0))
    thr.coef_table.to_csv(os.path.join(outdir, "threshold_coefficients.csv"))
    pd.DataFrame(thr.tests).to_csv(os.path.join(outdir, "threshold_tests.csv"), index=False)
    th._write_threshold_estimates(thr, os.path.join(outdir, "threshold_estimates.csv"))
    with open(os.path.join(outdir, "threshold_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(thr.summary() + "\n")
    gamma: Optional[float] = thr.thresholds[0] if thr.thresholds else None
    LOGGER.info("estimated SUHII threshold gamma = %s",
                f"{gamma:.4f}" if gamma is not None else "none (no significant threshold)")

    # Seo-Shin (2016) threshold-exogeneity test: decide whether SUHII must be
    # treated as endogenous before relying on the Hansen split.
    exog = None
    if cfg.get("endogeneity", {}).get("test_threshold_exogeneity", False) and thr.thresholds:
        exog = th.test_threshold_exogeneity(panel, cfg["threshold"], cfg["panel"],
                                            cfg["endogeneity"], thr.thresholds)
        if exog is not None:
            with open(os.path.join(outdir, "threshold_exogeneity.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write(exog.summary() + "\n")
            LOGGER.info("threshold exogeneity: cf p=%.3f -> %s",
                        exog.p_value,
                        "ENDOGENOUS" if exog.endogenous else "exogeneity not rejected")
    _done(t0)

    # ----- Stage 4: causal ML ----------------------------------------------
    t0 = _banner("4/5 causal_ml -> DML theta(SUHII)")
    cml = cm.run_causal_ml(panel, cfg)
    cml.curve.to_csv(os.path.join(outdir, "causal_theta_curve.csv"), index=False)
    with open(os.path.join(outdir, "causal_ml_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(cml.summary() + "\n")
    _done(t0)

    # ----- Stage 5: spatial + mechanism + dynamics -------------------------
    t0 = _banner("5/5 spatial_mechanism -> SDM, Moran, mediation, dynamics")
    # Fall back to config gamma, then sample median, if stage 3 found no threshold.
    if gamma is None:
        gamma = cfg["dynamics"].get("gamma")
    spat = sp.run_spatial_mechanism(panel, metadata, cfg, gamma=gamma)
    spat["moran"].to_csv(os.path.join(outdir, "morans_i_by_year.csv"), index=False)
    spat["sdm"].coef_table.to_csv(os.path.join(outdir, "sdm_coefficients.csv"))
    spat["sdm"].effects.to_csv(os.path.join(outdir, "sdm_effects.csv"))
    spat["dynamics_year"].to_csv(os.path.join(outdir, "dynamics_by_year.csv"), index=False)
    with open(os.path.join(outdir, "spatial_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(spat["sdm"].summary() + "\n\n")
        fh.write(spat["mediation"].summary() + "\n\n")
        fh.write(f"Warming trend: {spat['dynamics_trend']}\n"
                 f"(gamma = {spat['gamma']:.4f})\n")
    _done(t0)

    # ----- Manifest ---------------------------------------------------------
    LOGGER.info("=" * 70)
    LOGGER.info("COMPLETE. Key results:")
    ce_col = cfg["carbon_efficiency"]["output_col"]
    LOGGER.info("  CEE mean=%.4f | DP mean=%.4f | CCD mean=%.4f",
                panel[ce_col].mean(),
                panel[cfg["digital_productivity"]["output_col"]].mean(),
                panel[cfg["coupling"]["ccd_col"]].mean())
    ddf_col = cfg["carbon_efficiency"].get("ddf_output_col", "cee_ddf")
    if ddf_col in panel.columns:
        rho_sp = panel[ce_col].corr(panel[ddf_col], method="spearman")
        LOGGER.info("  CEE robustness: Spearman(super-SBM, directional-distance) = %.4f", rho_sp)
    LOGGER.info("  benchmark DP coef = %.4f (p=%.3g)",
                bench.loc[cfg["benchmark"]["core_regressor"], "coef"],
                bench.loc[cfg["benchmark"]["core_regressor"], "p_value"])
    if thr.thresholds:
        LOGGER.info("  Hansen threshold(s) = %s  (first LR 95%% CI [%.3f, %.3f])",
                    [round(g, 4) for g in thr.thresholds], *thr.gamma_ci)
        if getattr(thr, "gamma_cis_boot", None):
            LOGGER.info("    block-bootstrap CI (first threshold) [%.3f, %.3f]",
                        *thr.gamma_cis_boot[0])
    if exog is not None:
        LOGGER.info("  SUHII exogeneity: control-function p=%.3f -> %s",
                    exog.p_value, "ENDOGENOUS (instrument)" if exog.endogenous
                    else "not rejected (Hansen split stands)")
    LOGGER.info("  DML kink @ SUHII = %.4f | ATE = %.4f | repeats=%d | lambda=%.4g",
                cml.kink_suhii, cml.ate, getattr(cml, "n_repeats", 1),
                getattr(cml, "lam", 0.0))
    if "overlap" in cml.curve.columns:
        ov = cml.curve["overlap"].to_numpy()
        LOGGER.info("    theta(SUHII) overlap E[V^2|.] min=%.4g (weakest local identification)",
                    float(ov.min()))
    eff = spat["sdm"].effects
    if "indirect_p" in eff.columns:
        core = cfg["spatial"]["regressors"][0]
        LOGGER.info("  SDM rho = %.4f | %s total=%.4f (p=%.3g) indirect=%.4f (p=%.3g)",
                    spat["sdm"].rho, core,
                    eff.loc[core, "total"], eff.loc[core, "total_p"],
                    eff.loc[core, "indirect"], eff.loc[core, "indirect_p"])
    else:
        LOGGER.info("  SDM rho = %.4f", spat["sdm"].rho)
    LOGGER.info("  cooling-tax indirect = %.4f (CI [%.4f, %.4f])",
                spat["mediation"].indirect, *spat["mediation"].indirect_ci)
    LOGGER.info("  warming = %.4f/yr (p=%.3g)",
                spat["dynamics_trend"]["warming_per_year"],
                spat["dynamics_trend"]["p_value"])
    LOGGER.info("  all tables written to: %s", outdir)
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
