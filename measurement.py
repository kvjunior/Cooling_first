#!/usr/bin/env python3
# =============================================================================
# measurement.py
# -----------------------------------------------------------------------------
# Stage 2. Computes the three core constructs of the paper from the balanced
# panel produced by data_pipeline.py:
#
#   1. Carbon emission efficiency (CEE)
#        super-efficiency SBM with an undesirable output, solved as a linear
#        program via the Charnes-Cooper transformation.
#        References:
#          Tone, K. (2001) A slacks-based measure of efficiency in DEA. EJOR 130.
#          Tone, K. (2002) A slacks-based measure of super-efficiency.  EJOR 143.
#          Tone, K. (2003) Dealing with undesirable outputs in DEA: an SBM approach.
#
#   2. Digital productivity (DP)
#        entropy-weighted composite of twelve indicators (min-max normalized).
#
#   3. Coupling coordination degree (CCD)
#        D = sqrt(C * T), with C the coupling degree and T the comprehensive
#        coordination index between DP and CEE (both normalized to [0,1]).
#
# Design notes
#   * Pure functions operate on numpy arrays / DataFrames; the CLI wrapper is the
#     only part that touches config or disk -> fully unit-testable.
#   * DEA uses scipy.optimize.linprog(method="highs"): no external solver binary,
#     so the result is reproducible inside a plain conda environment.
#   * CEE is computed on a contemporaneous (annual) frontier by default; set
#     carbon_efficiency.frontier="pooled" for a meta-frontier.
#
# Usage
#   python measurement.py --config config.yaml \
#       --panel data/panel_yreb.parquet --out data/panel_yreb.parquet
# =============================================================================
from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linprog

LOGGER = logging.getLogger("measurement")


# =============================================================================
# 1. Carbon emission efficiency: SBM / super-SBM with undesirable output
# =============================================================================
def _sbm_undesirable(X: np.ndarray, Yg: np.ndarray, Yb: np.ndarray,
                     o: int, solver: str = "highs", rts: str = "crs") -> float:
    """
    Non-oriented SBM efficiency of DMU `o` WITH undesirable outputs (Tone, 2003),
    with DMU `o` included in the reference set. Returns rho in (0, 1].

    Original fractional program:
        rho = (1 - (1/m) sum_i s_i^- / x_io)
            / (1 + (1/(s1+s2)) (sum_r s_r^g / y_ro^g + sum_k s_k^b / y_ko^b))
    s.t.  X.lambda + s^- = x_o ;  Yg.lambda - s^g = y_o^g ;  Yb.lambda + s^b = y_o^b
          lambda, s^-, s^g, s^b >= 0

    Linearized (Charnes-Cooper) with t = 1/denominator and capitalized scaled
    variables Lambda = t.lambda, Sn = t.s^-, Sg = t.s^g, Sb = t.s^b. The optimal
    objective equals rho.
    """
    n, m = X.shape
    s1, s2 = Yg.shape[1], Yb.shape[1]
    xo, ygo, ybo = X[o], Yg[o], Yb[o]

    nvar = 1 + n + m + s1 + s2          # [t | Lambda(n) | Sn(m) | Sg(s1) | Sb(s2)]
    i_t = 0
    i_L = slice(1, 1 + n)
    i_Sn = slice(1 + n, 1 + n + m)
    i_Sg = slice(1 + n + m, 1 + n + m + s1)
    i_Sb = slice(1 + n + m + s1, nvar)

    # Objective: minimize tau = t - (1/m) sum_i Sn_i / x_io.
    c = np.zeros(nvar)
    c[i_t] = 1.0
    c[i_Sn] = -1.0 / (m * xo)

    A_eq, b_eq = [], []

    # Denominator normalization: t + (1/(s1+s2))(sum Sg/ygo + sum Sb/ybo) = 1.
    row = np.zeros(nvar)
    row[i_t] = 1.0
    row[i_Sg] = 1.0 / ((s1 + s2) * ygo)
    row[i_Sb] = 1.0 / ((s1 + s2) * ybo)
    A_eq.append(row); b_eq.append(1.0)

    # Inputs:  sum_j x_ij Lambda_j + Sn_i - x_io t = 0.
    for i in range(m):
        row = np.zeros(nvar)
        row[i_L] = X[:, i]
        row[1 + n + i] = 1.0
        row[i_t] = -xo[i]
        A_eq.append(row); b_eq.append(0.0)

    # Good outputs:  sum_j yg_rj Lambda_j - Sg_r - y_ro^g t = 0.
    for r in range(s1):
        row = np.zeros(nvar)
        row[i_L] = Yg[:, r]
        row[1 + n + m + r] = -1.0
        row[i_t] = -ygo[r]
        A_eq.append(row); b_eq.append(0.0)

    # Bad outputs:  sum_j yb_kj Lambda_j + Sb_k - y_ko^b t = 0.
    for k in range(s2):
        row = np.zeros(nvar)
        row[i_L] = Yb[:, k]
        row[1 + n + m + s1 + k] = 1.0
        row[i_t] = -ybo[k]
        A_eq.append(row); b_eq.append(0.0)

    # Variable returns to scale: sum_j lambda_j = 1  ->  sum_j Lambda_j = t
    # (Charnes-Cooper scaled). CRS imposes no such constraint.
    if rts == "vrs":
        row = np.zeros(nvar)
        row[i_L] = 1.0
        row[i_t] = -1.0
        A_eq.append(row); b_eq.append(0.0)

    res = linprog(c, A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq),
                  bounds=[(0.0, None)] * nvar, method=solver)
    return float(res.fun) if res.success else np.nan


def _super_sbm_undesirable(X: np.ndarray, Yg: np.ndarray, Yb: np.ndarray,
                           o: int, solver: str = "highs", rts: str = "crs") -> float:
    """
    Super-efficiency SBM of DMU `o` WITH undesirable outputs (Tone, 2002, extended),
    with DMU `o` EXCLUDED from the reference set. Returns delta >= 1 for a
    frontier DMU. Undesirable outputs are grouped with inputs (smaller is better).

    Fractional program:
        delta = ( (1/(m+s2)) ( sum_i xbar_i/x_io + sum_k zbar_k/y_ko^b ) )
              / ( (1/s1) sum_r ybar_r/y_ro^g )
    s.t.  xbar_i >= sum_{j!=o} lambda_j x_ij ;   xbar_i >= x_io
          zbar_k >= sum_{j!=o} lambda_j y_kj^b ; zbar_k >= y_ko^b
          ybar_r <= sum_{j!=o} lambda_j y_rj^g ; 0 < ybar_r <= y_ro^g
          lambda_j >= 0

    Linearized with t = 1/denominator; Xbar=t.xbar, Zbar=t.zbar, Ybar=t.ybar,
    Lambda=t.lambda. Optimal objective equals delta.
    """
    n, m = X.shape
    s1, s2 = Yg.shape[1], Yb.shape[1]
    xo, ygo, ybo = X[o], Yg[o], Yb[o]

    nvar = 1 + n + m + s2 + s1          # [t | Lambda(n) | Xbar(m) | Zbar(s2) | Ybar(s1)]
    i_t = 0
    i_L = slice(1, 1 + n)
    o_Xb = 1 + n
    o_Zb = 1 + n + m
    o_Yb = 1 + n + m + s2
    i_Yb = slice(o_Yb, nvar)

    # Objective: minimize (1/(m+s2))(sum Xbar/x_io + sum Zbar/ybo).
    c = np.zeros(nvar)
    c[o_Xb:o_Xb + m] = 1.0 / ((m + s2) * xo)
    c[o_Zb:o_Zb + s2] = 1.0 / ((m + s2) * ybo)

    # Normalization (equality): (1/s1) sum_r Ybar_r / ygo_r = 1.
    A_eq = np.zeros((1, nvar))
    A_eq[0, i_Yb] = 1.0 / (s1 * ygo)
    b_eq = np.array([1.0])

    A_ub, b_ub = [], []
    # sum_j x_ij Lambda_j - Xbar_i <= 0.
    for i in range(m):
        row = np.zeros(nvar); row[i_L] = X[:, i]; row[o_Xb + i] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    # sum_j yb_kj Lambda_j - Zbar_k <= 0.
    for k in range(s2):
        row = np.zeros(nvar); row[i_L] = Yb[:, k]; row[o_Zb + k] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    # Ybar_r - sum_j yg_rj Lambda_j <= 0.
    for r in range(s1):
        row = np.zeros(nvar); row[o_Yb + r] = 1.0; row[i_L] = -Yg[:, r]
        A_ub.append(row); b_ub.append(0.0)
    # x_io t - Xbar_i <= 0.
    for i in range(m):
        row = np.zeros(nvar); row[i_t] = xo[i]; row[o_Xb + i] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    # y_ko^b t - Zbar_k <= 0.
    for k in range(s2):
        row = np.zeros(nvar); row[i_t] = ybo[k]; row[o_Zb + k] = -1.0
        A_ub.append(row); b_ub.append(0.0)
    # Ybar_r - y_ro^g t <= 0.
    for r in range(s1):
        row = np.zeros(nvar); row[o_Yb + r] = 1.0; row[i_t] = -ygo[r]
        A_ub.append(row); b_ub.append(0.0)

    bounds = [(0.0, None)] * nvar
    bounds[1 + o] = (0.0, 0.0)          # exclude DMU o from the reference set

    # VRS: sum_{j!=o} lambda_j = 1 -> sum Lambda_j = t (Lambda_o is bounded to 0,
    # so summing all Lambda equals summing over the reference set j != o).
    if rts == "vrs":
        row = np.zeros(nvar)
        row[i_L] = 1.0
        row[i_t] = -1.0
        A_eq = np.vstack([A_eq, row])
        b_eq = np.append(b_eq, 0.0)

    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
                  A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=solver)
    return float(res.fun) if res.success else np.nan


def dea_scores(X: np.ndarray, Yg: np.ndarray, Yb: np.ndarray,
               super_efficiency: bool = True, solver: str = "highs",
               efficient_tol: float = 1e-6, rts: str = "crs",
               infeasible_fallback: str = "sbm") -> np.ndarray:
    """
    Efficiency scores for every DMU in a frontier group.

    For each DMU we first solve the standard SBM (rho <= 1). If super_efficiency
    is enabled and the DMU is on the frontier (rho >= 1 - tol), we re-solve the
    super-SBM (delta >= 1) to break ties among efficient units. The reported
    score is rho for inefficient DMUs and delta for efficient ones.

    Returns-to-scale is controlled by `rts` ({crs, vrs}). Super-SBM can be
    infeasible (Seiford & Zhu, 1999), especially under VRS; `infeasible_fallback`
    governs the response: "sbm" keeps the SBM score (= 1 for a frontier DMU, so
    the city is never silently dropped to NaN), "nan" propagates NaN.
    """
    n = X.shape[0]
    scores = np.empty(n)
    for o in range(n):
        rho = _sbm_undesirable(X, Yg, Yb, o, solver, rts=rts)
        if super_efficiency and np.isfinite(rho) and rho >= 1.0 - efficient_tol:
            delta = _super_sbm_undesirable(X, Yg, Yb, o, solver, rts=rts)
            if np.isfinite(delta) and delta >= 1.0 - efficient_tol:
                scores[o] = delta
            else:                                   # super-SBM infeasible/degenerate
                scores[o] = rho if infeasible_fallback == "sbm" else np.nan
        else:
            scores[o] = rho
    return scores


def _ddf_undesirable(X: np.ndarray, Yg: np.ndarray, Yb: np.ndarray, o: int,
                     solver: str = "highs", rts: str = "crs") -> float:
    """
    Chung-Fare-Grosskopf (1997) directional output distance function for DMU `o`,
    direction g = (y_o, -b_o). Solves (their LP 3.14):

        D = max beta
        s.t.  sum_j z_j yg_j  >= (1 + beta) yg_o      (goods: free disposability, >=)
              sum_j z_j yb_j   = (1 - beta) yb_o      (bads: weak disposability, =)
              sum_j z_j  x_j  <=         x_o          (inputs: free disposability, <=)
              z_j >= 0   [ , sum_j z_j = 1 for VRS ]

    The EQUALITY on bad outputs is the computational signature of weak
    disposability + null-jointness: emissions cannot be discarded for free.
    Returns beta* >= 0 (0 = on the frontier). Variables: [beta | z_1..z_n].
    """
    n, m = X.shape
    s1, s2 = Yg.shape[1], Yb.shape[1]
    xo, ygo, ybo = X[o], Yg[o], Yb[o]
    nvar = 1 + n                                    # [beta | z(n)]
    i_b, i_z = 0, slice(1, 1 + n)

    c = np.zeros(nvar); c[i_b] = -1.0               # maximize beta == minimize -beta

    A_ub, b_ub = [], []
    # Goods:  (1+beta) yg_o - sum_j z_j yg_j <= 0   ->  ygo*beta - Yg.z <= -ygo
    for r in range(s1):
        row = np.zeros(nvar); row[i_b] = ygo[r]; row[i_z] = -Yg[:, r]
        A_ub.append(row); b_ub.append(-ygo[r])
    # Inputs: sum_j z_j x_j - x_o <= 0
    for i in range(m):
        row = np.zeros(nvar); row[i_z] = X[:, i]
        A_ub.append(row); b_ub.append(xo[i])

    A_eq, b_eq = [], []
    # Bads (weak disposability):  sum_j z_j yb_j + yb_o*beta = yb_o
    for k in range(s2):
        row = np.zeros(nvar); row[i_z] = Yb[:, k]; row[i_b] = ybo[k]
        A_eq.append(row); b_eq.append(ybo[k])
    if rts == "vrs":                                # convexity: sum_j z_j = 1
        row = np.zeros(nvar); row[i_z] = 1.0
        A_eq.append(row); b_eq.append(1.0)

    bounds = [(0.0, None)] * nvar                   # beta >= 0, z >= 0
    res = linprog(c, A_ub=np.asarray(A_ub), b_ub=np.asarray(b_ub),
                  A_eq=np.asarray(A_eq), b_eq=np.asarray(b_eq),
                  bounds=bounds, method=solver)
    return float(-res.fun) if res.success else np.nan


def ddf_scores(X: np.ndarray, Yg: np.ndarray, Yb: np.ndarray,
               solver: str = "highs", rts: str = "crs") -> np.ndarray:
    """
    Directional-distance CEE for a frontier group, returned on the SBM's (0,1]
    scale via the monotone transform E = 1 / (1 + beta*) (= 1 on the frontier),
    so it is directly comparable to dea_scores. CFG's DDF is feasible by
    construction (no super-efficiency infeasibility).
    """
    n = X.shape[0]
    out = np.empty(n)
    for o in range(n):
        beta = _ddf_undesirable(X, Yg, Yb, o, solver, rts=rts)
        out[o] = 1.0 / (1.0 + beta) if np.isfinite(beta) else np.nan
    return out


def carbon_efficiency(panel: pd.DataFrame, cfg_ce: dict) -> pd.DataFrame:
    """
    Compute CEE for every city-year. Inputs/desired/undesired columns, the
    frontier mode, and returns-to-scale are taken from config -> carbon_efficiency.
    DEA is run separately within each frontier group (annual by default).

    Returns a DataFrame with the super-SBM CEE column and, when
    `compute_ddf_robustness` is set, the CFG directional-distance robustness
    column (E = 1/(1+beta*)) for the Spearman cross-check.
    """
    inp = cfg_ce["inputs"]
    des = cfg_ce["desired_outputs"]
    und = cfg_ce["undesired_outputs"]
    needed = inp + des + und
    missing = [c for c in needed if c not in panel.columns]
    if missing:
        raise KeyError(f"carbon_efficiency: panel missing columns {missing}")
    if (panel[needed] <= 0).any().any():
        raise ValueError("carbon_efficiency: all inputs/outputs must be > 0")

    rts = cfg_ce.get("returns_to_scale", "crs")
    solver = cfg_ce.get("solver", "highs")
    eff_tol = cfg_ce.get("efficient_tol", 1e-6)
    fallback = cfg_ce.get("infeasible_fallback", "sbm")
    want_ddf = bool(cfg_ce.get("compute_ddf_robustness", False))
    ddf_col = cfg_ce.get("ddf_output_col", "cee_ddf")

    cee = pd.Series(index=panel.index, dtype=float, name=cfg_ce["output_col"])
    ddf = pd.Series(index=panel.index, dtype=float, name=ddf_col) if want_ddf else None

    if cfg_ce.get("frontier", "annual") == "pooled":
        groups = [("pooled", panel.index)]
    else:
        groups = list(panel.groupby("year").groups.items())

    for label, idx in groups:
        sub = panel.loc[idx]
        X = sub[inp].to_numpy(float)
        Yg = sub[des].to_numpy(float)
        Yb = sub[und].to_numpy(float)
        cee.loc[idx] = dea_scores(
            X, Yg, Yb,
            super_efficiency=cfg_ce.get("super_efficiency", True),
            solver=solver, efficient_tol=eff_tol, rts=rts,
            infeasible_fallback=fallback,
        )
        if want_ddf:
            ddf.loc[idx] = ddf_scores(X, Yg, Yb, solver=solver, rts=rts)
        LOGGER.info("CEE frontier=%s n=%d mean=%.4f efficient=%d (rts=%s)",
                    label, len(idx), float(cee.loc[idx].mean()),
                    int((cee.loc[idx] >= 1.0 - eff_tol).sum()), rts)

    result = pd.DataFrame({cee.name: cee})
    if want_ddf:
        result[ddf_col] = ddf
        rho_sp = result[cee.name].corr(result[ddf_col], method="spearman")
        LOGGER.info("CEE robustness: Spearman(super-SBM, directional-distance) = %.4f", rho_sp)
    return result


# =============================================================================
# 2. Digital productivity: entropy-weighted composite index
# =============================================================================
def minmax_normalize(values: np.ndarray, direction: str = "positive") -> np.ndarray:
    """
    Column-wise min-max normalization to [0, 1]. A positive indicator maps its
    maximum to 1; a negative indicator is reversed. Constant columns map to 0.
    """
    v = np.asarray(values, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi <= lo:
        return np.zeros_like(v)
    z = (v - lo) / (hi - lo)
    return z if direction == "positive" else 1.0 - z


def entropy_weights(matrix: np.ndarray) -> np.ndarray:
    """
    Shannon-entropy weights for a non-negative (n x k) indicator matrix.

        p_ij = x_ij / sum_i x_ij
        e_j  = -(1/ln n) sum_i p_ij ln p_ij      (entropy; 0*ln0 := 0)
        d_j  = 1 - e_j                            (degree of diversification)
        w_j  = d_j / sum_j d_j

    Columns that are all-zero (e.g. constant after min-max) get weight 0; if
    every column is degenerate, weights fall back to uniform.
    """
    M = np.asarray(matrix, dtype=float)
    n, k = M.shape
    col_sum = M.sum(axis=0)
    P = np.divide(M, col_sum, out=np.zeros_like(M), where=col_sum > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_p = np.where(P > 0, np.log(P), 0.0)
    e = -(1.0 / np.log(n)) * (P * ln_p).sum(axis=0)
    d = np.where(col_sum > 0, 1.0 - e, 0.0)
    total = d.sum()
    if total <= 0:
        return np.full(k, 1.0 / k)
    return d / total


def digital_productivity(panel: pd.DataFrame,
                         indicators: List[dict]) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Entropy-weighted digital-productivity index per city-year.

    Returns (dp_series, weights_dict). Weights are computed over the pooled,
    min-max-normalized indicator matrix; DP is their weighted sum and therefore
    lies in [0, 1].
    """
    names = [ind["name"] for ind in indicators]
    missing = [c for c in names if c not in panel.columns]
    if missing:
        raise KeyError(f"digital_productivity: panel missing indicators {missing}")

    norm = np.column_stack([
        minmax_normalize(panel[ind["name"]].to_numpy(),
                         ind.get("direction", "positive"))
        for ind in indicators
    ])
    w = entropy_weights(norm)
    dp = norm @ w
    LOGGER.info("DP index: indicators=%d mean=%.4f range=[%.4f, %.4f]",
                len(names), float(dp.mean()), float(dp.min()), float(dp.max()))
    return (pd.Series(dp, index=panel.index, name="dp"),
            {nm: float(wj) for nm, wj in zip(names, w)})


# =============================================================================
# 3. Coupling coordination degree between DP and CEE
# =============================================================================
def coupling_coordination(u1: np.ndarray, u2: np.ndarray,
                          alpha: float = 0.5, beta: float = 0.5
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-system coupling coordination degree.

        C = 2 sqrt(U1 U2) / (U1 + U2)     coupling degree        in [0, 1]
        T = alpha U1 + beta U2            coordination index     in [0, 1]
        D = sqrt(C T)                     coupling coordination  in [0, 1]

    U1, U2 are expected in [0, 1]; values are clipped defensively. When
    U1 + U2 = 0, C (and hence D) is 0.
    """
    if not np.isclose(alpha + beta, 1.0):
        raise ValueError("coupling weights alpha + beta must equal 1")
    u1 = np.clip(np.asarray(u1, float), 0.0, 1.0)
    u2 = np.clip(np.asarray(u2, float), 0.0, 1.0)
    s = u1 + u2
    C = np.divide(2.0 * np.sqrt(u1 * u2), s, out=np.zeros_like(s), where=s > 0)
    T = alpha * u1 + beta * u2
    D = np.sqrt(C * T)
    return D, C, T


# =============================================================================
# Orchestration
# =============================================================================
def compute_constructs(panel: pd.DataFrame, cfg: dict
                       ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Append cee, dp, cee_norm, ccd, coupling_C, coupling_T to the panel and
    return (augmented_panel, dp_weights).
    """
    out = panel.copy()

    # 1. Carbon emission efficiency (returns a DataFrame: the super-SBM CEE plus,
    #    optionally, the CFG directional-distance robustness column).
    cee_df = carbon_efficiency(out, cfg["carbon_efficiency"])
    for col in cee_df.columns:
        out[col] = cee_df[col].to_numpy()

    # 2. Digital productivity.
    dp, weights = digital_productivity(out, cfg["digital_productivity"]["indicators"])
    out[cfg["digital_productivity"]["output_col"]] = dp

    # 3. Coupling coordination (both subsystems normalized to [0,1]).
    cc = cfg["coupling"]
    u1 = out[cfg["digital_productivity"]["output_col"]].to_numpy(float)  # DP in [0,1]
    cee = out[cfg["carbon_efficiency"]["output_col"]].to_numpy(float)
    cee_norm = minmax_normalize(cee, "positive")
    D, C, T = coupling_coordination(u1, cee_norm, cc["alpha"], cc["beta"])
    out["cee_norm"] = cee_norm
    out[cc["ccd_col"]] = D
    out["coupling_C"] = C
    out["coupling_T"] = T

    LOGGER.info("constructs done: CEE mean=%.4f | DP mean=%.4f | CCD mean=%.4f",
                float(out[cfg['carbon_efficiency']['output_col']].mean()),
                float(out[cfg['digital_productivity']['output_col']].mean()),
                float(out[cc['ccd_col']].mean()))
    return out, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CEE, DP and CCD.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--panel", default=None, help="input panel parquet")
    parser.add_argument("--out", default=None, help="output panel parquet")
    parser.add_argument("--weights-out", default="outputs/dp_weights.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    panel_path = args.panel or cfg["paths"]["panel_file"]
    out_path = args.out or cfg["paths"]["panel_file"]
    panel = pd.read_parquet(panel_path)
    LOGGER.info("loaded panel %s rows=%d", panel_path, len(panel))

    augmented, weights = compute_constructs(panel, cfg)

    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.weights_out) or ".", exist_ok=True)
    augmented.to_parquet(out_path, index=False)
    pd.Series(weights, name="weight").rename_axis("indicator").to_csv(args.weights_out)
    LOGGER.info("wrote constructs -> %s", out_path)
    LOGGER.info("wrote DP weights -> %s", args.weights_out)


if __name__ == "__main__":
    main()
