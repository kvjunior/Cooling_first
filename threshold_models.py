#!/usr/bin/env python3
# =============================================================================
# threshold_models.py
# -----------------------------------------------------------------------------
# Stage 3. The econometric core of the paper.
#
#   fe_benchmark      Two-way (city + year) fixed-effects regression of the
#                     outcome on the core regressor + controls, cluster-robust
#                     standard errors by city.
#
#   fe_2sls           Within (FE) two-stage least squares for the endogeneity
#                     robustness check, cluster-robust SE. NOTE: instruments
#                     must be time-varying (the within transform annihilates
#                     time-invariant geography).
#
#   fit_panel_threshold
#                     Hansen (1999) fixed-effects panel threshold model with
#                     SUHII as the threshold variable. Sequentially estimates
#                     up to `max_thresholds` regimes, tests each additional
#                     threshold with a residual bootstrap, reports regime-
#                     specific slopes (cluster-robust SE) and a likelihood-ratio
#                     confidence interval for the first threshold.
#
# This operationalizes "cooling first, decarbonizing second": a significant
# SUHII threshold at which the slope of the digital-productivity term changes
# sign/magnitude is the empirical content of the title.
#
# References
#   Hansen, B.E. (1999) Threshold effects in non-dynamic panels: estimation,
#     testing, and inference. Journal of Econometrics 93, 345-368.
#   Hansen, B.E. (1996) Inference when a nuisance parameter is not identified
#     under the null hypothesis. Econometrica 64, 413-430.
#
# Everything operates on the within-transformed (demeaned) panel; the FE are
# removed by demeaning and the regressions carry no intercept. Pure functions
# take DataFrames/arrays; only the CLI touches config and disk.
#
# Usage
#   python threshold_models.py --config config.yaml \
#       --panel data/panel_yreb.parquet --outdir outputs
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

LOGGER = logging.getLogger("threshold_models")


# =============================================================================
# Linear-algebra and panel helpers (within transform, OLS, cluster vcov)
# =============================================================================
def _factorize(series: pd.Series) -> np.ndarray:
    """Map labels to contiguous integer codes 0..G-1."""
    return pd.factorize(series, sort=True)[0]


def _group_means(M: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Row-aligned group means: returns an array shaped like M."""
    M2 = M if M.ndim == 2 else M.reshape(-1, 1)
    counts = np.bincount(codes).astype(float)
    out = np.empty_like(M2)
    for j in range(M2.shape[1]):
        sums = np.bincount(codes, weights=M2[:, j])
        out[:, j] = (sums / counts)[codes]
    return out if M.ndim == 2 else out.ravel()


def make_demeaner(entity_codes: np.ndarray,
                  time_codes: Optional[np.ndarray]):
    """
    Return a function that within-transforms an array.

    One-way (entity) FE:    a - mean_i
    Two-way (entity+time):  a - mean_i - mean_t + grand_mean
    The two-way closed form is exact for a balanced panel (which the pipeline
    guarantees).
    """
    def demean(M: np.ndarray) -> np.ndarray:
        M2 = M.astype(float)
        M2 = M2 if M2.ndim == 2 else M2.reshape(-1, 1)
        out = M2 - _group_means(M2, entity_codes)
        if time_codes is not None:
            out = out - _group_means(M2, time_codes) + M2.mean(axis=0, keepdims=True)
        return out if M.ndim == 2 else out.ravel()
    return demean


def _ssr_beta_resid(y: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """OLS via least squares; return (SSR, beta, residuals)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), beta, resid


def _cluster_vcov(X: np.ndarray, resid: np.ndarray,
                  cluster_codes: np.ndarray) -> np.ndarray:
    """Cluster-robust sandwich covariance with a finite-sample correction."""
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    G = int(cluster_codes.max()) + 1
    meat = np.zeros((k, k))
    for g in range(G):
        idx = cluster_codes == g
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    return corr * bread @ meat @ bread


def _coef_table(names: Sequence[str], beta: np.ndarray,
                vcov: np.ndarray, n: int, k: int) -> pd.DataFrame:
    """Assemble a coefficient table with t-stats and two-sided p-values."""
    from scipy import stats
    se = np.sqrt(np.clip(np.diag(vcov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    p = 2.0 * stats.t.sf(np.abs(t), df=max(n - k, 1))
    return pd.DataFrame({"coef": beta, "se": se, "t": t, "p_value": p}, index=list(names))


# =============================================================================
# Column preparation
# =============================================================================
def prepare_columns(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add derived regressors used by the models (currently lngdp = log gdp)."""
    out = panel.copy()
    if "gdp" in out.columns and "lngdp" not in out.columns:
        if (out["gdp"] <= 0).any():
            raise ValueError("gdp must be positive to take logs")
        out["lngdp"] = np.log(out["gdp"])
    return out


def _require(panel: pd.DataFrame, cols: Sequence[str], where: str) -> None:
    missing = [c for c in cols if c not in panel.columns]
    if missing:
        raise KeyError(f"{where}: panel missing columns {missing}")


# =============================================================================
# Two-way FE benchmark
# =============================================================================
def fe_benchmark(panel: pd.DataFrame, cfg_bench: dict, panel_cfg: dict) -> pd.DataFrame:
    """Two-way FE regression of `outcome` on core regressor + controls."""
    y_col = cfg_bench["outcome"]
    regs = [cfg_bench["core_regressor"]] + list(cfg_bench["controls"])
    _require(panel, [y_col] + regs, "fe_benchmark")

    ent = _factorize(panel[panel_cfg["id_col"]])
    tim = _factorize(panel[panel_cfg["time_col"]]) if cfg_bench.get("time_effects", True) else None
    demean = make_demeaner(ent, tim)

    yd = demean(panel[y_col].to_numpy(float))
    Xd = demean(panel[regs].to_numpy(float))
    ssr, beta, resid = _ssr_beta_resid(yd, Xd)

    clusters = _factorize(panel[cfg_bench.get("cluster", panel_cfg["id_col"])])
    vcov = _cluster_vcov(Xd, resid, clusters)
    n, k = Xd.shape
    table = _coef_table(regs, beta, vcov, n, k)
    LOGGER.info("fe_benchmark outcome=%s n=%d core(%s) coef=%.4f p=%.4g",
                y_col, n, regs[0], table.loc[regs[0], "coef"], table.loc[regs[0], "p_value"])
    return table


# =============================================================================
# Within 2SLS (IV robustness)
# =============================================================================
def fe_2sls(panel: pd.DataFrame, cfg_iv: dict, panel_cfg: dict) -> pd.DataFrame:
    """Fixed-effects two-stage least squares with cluster-robust SE."""
    y_col = cfg_iv["outcome"]
    endog = [cfg_iv["endogenous"]]
    exog = list(cfg_iv["exog_controls"])
    inst = list(cfg_iv["instruments"])
    if not inst:
        raise ValueError("fe_2sls requires at least one (time-varying) instrument")
    if len(inst) < len(endog):
        raise ValueError("model under-identified: need #instruments >= #endogenous")
    _require(panel, [y_col] + endog + exog + inst, "fe_2sls")

    ent = _factorize(panel[panel_cfg["id_col"]])
    tim = _factorize(panel[panel_cfg["time_col"]]) if cfg_iv.get("time_effects", True) else None
    demean = make_demeaner(ent, tim)

    yd = demean(panel[y_col].to_numpy(float))
    Pd = demean(panel[endog].to_numpy(float))       # endogenous
    Wd = demean(panel[exog].to_numpy(float))         # exogenous controls
    Zd = demean(panel[inst].to_numpy(float))         # excluded instruments

    # First stage: project endogenous onto [exog, instruments].
    FS = np.hstack([Wd, Zd])
    fs_beta, *_ = np.linalg.lstsq(FS, Pd, rcond=None)
    Phat = FS @ fs_beta

    # Second stage on fitted endogenous + exog.
    Xhat = np.hstack([Phat, Wd])
    names = endog + exog
    beta, *_ = np.linalg.lstsq(Xhat, yd, rcond=None)

    # Residuals use the ACTUAL (not fitted) endogenous regressors.
    Xact = np.hstack([Pd, Wd])
    resid = yd - Xact @ beta
    clusters = _factorize(panel[cfg_iv.get("cluster", panel_cfg["id_col"])])
    vcov = _cluster_vcov(Xhat, resid, clusters)      # 2SLS sandwich (bread on Xhat)
    n, k = Xhat.shape
    table = _coef_table(names, beta, vcov, n, k)
    LOGGER.info("fe_2sls endog=%s coef=%.4f p=%.4g (instruments=%s)",
                endog[0], table.loc[endog[0], "coef"], table.loc[endog[0], "p_value"], inst)
    return table


# =============================================================================
# Hansen (1999) panel threshold model
# =============================================================================
@dataclass
class ThresholdResult:
    outcome: str
    threshold_var: str
    thresholds: List[float]
    coef_table: pd.DataFrame
    tests: List[dict]
    gamma_ci: Tuple[float, float]            # LR CI of the FIRST threshold (back-compat)
    ssr: float
    sigma2: float
    nobs: int
    # Camera-ready additions (default-valued for backward compatibility):
    gamma_cis: List[Tuple[float, float]] = field(default_factory=list)   # LR CI per threshold
    gamma_cis_boot: List[Tuple[float, float]] = field(default_factory=list)  # block-bootstrap CI per threshold
    dof: int = 0                              # effective residual dof used for sigma^2
    n_regimes: int = field(init=False)

    def __post_init__(self):
        self.n_regimes = len(self.thresholds) + 1
        # Keep the scalar gamma_ci in sync with the list form.
        if self.gamma_cis and (self.gamma_ci is None or not np.isfinite(self.gamma_ci[0])):
            self.gamma_ci = self.gamma_cis[0]

    def summary(self) -> str:
        lines = [f"Panel threshold model  outcome={self.outcome}  q={self.threshold_var}",
                 f"  thresholds: {[round(g, 4) for g in self.thresholds]}",
                 f"  regimes: {self.n_regimes}  nobs: {self.nobs}  "
                 f"dof(sigma^2): {self.dof}  sigma^2: {self.sigma2:.5g}"]
        for i, g in enumerate(self.thresholds):
            lr = self.gamma_cis[i] if i < len(self.gamma_cis) else (np.nan, np.nan)
            line = f"  gamma_{i+1} = {g:.4f}   LR 95% CI [{lr[0]:.4f}, {lr[1]:.4f}]"
            if i < len(self.gamma_cis_boot):
                bt = self.gamma_cis_boot[i]
                line += f"   block-bootstrap CI [{bt[0]:.4f}, {bt[1]:.4f}]"
            lines.append(line)
        for t in self.tests:
            lines.append(f"  threshold #{t['k']}: F={t['F']:.3f}  "
                         f"bootstrap p={t['p_value']:.3f}  reps={t['reps']}")
        lines.append(self.coef_table.to_string(float_format=lambda v: f"{v:.4f}"))
        return "\n".join(lines)


def _regime_masks(q: np.ndarray, thresholds: Sequence[float]) -> List[np.ndarray]:
    """Masks for regimes (-inf, g1], (g1, g2], ..., (gK, +inf)."""
    edges = np.concatenate(([-np.inf], np.sort(thresholds), [np.inf]))
    return [(q > edges[r]) & (q <= edges[r + 1]) for r in range(len(edges) - 1)]


def _build_design(Z: np.ndarray, D: np.ndarray, q: np.ndarray,
                  thresholds: Sequence[float]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Raw design [controls | D*I(regime_0) | ... | D*I(regime_K)]."""
    masks = _regime_masks(q, thresholds)
    cols = [Z] + [D * m[:, None] for m in masks]
    return np.hstack(cols), masks


def _make_grid(q: np.ndarray, trim: float, max_grid: int) -> np.ndarray:
    """Candidate thresholds: trimmed unique q values, capped via quantiles."""
    lo, hi = np.quantile(q, [trim, 1.0 - trim])
    cand = np.unique(q[(q >= lo) & (q <= hi)])
    if len(cand) > max_grid:
        cand = np.unique(np.quantile(cand, np.linspace(0.0, 1.0, max_grid)))
    return cand


def _fit_given(yd: np.ndarray, Z: np.ndarray, D: np.ndarray, q: np.ndarray,
               thresholds: Sequence[float], demean, min_count: int):
    """SSR / beta / resid for a fixed threshold vector (or None if a regime is too small)."""
    des, masks = _build_design(Z, D, q, thresholds)
    if min(int(m.sum()) for m in masks) < min_count:
        return None
    Xd = demean(des)
    ssr, beta, resid = _ssr_beta_resid(yd, Xd)
    return ssr, beta, resid, Xd


def _search_additional(yd, Z, D, q, fixed, grid, demean, min_count):
    """Best new threshold given `fixed` thresholds held constant."""
    best = (np.inf, None)
    for g in grid:
        if any(abs(g - f) < 1e-12 for f in fixed):
            continue
        out = _fit_given(yd, Z, D, q, list(fixed) + [g], demean, min_count)
        if out is not None and out[0] < best[0]:
            best = (out[0], g)
    return best  # (ssr, gamma)


def _sequential(yd, Z, D, q, n_thr, grid, demean, min_count, refine=True):
    """Sequentially estimate up to n_thr thresholds, with one refinement pass."""
    thresholds: List[float] = []
    for _ in range(n_thr):
        ssr, g = _search_additional(yd, Z, D, q, thresholds, grid, demean, min_count)
        if g is None:
            break
        thresholds.append(g)
        if refine and len(thresholds) > 1:
            for i in range(len(thresholds)):
                others = thresholds[:i] + thresholds[i + 1:]
                _, gi = _search_additional(yd, Z, D, q, others, grid, demean, min_count)
                if gi is not None:
                    thresholds[i] = gi
        thresholds = sorted(thresholds)
    fit = _fit_given(yd, Z, D, q, thresholds, demean, min_count)
    return thresholds, fit  # fit = (ssr, beta, resid, Xd)


def _entity_blocks(entity_codes: np.ndarray) -> Optional[List[np.ndarray]]:
    """
    Row indices for each entity, in their original (time-sorted) order. Returns
    None if the panel is unbalanced (blocks of unequal length), in which case the
    caller falls back to i.i.d. resampling.
    """
    n_ent = int(entity_codes.max()) + 1
    blocks = [np.where(entity_codes == i)[0] for i in range(n_ent)]
    lengths = {len(b) for b in blocks}
    return blocks if len(lengths) == 1 else None


def _resample_block(source: np.ndarray, blocks: List[np.ndarray],
                    rng: np.random.Generator) -> np.ndarray:
    """
    Hansen (1999, sec. 4.1) block-by-individual residual bootstrap: draw n_ent
    donor entities with replacement and re-attach each donor's whole length-T
    residual vector to an original entity slot (regressors held fixed).
    """
    n_ent = len(blocks)
    donors = rng.integers(0, n_ent, size=n_ent)
    out = np.empty_like(source)
    for i in range(n_ent):
        out[blocks[i]] = source[blocks[donors[i]]]
    return out


def _bootstrap_pvalue(yd, Z, D, q, demean, min_count, grid, k, F_obs, reps, rng,
                      dof, blocks=None, alt_resid=None):
    """
    Bootstrap p-value for the k-th threshold (H0: k-1 thresholds vs H1: k).

    Hansen (1999): hold regressors and the threshold variable fixed; generate
    y* under H0 from the (k-1)-threshold fit, with errors resampled in
    individual blocks (`blocks`) when available, else i.i.d. The error source is
    the alternative (k-threshold) residuals when `alt_resid` is supplied
    (Hansen Eq. 21 generalization), otherwise the H0 residuals. F* is computed
    with the same effective dof as the observed F, so the comparison is exact.
    """
    th0, fit0 = _sequential(yd, Z, D, q, k - 1, grid, demean, min_count)
    if fit0 is None:
        return np.nan
    _, _, resid0, _ = fit0
    fitted0 = yd - resid0
    src = alt_resid if alt_resid is not None else resid0
    src = src - src.mean()                       # recentre the error pool
    n = len(yd)
    count = 0
    for _ in range(reps):
        e = (_resample_block(src, blocks, rng) if blocks is not None
             else src[rng.integers(0, n, size=n)])
        ystar = fitted0 + e
        out0 = _fit_given(ystar, Z, D, q, th0, demean, min_count)   # SSR under H0
        _, fitb = _sequential(ystar, Z, D, q, k, grid, demean, min_count)  # SSR under H1
        if out0 is None or fitb is None:
            continue
        s0b, s1b = out0[0], fitb[0]
        if s1b <= 0:
            continue
        Fb = (s0b - s1b) / (s1b / dof)
        if Fb >= F_obs:
            count += 1
    return count / reps


def fit_panel_threshold(panel: pd.DataFrame, cfg_thr: dict, panel_cfg: dict,
                        seed: int = 0) -> ThresholdResult:
    """Estimate the Hansen FE panel threshold model defined in config -> threshold."""
    y_col = cfg_thr["outcome"]
    q_col = cfg_thr["threshold_var"]
    reg_cols = list(cfg_thr["regime_regressors"])
    ctrl_cols = list(cfg_thr["controls"])
    _require(panel, [y_col, q_col] + reg_cols + ctrl_cols, "fit_panel_threshold")

    two_way = cfg_thr.get("time_effects", True)
    ent = _factorize(panel[panel_cfg["id_col"]])
    tim = _factorize(panel[panel_cfg["time_col"]]) if two_way else None
    demean = make_demeaner(ent, tim)
    clusters = _factorize(panel[cfg_thr.get("cluster", panel_cfg["id_col"])])

    y = panel[y_col].to_numpy(float)
    q = panel[q_col].to_numpy(float)
    Z = panel[ctrl_cols].to_numpy(float)            # regime-invariant controls
    D = panel[reg_cols].to_numpy(float)             # regime-dependent regressors
    yd = demean(y)
    n = len(yd)

    # Effective residual dof for sigma^2 (Hansen 1999, Eq. 10): the within
    # transform costs one df per entity. One-way FE -> n_ent*(T-1);
    # two-way FE -> (n_ent-1)*(T-1). This is what enters the LR gamma CI;
    # using nT instead makes the CI ~T/(T-1) too narrow.
    n_ent = int(panel[panel_cfg["id_col"]].nunique())
    n_time = int(panel[panel_cfg["time_col"]].nunique())
    df_correction = bool(cfg_thr.get("df_correction", True))
    if not df_correction:
        dof = n                                     # legacy (biased) behaviour
    elif two_way:
        dof = (n_ent - 1) * (n_time - 1)
    else:
        dof = n_ent * (n_time - 1)

    grid = _make_grid(q, cfg_thr.get("trim", 0.05), cfg_thr.get("max_grid", 150))
    min_count = max(int(cfg_thr.get("min_regime_frac", 0.10) * n), Z.shape[1] + 2 * D.shape[1] + 2)
    rng = np.random.default_rng(seed)
    reps = int(cfg_thr.get("bootstrap_reps", 300))
    max_thr = int(cfg_thr.get("max_thresholds", 2))
    select_alpha = float(cfg_thr.get("select_alpha", 0.05))

    # Bootstrap configuration.
    method = cfg_thr.get("bootstrap_method", "block")
    blocks = _entity_blocks(ent) if method == "block" else None
    if method == "block" and blocks is None:
        LOGGER.warning("threshold bootstrap: panel unbalanced; falling back to i.i.d. resampling")
    use_alt = bool(cfg_thr.get("alt_residual_bootstrap", True))

    # Linear (no-threshold) model: single coefficient on D.
    lin = _fit_given(yd, Z, D, q, [], demean, min_count)
    ssr_prev = lin[0]

    thresholds: List[float] = []
    tests: List[dict] = []
    for k in range(1, max_thr + 1):
        th_k, fit_k = _sequential(yd, Z, D, q, k, grid, demean, min_count)
        if fit_k is None or len(th_k) < k:
            break
        ssr_k = fit_k[0]
        F_k = (ssr_prev - ssr_k) / (ssr_k / dof)
        # Alternative-model residuals for the error pool (Hansen Eq. 21).
        alt_resid = fit_k[2] if use_alt else None
        p_k = _bootstrap_pvalue(yd, Z, D, q, demean, min_count, grid, k, F_k,
                                reps, rng, dof, blocks=blocks, alt_resid=alt_resid)
        tests.append({"k": k, "F": F_k, "p_value": p_k, "reps": reps})
        LOGGER.info("threshold #%d: gamma=%s F=%.3f boot_p=%.3f",
                    k, [round(g, 4) for g in th_k], F_k, p_k)
        # Adopt the k-th threshold only if it is statistically significant.
        if np.isfinite(p_k) and p_k < select_alpha:
            thresholds = th_k
            ssr_prev = ssr_k
        else:
            break

    if not thresholds:                              # no significant threshold
        ssr, beta, resid, Xd = lin
        names = [f"{c}" for c in ctrl_cols] + [f"{r}" for r in reg_cols]
        vcov = _cluster_vcov(Xd, resid, clusters)
        table = _coef_table(names, beta, vcov, *Xd.shape)
        return ThresholdResult(y_col, q_col, [], table, tests,
                               (np.nan, np.nan), ssr, ssr / dof, n, dof=dof)

    # Final model at the selected thresholds.
    ssr, beta, resid, Xd = _fit_given(yd, Z, D, q, thresholds, demean, min_count)
    sigma2 = ssr / dof
    vcov = _cluster_vcov(Xd, resid, clusters)

    masks = _regime_masks(q, thresholds)
    names = list(ctrl_cols)
    for r in range(len(masks)):
        names += [f"{reg}__regime{r+1}" for reg in reg_cols]
    table = _coef_table(names, beta, vcov, *Xd.shape)

    # LR confidence interval for EVERY adopted threshold (Hansen sec. 5.3),
    # each profiled while holding the other threshold(s) at their estimates.
    alpha_ci = cfg_thr.get("alpha_ci", 0.95)
    report_all = bool(cfg_thr.get("report_all_cis", True))
    idxs = range(len(thresholds)) if report_all else [0]
    gamma_cis = [_gamma_ci(yd, Z, D, q, thresholds, demean, min_count, ssr, sigma2,
                           which=i, alpha=alpha_ci) for i in idxs]

    # Optional heteroskedasticity-/cluster-robust block-bootstrap gamma CI.
    gamma_cis_boot: List[Tuple[float, float]] = []
    if bool(cfg_thr.get("gamma_ci_bootstrap", False)):
        gamma_cis_boot = _gamma_ci_bootstrap(
            y, Z, D, q, ent, tim, grid, min_count, len(thresholds),
            reps=int(cfg_thr.get("gamma_ci_bootstrap_reps", 499)),
            alpha=alpha_ci, seed=seed + 7, blocks=blocks)

    return ThresholdResult(y_col, q_col, thresholds, table, tests, gamma_cis[0],
                           ssr, sigma2, n, gamma_cis=gamma_cis,
                           gamma_cis_boot=gamma_cis_boot, dof=dof)


def _gamma_ci(yd, Z, D, q, thresholds, demean, min_count, ssr_hat, sigma2,
              which: int = 0, alpha: float = 0.95) -> Tuple[float, float]:
    """
    Hansen's no-rejection-region LR confidence interval for the `which`-th
    threshold, profiling it while holding the other threshold(s) fixed at their
    estimates. crit = -2 ln(1 - sqrt(alpha)). sigma2 must already use the
    FE-corrected dof.
    """
    crit = -2.0 * np.log(1.0 - np.sqrt(alpha))
    others = [g for j, g in enumerate(thresholds) if j != which]
    g_hat = thresholds[which]
    g_lo, g_hi = g_hat, g_hat
    grid = _make_grid(q, 0.01, 400)
    for g in grid:
        if any(abs(g - o) < 1e-12 for o in others):
            continue
        out = _fit_given(yd, Z, D, q, sorted([g] + others), demean, min_count)
        if out is None:
            continue
        lr = (out[0] - ssr_hat) / sigma2
        if lr <= crit:
            g_lo, g_hi = min(g_lo, g), max(g_hi, g)
    return (g_lo, g_hi)


def _gamma_ci_bootstrap(y, Z, D, q, ent, tim, grid, min_count, n_thr,
                        reps, alpha, seed, blocks=None) -> List[Tuple[float, float]]:
    """
    Heteroskedasticity-/cluster-robust confidence interval for the threshold(s)
    by a nonparametric cluster (block) bootstrap: resample whole entities with
    replacement (their full y, regressors and threshold variable), re-estimate
    the threshold(s), and take percentile intervals across replications. Robust
    to within-city heteroskedasticity and serial correlation, complementing the
    homoskedastic LR interval.
    """
    if blocks is None:
        blocks = _entity_blocks(ent)
    if blocks is None:
        return []
    rng = np.random.default_rng(seed)
    n_ent = len(blocks)
    T = len(blocks[0])
    n_time = int(tim.max()) + 1 if tim is not None else T
    draws: List[List[float]] = [[] for _ in range(n_thr)]
    for _ in range(reps):
        donors = rng.integers(0, n_ent, size=n_ent)
        rows = np.concatenate([blocks[d] for d in donors])
        yb, Zb, Db, qb = y[rows], Z[rows], D[rows], q[rows]
        # Re-label pseudo-entities 0..n_ent-1 (preserve within-block dependence);
        # times repeat within each block in their original order.
        eb = np.repeat(np.arange(n_ent), T)
        tb = np.tile(np.arange(T), n_ent) if tim is not None else None
        dm = make_demeaner(eb, tb)
        ydb = dm(yb)
        th_b, fit_b = _sequential(ydb, Zb, Db, qb, n_thr, grid, dm, min_count)
        if fit_b is None or len(th_b) < n_thr:
            continue
        for i in range(n_thr):
            draws[i].append(th_b[i])
    lo_q, hi_q = 100 * (1 - alpha) / 2, 100 * (1 + alpha) / 2
    out = []
    for i in range(n_thr):
        if draws[i]:
            out.append((float(np.percentile(draws[i], lo_q)),
                        float(np.percentile(draws[i], hi_q))))
        else:
            out.append((np.nan, np.nan))
    return out


# =============================================================================
# CLI
# =============================================================================
@dataclass
class ExogeneityResult:
    threshold_var: str
    cf_coef: float            # coefficient on the first-stage residual (control function)
    cf_se: float
    t_stat: float
    p_value: float
    first_stage_f: float      # weak-instrument check on the lag in the first stage
    nobs: int
    endogenous: bool          # p_value < alpha

    def summary(self) -> str:
        verdict = ("REJECT exogeneity: treat SUHII as ENDOGENOUS threshold"
                   if self.endogenous else
                   "fail to reject: SUHII exogeneity is not rejected")
        return ("Seo-Shin threshold-exogeneity test (control function)\n"
                f"  threshold var: {self.threshold_var}  n={self.nobs}\n"
                f"  control-function coef = {self.cf_coef:.4f} (se {self.cf_se:.4f}, "
                f"t {self.t_stat:.2f}, p {self.p_value:.3f})\n"
                f"  first-stage F (lag instrument) = {self.first_stage_f:.2f}\n"
                f"  -> {verdict}")


def test_threshold_exogeneity(panel: pd.DataFrame, cfg_thr: dict, panel_cfg: dict,
                              cfg_endog: dict, thresholds: Sequence[float]
                              ) -> Optional[ExogeneityResult]:
    """
    Control-function (Smith-Blundell / Hausman) test for endogeneity of the
    threshold variable, the practical Seo & Shin (2016) diagnostic.

    Stage 1 (within-transformed): regress the threshold variable q on its own
    first lag plus the exogenous controls; take residual v_hat (the part of q
    not explained by predetermined information). Stage 2: re-fit the threshold
    regression at the estimated regime split with v_hat added as an extra
    regressor. Under H0 (q exogenous) the coefficient on v_hat is zero; a
    cluster-robust t-test rejecting H0 says q is endogenous to the outcome and
    the threshold should be instrumented. The lagged instrument costs the first
    sample year, so the test runs on the T-1 within panel.

    Returns None if there is no adopted threshold or the lag is unavailable.
    """
    if not thresholds:
        return None
    q_col = cfg_thr["threshold_var"]
    y_col = cfg_thr["outcome"]
    reg_cols = list(cfg_thr["regime_regressors"])
    ctrl_cols = list(cfg_thr["controls"])
    id_col, time_col = panel_cfg["id_col"], panel_cfg["time_col"]
    _require(panel, [y_col, q_col] + reg_cols + ctrl_cols, "test_threshold_exogeneity")

    # Build the within-city first lag of the threshold variable.
    p = panel.sort_values([id_col, time_col]).copy()
    p["_q_lag"] = p.groupby(id_col)[q_col].shift(1)
    p = p.dropna(subset=["_q_lag"]).reset_index(drop=True)
    if len(p) < (len(ctrl_cols) + len(reg_cols) + 4):
        return None

    two_way = cfg_thr.get("time_effects", True)
    ent = _factorize(p[id_col])
    tim = _factorize(p[time_col]) if two_way else None
    demean = make_demeaner(ent, tim)
    clusters = _factorize(p[cfg_thr.get("cluster", id_col)])

    q = p[q_col].to_numpy(float)
    Z = p[ctrl_cols].to_numpy(float)
    D = p[reg_cols].to_numpy(float)
    y = p[y_col].to_numpy(float)
    q_lag = p["_q_lag"].to_numpy(float)

    # ---- First stage: q ~ q_lag + controls (within) ----
    fs_X = demean(np.hstack([q_lag[:, None], Z]))
    qd = demean(q)
    _, fs_beta, v_hat = _ssr_beta_resid(qd, fs_X)
    # First-stage F on the single lag instrument (cluster-robust).
    fs_vcov = _cluster_vcov(fs_X, v_hat, clusters)
    f_lag = float((fs_beta[0] ** 2) / fs_vcov[0, 0]) if fs_vcov[0, 0] > 0 else np.inf

    # ---- Second stage: threshold regression at the estimated split + v_hat ----
    des, masks = _build_design(Z, D, q, thresholds)
    des = np.hstack([des, v_hat[:, None]])           # append the control function
    Xd = demean(des)
    yd = demean(y)
    _, beta, resid = _ssr_beta_resid(yd, Xd)
    vcov = _cluster_vcov(Xd, resid, clusters)
    j = Xd.shape[1] - 1                              # v_hat is the last column
    cf_coef = float(beta[j])
    cf_se = float(np.sqrt(vcov[j, j]))
    t_stat = cf_coef / cf_se if cf_se > 0 else np.nan
    p_value = float(2.0 * stats.norm.sf(abs(t_stat))) if np.isfinite(t_stat) else np.nan
    alpha = float(cfg_endog.get("alpha", 0.05))
    return ExogeneityResult(q_col, cf_coef, cf_se, t_stat, p_value, f_lag,
                            len(p), bool(np.isfinite(p_value) and p_value < alpha))


def _write_threshold_estimates(res: ThresholdResult, path: str) -> None:
    """Persist the estimated threshold(s) with per-threshold LR and bootstrap CIs."""
    if res.thresholds:
        k = len(res.thresholds)
        lr = res.gamma_cis if res.gamma_cis else [res.gamma_ci]
        lr = (lr + [(np.nan, np.nan)] * k)[:k]
        bt = (res.gamma_cis_boot + [(np.nan, np.nan)] * k)[:k]
        df = pd.DataFrame({
            "k": list(range(1, k + 1)),
            "gamma": res.thresholds,
            "lr_ci_lo": [c[0] for c in lr], "lr_ci_hi": [c[1] for c in lr],
            "boot_ci_lo": [c[0] for c in bt], "boot_ci_hi": [c[1] for c in bt],
        })
        # Back-compat aliases consumed by visualize.py (the LR CI is the default).
        df["ci_lo"], df["ci_hi"] = df["lr_ci_lo"], df["lr_ci_hi"]
    else:
        df = pd.DataFrame({"k": [], "gamma": [], "lr_ci_lo": [], "lr_ci_hi": [],
                           "boot_ci_lo": [], "boot_ci_hi": [], "ci_lo": [], "ci_hi": []})
    df.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark + Hansen panel threshold.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    panel = pd.read_parquet(args.panel or cfg["paths"]["panel_file"])
    panel = prepare_columns(panel, cfg)
    outdir = args.outdir or cfg["paths"]["outputs_dir"]
    os.makedirs(outdir, exist_ok=True)

    bench = fe_benchmark(panel, cfg["benchmark"], cfg["panel"])
    bench.to_csv(os.path.join(outdir, "benchmark_fe.csv"))

    res = fit_panel_threshold(panel, cfg["threshold"], cfg["panel"], seed=cfg["seed"])
    res.coef_table.to_csv(os.path.join(outdir, "threshold_coefficients.csv"))
    pd.DataFrame(res.tests).to_csv(os.path.join(outdir, "threshold_tests.csv"), index=False)
    _write_threshold_estimates(res, os.path.join(outdir, "threshold_estimates.csv"))
    with open(os.path.join(outdir, "threshold_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(res.summary() + "\n")

    LOGGER.info("benchmark + threshold results written to %s", outdir)
    print("\n" + res.summary())


if __name__ == "__main__":
    main()
