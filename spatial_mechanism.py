#!/usr/bin/env python3
# =============================================================================
# spatial_mechanism.py
# -----------------------------------------------------------------------------
# Stage 4. Spatial spillovers, the cooling-tax mechanism, and warming dynamics.
#
#   build_weights        Row-standardized spatial weight matrix W over the YREB
#                        cities (inverse-distance / k-nearest geographic, or
#                        economic-distance).
#
#   morans_i             Global Moran's I (per year) with analytical inference,
#                        establishing that the outcome is spatially clustered.
#
#   fit_sdm              Maximum-likelihood Spatial Durbin Model with fixed
#                        effects, y = rho W y + X b + W X t + e, plus
#                        LeSage-Pace direct / indirect / total effects. Estimated
#                        self-contained (concentrated log-likelihood in rho via
#                        scipy), so it does not depend on a spatial-panel package.
#
#   mediation_fe         The "cooling tax": a fixed-effects causal-mediation test
#                        of heat (SUHII) -> cooling demand (CDD) -> carbon
#                        efficiency (CEE), with a cluster/entity block bootstrap
#                        for the indirect effect.
#
#   threshold_crossing   Share of cities above the SUHII threshold each year and
#                        a fixed-effects warming-trend test -- the empirical
#                        content of "warming erodes the digital dividend".
#
# Reuses the FE / cluster machinery from threshold_models.py (single source).
#
# Usage
#   python spatial_mechanism.py --config config.yaml \
#       --panel data/panel_yreb.parquet --metadata data/city_metadata.csv \
#       --gamma 1.20 --outdir outputs
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.optimize import minimize_scalar

from threshold_models import _cluster_vcov, _factorize, make_demeaner

LOGGER = logging.getLogger("spatial_mechanism")

_EPS = 1e-12


# =============================================================================
# Spatial weights
# =============================================================================
def _haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distance (km) among points given lat/lon degrees."""
    R = 6371.0088
    la = np.radians(lat)[:, None]
    lo = np.radians(lon)[:, None]
    dlat = la - la.T
    dlon = lo - lo.T
    a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _row_standardize(W: np.ndarray) -> np.ndarray:
    """Scale each row to sum to one (rows of all zeros are left as zeros)."""
    rs = W.sum(axis=1, keepdims=True)
    return np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)


def build_weights(meta: pd.DataFrame, cfg_sp: dict,
                  panel: Optional[pd.DataFrame] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a row-standardized W and return (W, city_order) where city_order are
    the city_id values in W's row/column order (sorted ascending).
    """
    meta = meta.sort_values("city_id").reset_index(drop=True)
    city_order = meta["city_id"].to_numpy()
    n = len(meta)
    kind = cfg_sp.get("weights", "geographic")

    if kind == "geographic":
        D = _haversine_km(meta["lat"].to_numpy(float), meta["lon"].to_numpy(float))
        np.fill_diagonal(D, np.inf)
        knn = int(cfg_sp.get("knn", 0))
        if knn > 0:
            W = np.zeros((n, n))
            order = np.argsort(D, axis=1)[:, :knn]
            for i in range(n):
                W[i, order[i]] = 1.0
        else:
            decay = float(cfg_sp.get("distance_decay", 1.0))
            W = 1.0 / np.power(D, decay)
            np.fill_diagonal(W, 0.0)
    elif kind == "economic":
        if panel is None:
            raise ValueError("economic weights require the panel to derive levels")
        lvl_col = cfg_sp.get("economic_level", "lngdp")
        if lvl_col not in panel.columns and lvl_col == "lngdp" and "gdp" in panel.columns:
            panel = panel.assign(lngdp=np.log(panel["gdp"]))
        lvl = (panel.groupby("city_id")[lvl_col].mean()
               .reindex(city_order).to_numpy(float))
        diff = np.abs(lvl[:, None] - lvl[None, :])
        np.fill_diagonal(diff, np.inf)
        W = 1.0 / (diff + _EPS)
        np.fill_diagonal(W, 0.0)
    else:
        raise ValueError(f"unknown weights kind: {kind}")

    return _row_standardize(W), city_order


def _w_index(panel: pd.DataFrame, city_order: np.ndarray, id_col: str) -> np.ndarray:
    """Map each panel row to its index (0..N-1) in W's city ordering."""
    pos = {cid: i for i, cid in enumerate(city_order)}
    idx = panel[id_col].map(pos)
    if idx.isna().any():
        raise ValueError("panel contains cities absent from the weight matrix")
    return idx.to_numpy(int)


def spatial_lag(values: np.ndarray, w_index: np.ndarray,
                year_codes: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute W applied within each period. `values` may be 1-D (n,) or 2-D (n,k);
    the lag is W @ value for the cross-section of cities in each year.
    """
    v2 = values if values.ndim == 2 else values.reshape(-1, 1)
    out = np.empty_like(v2, dtype=float)
    N = W.shape[0]
    for yr in np.unique(year_codes):
        rows = np.where(year_codes == yr)[0]
        wi = w_index[rows]
        mat = np.empty((N, v2.shape[1]))
        mat[wi] = v2[rows]
        lag = W @ mat
        out[rows] = lag[wi]
    return out if values.ndim == 2 else out.ravel()


# =============================================================================
# Moran's I (global), per year, with analytical inference
# =============================================================================
def morans_i(x: np.ndarray, W: np.ndarray) -> dict:
    """Global Moran's I with the normality-assumption z-test for one cross-section."""
    n = len(x)
    z = x - x.mean()
    S0 = W.sum()
    num = z @ (W @ z)
    den = z @ z
    I = (n / S0) * (num / den) if den > 0 and S0 > 0 else np.nan
    EI = -1.0 / (n - 1)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)
    varI = ((n ** 2 * S1 - n * S2 + 3 * S0 ** 2) / ((n ** 2 - 1) * S0 ** 2)) - EI ** 2
    zscore = (I - EI) / np.sqrt(varI) if varI > 0 else np.nan
    p = 2.0 * stats.norm.sf(abs(zscore)) if np.isfinite(zscore) else np.nan
    return {"I": float(I), "E_I": float(EI), "z": float(zscore), "p_value": float(p)}


def morans_i_by_year(panel: pd.DataFrame, col: str, W: np.ndarray,
                     w_index: np.ndarray, year_codes: np.ndarray,
                     years: np.ndarray) -> pd.DataFrame:
    """Per-year global Moran's I for `col`."""
    N = W.shape[0]
    rows_out = []
    for yc, yr in zip(np.unique(year_codes), np.unique(years)):
        rows = np.where(year_codes == yc)[0]
        wi = w_index[rows]
        vec = np.empty(N)
        vec[wi] = panel[col].to_numpy(float)[rows]
        res = morans_i(vec, W)
        res["year"] = int(yr)
        rows_out.append(res)
    return pd.DataFrame(rows_out)[["year", "I", "E_I", "z", "p_value"]]


# =============================================================================
# Spatial Durbin Model (ML, fixed effects) + LeSage-Pace effects
# =============================================================================
@dataclass
class SDMResult:
    rho: float
    rho_se: float
    coef_table: pd.DataFrame          # beta (X) and theta (WX)
    effects: pd.DataFrame             # direct / indirect / total (+ SE/z/p/CI) per regressor
    loglik: float
    nobs: int
    moran: pd.DataFrame
    dof: int = 0                      # effective residual dof for sigma^2
    bias_corrected: bool = False      # Lee & Yu (2010) small-T correction applied

    def summary(self) -> str:
        tag = "Lee-Yu bias-corrected" if self.bias_corrected else "uncorrected"
        lines = [f"Spatial Durbin Model  n={self.nobs}  rho={self.rho:.4f} "
                 f"(se {self.rho_se:.4f})  logL={self.loglik:.2f}  [{tag}, dof={self.dof}]",
                 self.coef_table.to_string(float_format=lambda v: f"{v:.4f}"),
                 "LeSage-Pace effects (simulation-based inference):",
                 self.effects.to_string(float_format=lambda v: f"{v:.4f}")]
        return "\n".join(lines)


def _make_fe_demeaner(panel, pcfg, fe):
    if fe == "none":
        return lambda M: (M.astype(float) if M.ndim == 2 else M.astype(float))
    ent = _factorize(panel[pcfg["id_col"]])
    tim = _factorize(panel[pcfg["time_col"]]) if fe == "twoway" else None
    return make_demeaner(ent, tim)


def _effects_point(rho: float, gamma: np.ndarray, W: np.ndarray, p: int):
    """LeSage-Pace direct/indirect/total for each regressor at a given (rho, gamma).
    M_k = (I - rho W)^{-1} (beta_k I + theta_k W); direct = tr(M_k)/N, total = sum(M_k)/N."""
    N = W.shape[0]
    A_inv = np.linalg.inv(np.eye(N) - rho * W)
    direct = np.empty(p); total = np.empty(p)
    for k in range(p):
        Mk = A_inv @ (gamma[k] * np.eye(N) + gamma[p + k] * W)
        direct[k] = np.trace(Mk) / N
        total[k] = Mk.sum() / N
    return direct, total, (total - direct)


def _lesage_pace_effects(rho, gamma, W, regs, Vg, rho_se, n_sim, alpha, rng):
    """
    Simulation-based inference for the LeSage-Pace effects (LeSage & Pace 2009,
    sec. 2.7): draw parameter vectors from their estimated sampling distribution
    and recompute direct/indirect/total each draw, then summarise as SE + z + p +
    percentile CI. rho is drawn independently as N(rho, rho_se^2); the slope
    vector gamma as N(gamma, Vg). (rho and gamma are asymptotically independent
    in the concentrated likelihood, so the product draw is appropriate.)
    """
    p = len(regs)
    d0, t0, i0 = _effects_point(rho, gamma, W, p)
    sims = {"direct": np.empty((n_sim, p)), "indirect": np.empty((n_sim, p)),
            "total": np.empty((n_sim, p))}
    lo_b, hi_b = rho_bounds_default = (-0.999, 0.999)
    ok = 0
    for s in range(n_sim):
        rho_s = float(np.clip(rng.normal(rho, rho_se if np.isfinite(rho_se) else 0.0), lo_b, hi_b))
        gamma_s = rng.multivariate_normal(gamma, Vg)
        try:
            ds, ts, is_ = _effects_point(rho_s, gamma_s, W, p)
        except np.linalg.LinAlgError:               # singular I - rho_s W (rare)
            continue
        sims["direct"][ok], sims["total"][ok], sims["indirect"][ok] = ds, ts, is_
        ok += 1
    for key in sims:
        sims[key] = sims[key][:ok]

    z = stats.norm.ppf(0.5 + alpha / 2.0)
    lo_q, hi_q = 100 * (1 - alpha) / 2, 100 * (1 + alpha) / 2
    rows = []
    point = {"direct": d0, "indirect": i0, "total": t0}
    for k, r in enumerate(regs):
        row = {"regressor": r}
        for key in ("direct", "indirect", "total"):
            est = point[key][k]
            col = sims[key][:, k]
            se = float(col.std(ddof=1)) if len(col) > 1 else np.nan
            zz = est / se if se and se > 0 else np.nan
            row[key] = est
            row[f"{key}_se"] = se
            row[f"{key}_z"] = zz
            row[f"{key}_p"] = float(2 * stats.norm.sf(abs(zz))) if np.isfinite(zz) else np.nan
            row[f"{key}_lo"] = float(np.percentile(col, lo_q)) if len(col) else np.nan
            row[f"{key}_hi"] = float(np.percentile(col, hi_q)) if len(col) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("regressor")


def fit_sdm(panel: pd.DataFrame, cfg_sp: dict, pcfg: dict,
            W: np.ndarray, w_index: np.ndarray) -> SDMResult:
    """
    ML estimation of the FE Spatial Durbin Model with the Lee & Yu (2010)
    small-T bias correction, plus LeSage-Pace effects with simulation-based SEs.

    Two corrections (Lee & Yu 2010), active when `bias_correction` is set and the
    panel is demeaned:
      (1) Variance / dof: the within transform costs one df per entity (and, for
          two-way, per period). sigma^2 uses the FE-corrected dof
          [(n-1)(T-1) two-way; n(T-1) entity-only] instead of nT, removing the
          Neyman-Scott downward bias (~(T-1)/T) that deflates every SE.
      (2) Jacobian: time-demeaning a row-normalized W removes its unit
          eigenvalue, so the concentrated log-likelihood's spatial Jacobian must
          carry an additive -(T-1)*log(1-rho) term (their Lemma A.2). Omitting it
          tilts the objective and shifts rho_hat at small n. The term is applied
          only under two-way demeaning (the case that removes the eigenvalue).

    Scope note: this implements the dominant-order corrections (the Jacobian
    tilt and the (T-1)/T variance bias). A residual O(1/T) incidental-parameters
    bias in rho remains at very small T and shrinks as T grows (verified
    monotone in T); fully removing it needs Lee & Yu's analytic score term
    B_hat_n/n (eq. 34), which is negligible at this paper's n=110 and is left as
    a documented limitation rather than coded.
    """
    y_col = cfg_sp["outcome"]
    regs = list(cfg_sp["regressors"])
    fe = cfg_sp.get("fe", "twoway")
    bias_correction = bool(cfg_sp.get("bias_correction", True))
    demean = _make_fe_demeaner(panel, pcfg, fe)
    year_codes = _factorize(panel[pcfg["time_col"]])

    y = demean(panel[y_col].to_numpy(float))
    X = demean(panel[regs].to_numpy(float))
    WX = spatial_lag(X, w_index, year_codes, W)
    Wy = spatial_lag(y, w_index, year_codes, W)
    Z = np.hstack([X, WX])                          # [beta | theta] design
    n = len(y)
    N = W.shape[0]
    Tn = n // N
    Imat = np.eye(N)
    lo, hi = cfg_sp.get("rho_bounds", [-0.99, 0.99])

    # Effective residual dof for sigma^2 (Lee & Yu 2010). Entity-only FE costs
    # N means; two-way additionally costs (T-1) time means.
    if not bias_correction or fe == "none":
        dof = n
    elif fe == "twoway":
        dof = (N - 1) * (Tn - 1)
    else:                                            # entity-only
        dof = N * (Tn - 1)

    # Jacobian correction term active only under two-way demeaning of a
    # row-normalized W (where the removed eigenvalue is exactly 1).
    jac_correction = bias_correction and fe == "twoway"

    def _jacobian(rho: float) -> float:
        _, logdet = np.linalg.slogdet(Imat - rho * W)
        jac = Tn * logdet
        if jac_correction:
            jac -= (Tn - 1) * np.log1p(-rho)         # -(T-1) log(1 - rho)
        return jac

    def neg_profile_ll(rho: float) -> float:
        Ay = y - rho * Wy
        gamma, *_ = np.linalg.lstsq(Z, Ay, rcond=None)
        resid = Ay - Z @ gamma
        sigma2 = (resid @ resid) / dof              # FE-corrected denominator
        ll = -(dof / 2) * (np.log(2 * np.pi) + 1) - (dof / 2) * np.log(sigma2 + _EPS) + _jacobian(rho)
        return -ll

    opt = minimize_scalar(neg_profile_ll, bounds=(lo, hi), method="bounded")
    rho = float(opt.x)
    loglik = -opt.fun

    # Conditional coefficient estimates and SEs at rho_hat, using corrected sigma^2.
    Ay = y - rho * Wy
    gamma, *_ = np.linalg.lstsq(Z, Ay, rcond=None)
    resid = Ay - Z @ gamma
    sigma2 = (resid @ resid) / dof
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    Vg = sigma2 * ZtZ_inv
    se = np.sqrt(np.clip(np.diag(Vg), 0, None))
    names = [f"{r}" for r in regs] + [f"W_{r}" for r in regs]
    tvals = np.where(se > 0, gamma / se, np.nan)
    pvals = 2.0 * stats.norm.sf(np.abs(tvals))
    coef_table = pd.DataFrame({"coef": gamma, "se": se, "z": tvals, "p_value": pvals},
                              index=names)

    # rho SE via numerical second derivative of the (corrected) profile log-likelihood.
    h = 1e-3
    ll0 = -neg_profile_ll(rho)
    llp = -neg_profile_ll(min(rho + h, hi - 1e-6))
    llm = -neg_profile_ll(max(rho - h, lo + 1e-6))
    info = -(llp - 2 * ll0 + llm) / (h ** 2)
    rho_se = float(np.sqrt(1.0 / info)) if info > 0 else np.nan

    # LeSage-Pace direct/indirect/total effects with simulation-based inference,
    # drawing (rho, gamma) from their estimated sampling distribution (the
    # corrected covariance) and recomputing the effects each draw.
    effects = _lesage_pace_effects(
        rho, gamma, W, regs, Vg, rho_se,
        n_sim=int(cfg_sp.get("effects_n_sim", 1000)),
        alpha=float(cfg_sp.get("effects_alpha_ci", 0.95)),
        rng=np.random.default_rng(cfg_sp.get("seed", 0)))

    LOGGER.info("SDM rho=%.4f (se %.4f) logL=%.2f dof=%d corrected=%s core(%s) direct=%.4f total=%.4f",
                rho, rho_se, loglik, dof, bias_correction, regs[0],
                effects.loc[regs[0], "direct"], effects.loc[regs[0], "total"])
    return SDMResult(rho, rho_se, coef_table, effects, loglik, n, pd.DataFrame(),
                     dof=dof, bias_corrected=bias_correction)


# =============================================================================
# Cooling-tax mediation (fixed effects, block bootstrap)
# =============================================================================
@dataclass
class MediationResult:
    a: float
    b: float
    direct: float
    indirect: float
    total: float
    prop_mediated: float
    indirect_ci: Tuple[float, float]
    sobel_p: float
    nobs: int

    def summary(self) -> str:
        return (f"Cooling-tax mediation  n={self.nobs}\n"
                f"  a (treat->mediator)      : {self.a:.4f}\n"
                f"  b (mediator->outcome|treat): {self.b:.4f}\n"
                f"  direct effect            : {self.direct:.4f}\n"
                f"  indirect (a*b)           : {self.indirect:.4f}  "
                f"95% CI [{self.indirect_ci[0]:.4f}, {self.indirect_ci[1]:.4f}]  "
                f"Sobel p={self.sobel_p:.4f}\n"
                f"  total effect             : {self.total:.4f}\n"
                f"  proportion mediated      : {self.prop_mediated:.3f}")


def _fe_ols(y, X, demean, clusters):
    """Within OLS; returns (beta, se, resid) with cluster-robust SE."""
    yd = demean(y)
    Xd = demean(X)
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid = yd - Xd @ beta
    V = _cluster_vcov(Xd, resid, clusters)
    return beta, np.sqrt(np.clip(np.diag(V), 0, None)), resid


def mediation_fe(panel: pd.DataFrame, cfg_med: dict, pcfg: dict,
                 seed: int = 0) -> MediationResult:
    """FE causal mediation of treatment -> mediator -> outcome (the cooling tax)."""
    t_col, m_col, y_col = cfg_med["treatment"], cfg_med["mediator"], cfg_med["outcome"]
    ctrl = list(cfg_med["controls"])
    fe = cfg_med.get("fe", "twoway")

    # Optionally z-score the mediator. The indirect effect a*b is invariant to
    # this rescaling; it only makes a and b individually interpretable and keeps
    # the bootstrap well-conditioned when the mediator (e.g. CDD) is on a very
    # different scale from the treatment (e.g. SUHII).
    if cfg_med.get("standardize_mediator", False):
        m = panel[m_col].to_numpy(float)
        sd = m.std()
        if sd > _EPS:
            panel = panel.assign(**{m_col: (m - m.mean()) / sd})

    demean = _make_fe_demeaner(panel, pcfg, fe)
    clusters = _factorize(panel[cfg_med.get("cluster", pcfg["id_col"])])

    T = panel[t_col].to_numpy(float)
    Mv = panel[m_col].to_numpy(float)
    Y = panel[y_col].to_numpy(float)
    C = panel[ctrl].to_numpy(float) if ctrl else np.empty((len(T), 0))

    # Equation 1: M = a*T + controls.
    X1 = np.hstack([T[:, None], C])
    b1, se1, _ = _fe_ols(Mv, X1, demean, clusters)
    a, a_se = b1[0], se1[0]
    # Equation 2: Y = c'*T + b*M + controls.
    X2 = np.hstack([T[:, None], Mv[:, None], C])
    b2, se2, _ = _fe_ols(Y, X2, demean, clusters)
    cprime, b, b_se = b2[0], b2[1], se2[1]

    indirect = a * b
    direct = cprime
    total = direct + indirect
    prop = indirect / total if abs(total) > _EPS else np.nan

    # Sobel test.
    sobel_se = np.sqrt(b ** 2 * a_se ** 2 + a ** 2 * b_se ** 2)
    sobel_z = indirect / sobel_se if sobel_se > 0 else np.nan
    sobel_p = 2.0 * stats.norm.sf(abs(sobel_z)) if np.isfinite(sobel_z) else np.nan

    # Entity block bootstrap for the indirect-effect CI.
    rng = np.random.default_rng(seed)
    ids = panel[pcfg["id_col"]].to_numpy()
    uniq = np.unique(ids)
    by_id = {u: np.where(ids == u)[0] for u in uniq}
    reps = int(cfg_med.get("bootstrap_reps", 1000))
    boot = np.empty(reps)
    for r in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_id[u] for u in pick])
        sub = panel.iloc[idx]
        dm = _make_fe_demeaner(sub, pcfg, fe)
        cl = _factorize(sub[cfg_med.get("cluster", pcfg["id_col"])])
        Tb = sub[t_col].to_numpy(float); Mb = sub[m_col].to_numpy(float)
        Yb = sub[y_col].to_numpy(float)
        Cb = sub[ctrl].to_numpy(float) if ctrl else np.empty((len(Tb), 0))
        ab, _, _ = _fe_ols(Mb, np.hstack([Tb[:, None], Cb]), dm, cl)
        bb, _, _ = _fe_ols(Yb, np.hstack([Tb[:, None], Mb[:, None], Cb]), dm, cl)
        boot[r] = ab[0] * bb[1]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    LOGGER.info("mediation a=%.4f b=%.4f indirect=%.4f CI=[%.4f,%.4f] prop=%.3f",
                a, b, indirect, ci[0], ci[1], prop)
    return MediationResult(a, b, direct, indirect, total, prop, ci, sobel_p, len(T))


# =============================================================================
# Threshold-crossing dynamics (warming erodes the dividend)
# =============================================================================
def threshold_crossing(panel: pd.DataFrame, gamma: float, cfg_dyn: dict,
                       pcfg: dict) -> Tuple[pd.DataFrame, dict]:
    """Per-year share of cities above the SUHII threshold + a FE warming trend."""
    q_col = cfg_dyn.get("suhii_col", "suhii")
    by_year = (panel.assign(_above=(panel[q_col] > gamma).astype(float))
               .groupby(pcfg["time_col"])
               .agg(mean_suhii=(q_col, "mean"),
                    share_above=("_above", "mean"),
                    n_above=("_above", "sum"))
               .reset_index())

    # FE warming trend: suhii_it = lambda * year + a_i + e.
    ent = _factorize(panel[pcfg["id_col"]])
    demean = make_demeaner(ent, None)
    yr = panel[pcfg["time_col"]].to_numpy(float)
    qd = demean(panel[q_col].to_numpy(float))
    xd = demean(yr.reshape(-1, 1))
    slope, *_ = np.linalg.lstsq(xd, qd, rcond=None)
    resid = qd - xd @ slope
    V = _cluster_vcov(xd, resid, ent)
    se = float(np.sqrt(max(V[0, 0], 0)))
    z = slope[0] / se if se > 0 else np.nan
    trend = {"warming_per_year": float(slope[0]), "se": se,
             "z": float(z), "p_value": float(2.0 * stats.norm.sf(abs(z)))}
    LOGGER.info("dynamics gamma=%.4f share_above %.2f->%.2f  warming=%.4f/yr p=%.4g",
                gamma, by_year["share_above"].iloc[0], by_year["share_above"].iloc[-1],
                trend["warming_per_year"], trend["p_value"])
    return by_year, trend


# =============================================================================
# Orchestration
# =============================================================================
def run_spatial_mechanism(panel: pd.DataFrame, meta: pd.DataFrame, cfg: dict,
                          gamma: Optional[float] = None) -> dict:
    """Run weights, Moran's I, SDM, mediation, and dynamics; return a results dict."""
    pcfg = cfg["panel"]
    if "lngdp" not in panel.columns and "gdp" in panel.columns:
        panel = panel.assign(lngdp=np.log(panel["gdp"]))

    W, city_order = build_weights(meta, cfg["spatial"], panel)
    w_index = _w_index(panel, city_order, pcfg["id_col"])
    year_codes = _factorize(panel[pcfg["time_col"]])
    years = panel[pcfg["time_col"]].to_numpy()

    moran = morans_i_by_year(panel, cfg["spatial"]["outcome"], W,
                             w_index, year_codes, years)
    sdm = fit_sdm(panel, cfg["spatial"], pcfg, W, w_index)
    sdm.moran = moran

    med = mediation_fe(panel, cfg["mediation"], pcfg, seed=int(cfg.get("seed", 0)))

    if gamma is None:
        gamma = cfg["dynamics"].get("gamma")
    if gamma is None:
        gamma = float(np.median(panel[cfg["dynamics"]["suhii_col"]]))
        LOGGER.warning("no gamma supplied; using sample median SUHII=%.4f", gamma)
    dyn_year, dyn_trend = threshold_crossing(panel, gamma, cfg["dynamics"], pcfg)

    return {"W": W, "moran": moran, "sdm": sdm, "mediation": med,
            "dynamics_year": dyn_year, "dynamics_trend": dyn_trend, "gamma": gamma}


def main() -> None:
    parser = argparse.ArgumentParser(description="SDM + mechanism + dynamics.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--gamma", type=float, default=None,
                        help="SUHII threshold from Stage 3b (for dynamics)")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    panel = pd.read_parquet(args.panel or cfg["paths"]["panel_file"])
    meta = pd.read_csv(args.metadata or cfg["paths"]["metadata_file"])
    res = run_spatial_mechanism(panel, meta, cfg, gamma=args.gamma)

    outdir = args.outdir or cfg["paths"]["outputs_dir"]
    os.makedirs(outdir, exist_ok=True)
    res["moran"].to_csv(os.path.join(outdir, "morans_i_by_year.csv"), index=False)
    res["sdm"].coef_table.to_csv(os.path.join(outdir, "sdm_coefficients.csv"))
    res["sdm"].effects.to_csv(os.path.join(outdir, "sdm_effects.csv"))
    res["dynamics_year"].to_csv(os.path.join(outdir, "dynamics_by_year.csv"), index=False)
    with open(os.path.join(outdir, "spatial_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(res["sdm"].summary() + "\n\n")
        fh.write(res["mediation"].summary() + "\n\n")
        fh.write(f"Warming trend: {res['dynamics_trend']}\n(gamma={res['gamma']:.4f})\n")

    LOGGER.info("spatial mechanism results written to %s", outdir)
    print("\n" + res["sdm"].summary())
    print("\n" + res["mediation"].summary())


if __name__ == "__main__":
    main()
