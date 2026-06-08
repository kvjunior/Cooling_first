#!/usr/bin/env python3
# =============================================================================
# causal_ml.py
# -----------------------------------------------------------------------------
# Stage 3d. The data-driven counterpart to the parametric Hansen threshold.
#
# Goal: recover theta(SUHII) = d(outcome)/d(digital productivity) as a SMOOTH
# function of urban heat, WITHOUT imposing a kink. If the digital-decarbonization
# dividend is "thermally gated", theta should be high at low SUHII and decline
# (toward / below zero) as SUHII rises -- and the decline should sit near the
# Hansen threshold estimated in threshold_models.py. Recovering that shape from a
# flexible estimator is the "we did not assume the nonlinearity, we found it"
# robustness check.
#
# Method: Double / Debiased Machine Learning (Chernozhukov et al., 2018) with a
# partially-linear, heterogeneous-coefficient final stage (Robinson, 1988):
#
#     Y_tilde = theta(X) * T_tilde + e
#       Y_tilde = Y - E[Y | W, X]      (cross-fitted ML residual)
#       T_tilde = T - E[T | W, X]      (cross-fitted ML residual)
#       theta(X) = B(X) c              (B = B-spline basis in X = SUHII)
#
# Fixed effects are partialled out by two-way (city + year) demeaning before the
# DML stage. Cross-fitting folds are grouped BY CITY (GroupKFold) so no city
# appears in both the training and evaluation folds of a nuisance model.
# Standard errors on theta(X) are cluster-robust by city.
#
# Compute note: the default nuisance learners (histogram gradient boosting) and
# the optional econml causal forest are CPU-parallel (joblib across the 64
# cores). The GPUs are only exercised by the OPTIONAL neural-network nuisance
# variant, which is not required and not enabled by default.
#
# Usage
#   python causal_ml.py --config config.yaml \
#       --panel data/panel_yreb.parquet --outdir outputs
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

# Reuse the panel/FE machinery from Stage 3 (single source of truth, no dup).
from threshold_models import _cluster_vcov, _factorize, make_demeaner

# Optional field-standard engine.
try:
    from econml.dml import CausalForestDML
    _HAS_ECONML = True
except Exception:                                    # pragma: no cover
    _HAS_ECONML = False

LOGGER = logging.getLogger("causal_ml")


# =============================================================================
# Nuisance learners and cross-fitting (the "double" / "debiased" machinery)
# =============================================================================
def _make_nuisance(kind: str, seed: int):
    """Factory for a fresh nuisance regressor."""
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.1, max_iter=300,
            l2_regularization=1.0, random_state=seed)
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    raise ValueError(f"unknown nuisance learner: {kind}")


def crossfit_residuals(Y: np.ndarray, T: np.ndarray, F: np.ndarray,
                       groups: np.ndarray, n_folds: int,
                       make_model: Callable[[int], object], seed: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-fitted Robinson residuals.

    For each fold, fit E[Y|F] and E[T|F] on the OTHER folds and predict on the
    held-out fold, so every observation gets an out-of-sample residual. Folds are
    grouped by `groups` (city) to respect panel dependence. F stacks confounders
    and the moderator: F = [W_demeaned | X_raw].
    """
    n = len(Y)
    Yhat = np.empty(n)
    That = np.empty(n)
    gkf = GroupKFold(n_splits=n_folds)
    for fold, (tr, te) in enumerate(gkf.split(F, Y, groups=groups)):
        my = make_model(seed + fold)
        mt = make_model(seed + 1000 + fold)
        my.fit(F[tr], Y[tr])
        mt.fit(F[tr], T[tr])
        Yhat[te] = my.predict(F[te])
        That[te] = mt.predict(F[te])
    return Y - Yhat, T - That


# =============================================================================
# Spline varying-coefficient final stage:  Y_tilde = theta(X) * T_tilde + e
# =============================================================================
def _fit_spline_transformer(x: np.ndarray, n_knots: int, degree: int
                            ) -> SplineTransformer:
    st = SplineTransformer(n_knots=n_knots, degree=degree,
                           include_bias=True, extrapolation="constant")
    st.fit(x.reshape(-1, 1))
    return st


@dataclass
class ThetaCurve:
    grid: np.ndarray
    theta: np.ndarray
    se: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    overlap: Optional[np.ndarray] = None     # E[V^2 | SUHII] local-overlap diagnostic

    def to_frame(self) -> pd.DataFrame:
        d = {"suhii": self.grid, "theta": self.theta,
             "se": self.se, "ci_lo": self.ci_lo, "ci_hi": self.ci_hi}
        if self.overlap is not None:
            d["overlap"] = self.overlap
        return pd.DataFrame(d)


def _penalized_cluster_vcov(X: np.ndarray, resid: np.ndarray,
                            cluster_codes: np.ndarray, bread: np.ndarray) -> np.ndarray:
    """
    Cluster-robust sandwich with a caller-supplied bread = (M'M + lam P)^{-1}.
    Mirrors threshold_models._cluster_vcov but uses the penalized bread so the
    roughness penalty is reflected in the standard errors. With bread = pinv(M'M)
    (lam = 0) this reduces to the ordinary cluster-robust covariance.
    """
    n, k = X.shape
    G = int(cluster_codes.max()) + 1
    meat = np.zeros((k, k))
    for g in range(G):
        idx = cluster_codes == g
        s = X[idx].T @ resid[idx]
        meat += np.outer(s, s)
    corr = (G / (G - 1)) * ((n - 1) / (n - k))
    return corr * bread @ meat @ bread


def _second_difference_operator(p: int) -> np.ndarray:
    """
    (p-2, p) second-difference matrix D2 such that ||D2 c||^2 is the discrete
    integrated squared curvature of the spline coefficients c. This is the
    roughness penalty Omega(theta) of the R-learner (Nie & Wager, 2021): it
    shrinks theta(x) toward a straight line, not toward zero, so the level of the
    effect is not biased -- only wiggliness is penalized.
    """
    if p < 3:
        return np.zeros((0, p))
    D = np.zeros((p - 2, p))
    for i in range(p - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def fit_varying_coefficient(Yt: np.ndarray, Tt: np.ndarray, x: np.ndarray,
                            clusters: np.ndarray, grid: np.ndarray,
                            n_knots: int = 6, degree: int = 3,
                            alpha_ci: float = 0.95, lam: float = 0.0,
                            st: Optional[SplineTransformer] = None) -> ThetaCurve:
    """
    Estimate theta(x) = B(x) c by regressing Y_tilde on (T_tilde (x) B(x)) with an
    optional second-difference roughness penalty (the R-learner penalty), and
    return the curve on `grid` with cluster-robust confidence bands.

    Penalized normal equations:  (M'M + lam D2'D2) c = M'Y_tilde,
    with M = T_tilde (x) B(x). The penalty bread P = M'M + lam D2'D2 enters the
    cluster-robust sandwich so the bands reflect the shrinkage. lam = 0 recovers
    the unpenalized estimator exactly. A pre-fit spline transformer `st` may be
    supplied so repeated splits share an identical basis.
    """
    if st is None:
        st = _fit_spline_transformer(x, n_knots, degree)
    B = st.transform(x.reshape(-1, 1))               # (n, p)
    M = Tt[:, None] * B                              # interaction design (n, p)
    p = M.shape[1]
    D2 = _second_difference_operator(p)
    P = M.T @ M + lam * (D2.T @ D2)                  # penalized bread
    P_inv = np.linalg.pinv(P)
    c = P_inv @ (M.T @ Yt)
    resid = Yt - M @ c
    # Penalty-aware cluster-robust sandwich: P^{-1} (sum_g M_g' u_g u_g' M_g) P^{-1}.
    Vc = _penalized_cluster_vcov(M, resid, clusters, P_inv)

    Bg = st.transform(grid.reshape(-1, 1))           # (g, p)
    theta = Bg @ c
    var = np.einsum("gi,ij,gj->g", Bg, Vc, Bg)       # row-wise B V B'
    se = np.sqrt(np.clip(var, 0.0, None))
    z = stats.norm.ppf(0.5 + alpha_ci / 2.0)
    return ThetaCurve(grid, theta, se, theta - z * se, theta + z * se)


def select_penalty_cv(Yt: np.ndarray, Tt: np.ndarray, x: np.ndarray,
                      groups: np.ndarray, lambda_grid: Sequence[float],
                      n_knots: int, degree: int, n_folds: int) -> Tuple[float, dict]:
    """
    Choose the roughness penalty lambda by city-grouped CV on the R-loss
    (Nie & Wager, 2021): for each lambda, fit theta on training cities and
    evaluate sum (Y_tilde - theta(x) T_tilde)^2 on held-out cities. A single
    shared spline basis (fit on all x) keeps the design identical across folds
    and lambdas. Returns (best_lambda, {lambda: mean_cv_loss}).
    """
    lambda_grid = list(lambda_grid)
    if len(lambda_grid) == 1:
        return float(lambda_grid[0]), {float(lambda_grid[0]): np.nan}
    st = _fit_spline_transformer(x, n_knots, degree)
    B = st.transform(x.reshape(-1, 1))
    p = B.shape[1]
    D2 = _second_difference_operator(p)
    PtP = D2.T @ D2
    gkf = GroupKFold(n_splits=n_folds)
    losses = {float(l): [] for l in lambda_grid}
    for tr, te in gkf.split(B, Yt, groups=groups):
        Mtr = Tt[tr, None] * B[tr]
        Mte = Tt[te, None] * B[te]
        MtM = Mtr.T @ Mtr
        MtY = Mtr.T @ Yt[tr]
        for l in lambda_grid:
            c = np.linalg.pinv(MtM + l * PtP) @ MtY
            r = Yt[te] - Mte @ c                      # held-out R-loss residual
            losses[float(l)].append(float(r @ r))
    mean_loss = {l: float(np.mean(v)) for l, v in losses.items()}
    best = min(mean_loss, key=mean_loss.get)
    return best, mean_loss


def overlap_diagnostic(Tt: np.ndarray, x: np.ndarray, grid: np.ndarray,
                       n_knots: int, degree: int) -> np.ndarray:
    """
    Local-overlap diagnostic E[V^2 | SUHII], V = residualized treatment T_tilde.
    DML identifies theta(x) only where the treatment retains variation after
    residualization; small E[V^2|x] => weak local identification and attenuated,
    wide theta there (the edge-attenuation the overlap condition explains). We
    estimate it by smoothing V^2 on a SUHII spline and evaluating on `grid`.
    """
    st = _fit_spline_transformer(x, n_knots, degree)
    B = st.transform(x.reshape(-1, 1))
    v2 = Tt ** 2
    c, *_ = np.linalg.lstsq(B, v2, rcond=None)
    Bg = st.transform(grid.reshape(-1, 1))
    return np.clip(Bg @ c, 0.0, None)


def partial_linear_ate(Yt: np.ndarray, Tt: np.ndarray, clusters: np.ndarray
                       ) -> Tuple[float, float]:
    """Constant-effect DML average treatment effect (regress Y_tilde on T_tilde)."""
    M = Tt[:, None]
    beta, *_ = np.linalg.lstsq(M, Yt, rcond=None)
    resid = Yt - M @ beta
    V = _cluster_vcov(M, resid, clusters)
    return float(beta[0]), float(np.sqrt(V[0, 0]))


def locate_kink(grid: np.ndarray, theta: np.ndarray) -> float:
    """
    Data-driven analogue of the Hansen threshold: the SUHII level at which
    theta(SUHII) declines most steeply (the onset of the 'cooling tax').
    """
    d = np.gradient(theta, grid)
    return float(grid[int(np.argmin(d))])


# =============================================================================
# Optional engine: econml CausalForestDML
# =============================================================================
def fit_causal_forest(Yt, Tt, X, n_trees, seed, grid, alpha_ci) -> ThetaCurve:
    """Heterogeneous effect via econml CausalForestDML (optional)."""
    if not _HAS_ECONML:                              # pragma: no cover
        raise ImportError("econml is not installed; set use_causal_forest=false "
                          "or `conda install -c conda-forge econml`.")
    est = CausalForestDML(model_y="auto", model_t="auto",
                          n_estimators=n_trees, random_state=seed)
    est.fit(Yt, Tt, X=X.reshape(-1, 1))
    g = grid.reshape(-1, 1)
    theta = est.effect(g).ravel()
    lo, hi = est.effect_interval(g, alpha=1 - alpha_ci)
    lo, hi = lo.ravel(), hi.ravel()
    se = (hi - lo) / (2.0 * stats.norm.ppf(0.5 + alpha_ci / 2.0))
    return ThetaCurve(grid, theta, se, lo, hi)


# =============================================================================
# Orchestration
# =============================================================================
@dataclass
class CausalMLResult:
    curve: pd.DataFrame
    ate: float
    ate_se: float
    kink_suhii: float
    nobs: int
    engine: str
    lam: float = 0.0
    n_repeats: int = 1

    def summary(self) -> str:
        z = self.ate / self.ate_se if self.ate_se else np.nan
        lo, hi = self.curve["theta"].iloc[0], self.curve["theta"].iloc[-1]
        s = (f"Causal ML (DML, engine={self.engine})  n={self.nobs}\n"
             f"  cross-fit repeats (S-split median): {self.n_repeats}\n"
             f"  roughness penalty lambda: {self.lam:.4g}\n"
             f"  ATE (constant effect of DP): {self.ate:.4f} (se {self.ate_se:.4f}, z {z:.2f})\n"
             f"  theta at lowest SUHII : {lo:.4f}\n"
             f"  theta at highest SUHII: {hi:.4f}\n"
             f"  data-driven kink (steepest decline) at SUHII = {self.kink_suhii:.4f}")
        if "overlap" in self.curve.columns:
            ov = self.curve["overlap"].to_numpy()
            s += (f"\n  overlap E[V^2|SUHII]: min {ov.min():.4g} at "
                  f"SUHII={self.curve['suhii'].iloc[int(np.argmin(ov))]:.3f} "
                  f"(weakest local identification)")
        return s


def run_causal_ml(panel: pd.DataFrame, cfg: dict) -> CausalMLResult:
    """Full DML pipeline: FE partialling -> cross-fit -> varying-coefficient theta(SUHII).

    Camera-ready refinements (Nie & Wager 2021; Chernozhukov et al. 2018):
      * second-difference roughness penalty on the spline, lambda chosen by
        city-grouped CV on the R-loss (or a fixed float);
      * S-split median over `n_repeats` independent cross-fit partitions, which
        removes the dependence of the estimate on any single random fold split;
      * an E[V^2|SUHII] overlap diagnostic flagging where theta is weakly
        identified (the edge attenuation the overlap condition predicts).
    """
    cm = cfg["causal_ml"]
    pcfg = cfg["panel"]
    y_col, t_col, x_col = cm["outcome"], cm["treatment"], cm["moderator"]
    conf = list(cm["confounders"])
    for c in [y_col, t_col, x_col] + conf:
        if c not in panel.columns:
            raise KeyError(f"causal_ml: panel missing column '{c}'")

    ent = _factorize(panel[pcfg["id_col"]])
    tim = _factorize(panel[pcfg["time_col"]]) if cm.get("time_effects", True) else None
    demean = make_demeaner(ent, tim)
    clusters = _factorize(panel[cm.get("cluster", pcfg["id_col"])])

    # Partial out FE from outcome, treatment, and confounders; keep moderator raw.
    Yd = demean(panel[y_col].to_numpy(float))
    Td = demean(panel[t_col].to_numpy(float))
    Wd = demean(panel[conf].to_numpy(float))
    x = panel[x_col].to_numpy(float)
    F = np.hstack([Wd, x[:, None]])                  # nuisance features

    seed = int(cfg.get("seed", 0))
    n_folds = int(cm.get("n_folds", 5))
    n_knots = int(cm.get("spline_knots", 6))
    degree = int(cm.get("spline_degree", 3))
    alpha_ci = cm.get("alpha_ci", 0.95)

    trim = cm.get("trim", 0.05)
    lo, hi = np.quantile(x, [trim, 1 - trim])
    grid = np.linspace(lo, hi, int(cm.get("grid_points", 60)))

    # --- econml branch (unchanged engine; single split) ---
    if cm.get("use_causal_forest", False):
        Yt, Tt = crossfit_residuals(
            Yd, Td, F, groups=ent, n_folds=n_folds,
            make_model=lambda s: _make_nuisance(cm.get("nuisance", "hgb"), s), seed=seed)
        curve = fit_causal_forest(Yt, Tt, x, int(cm.get("forest_trees", 2000)),
                                  seed, grid, alpha_ci)
        ate, ate_se = partial_linear_ate(Yt, Tt, clusters)
        kink = locate_kink(curve.grid, curve.theta)
        LOGGER.info("causal_ml engine=econml ATE=%.4f kink@%.4f", ate, kink)
        return CausalMLResult(curve.to_frame(), ate, ate_se, kink, len(Yt),
                              "econml.CausalForestDML", lam=0.0, n_repeats=1)

    # --- S-split median over repeated cross-fit partitions ---
    n_repeats = max(1, int(cm.get("n_repeats", 1)))
    penalty_on = bool(cm.get("spline_penalty", False))
    lam_cfg = cm.get("spline_penalty_lambda", 0.0)
    lambda_grid = list(cm.get("lambda_grid", [0.0]))

    thetas, ses, ates, ate_ses, lams = [], [], [], [], []
    # One shared basis across repeats keeps theta comparable.
    st_shared = _fit_spline_transformer(x, n_knots, degree)
    overlap_acc = []
    for r in range(n_repeats):
        Yt, Tt = crossfit_residuals(
            Yd, Td, F, groups=ent, n_folds=n_folds,
            make_model=lambda s: _make_nuisance(cm.get("nuisance", "hgb"), s),
            seed=seed + 101 * r)
        # Roughness penalty: CV-select per repeat, fixed float, or none.
        if not penalty_on:
            lam = 0.0
        elif isinstance(lam_cfg, str) and lam_cfg.lower() == "cv":
            lam, _ = select_penalty_cv(Yt, Tt, x, ent, lambda_grid,
                                       n_knots, degree, n_folds)
        else:
            lam = float(lam_cfg)
        lams.append(lam)
        curve_r = fit_varying_coefficient(
            Yt, Tt, x, clusters, grid, n_knots=n_knots, degree=degree,
            alpha_ci=alpha_ci, lam=lam, st=st_shared)
        thetas.append(curve_r.theta)
        ses.append(curve_r.se)
        a, ase = partial_linear_ate(Yt, Tt, clusters)
        ates.append(a); ate_ses.append(ase)
        if cm.get("report_overlap", False):
            overlap_acc.append(overlap_diagnostic(Tt, x, grid, n_knots, degree))

    thetas = np.vstack(thetas)                       # (R, g)
    ses = np.vstack(ses)
    theta_med = np.median(thetas, axis=0)
    # Chernozhukov S-split variance: within-split variance (median) PLUS the
    # across-split dispersion of the point estimates.
    within = np.median(ses ** 2, axis=0)
    across = thetas.var(axis=0, ddof=1) if n_repeats > 1 else 0.0
    se_med = np.sqrt(np.clip(within + across, 0.0, None))
    z = stats.norm.ppf(0.5 + alpha_ci / 2.0)
    overlap = np.median(np.vstack(overlap_acc), axis=0) if overlap_acc else None
    curve = ThetaCurve(grid, theta_med, se_med,
                       theta_med - z * se_med, theta_med + z * se_med, overlap=overlap)

    ate = float(np.median(ates))
    ate_se = float(np.sqrt(np.median(np.array(ate_ses) ** 2)
                           + (np.var(ates, ddof=1) if n_repeats > 1 else 0.0)))
    lam_report = float(np.median(lams))
    kink = locate_kink(curve.grid, curve.theta)
    LOGGER.info("causal_ml engine=DML+spline repeats=%d lambda=%.4g ATE=%.4f "
                "theta[lo]=%.4f theta[hi]=%.4f kink@%.4f",
                n_repeats, lam_report, ate, curve.theta[0], curve.theta[-1], kink)
    return CausalMLResult(curve.to_frame(), ate, ate_se, kink, len(Yt),
                          "DML + spline varying-coefficient",
                          lam=lam_report, n_repeats=n_repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description="DML theta(SUHII) for the DP effect.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--panel", default=None)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    # lngdp is created by threshold_models.prepare_columns; replicate the need here.
    panel = pd.read_parquet(args.panel or cfg["paths"]["panel_file"])
    if "lngdp" not in panel.columns and "gdp" in panel.columns:
        panel["lngdp"] = np.log(panel["gdp"])

    res = run_causal_ml(panel, cfg)
    outdir = args.outdir or cfg["paths"]["outputs_dir"]
    os.makedirs(outdir, exist_ok=True)
    res.curve.to_csv(os.path.join(outdir, "causal_theta_curve.csv"), index=False)
    with open(os.path.join(outdir, "causal_ml_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(res.summary() + "\n")
    LOGGER.info("causal ML results written to %s", outdir)
    print("\n" + res.summary())


if __name__ == "__main__":
    main()
