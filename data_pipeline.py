#!/usr/bin/env python3
# =============================================================================
# data_pipeline.py
# -----------------------------------------------------------------------------
# Stage 1 of the codebase. Turns raw, heterogeneous sources into a single,
# validated, strictly balanced city-year panel ready for measurement.py.
#
# Two modes
#   real       : read user-supplied CSVs from paths.raw_dir (schemas in config)
#   synthetic  : fabricate a schema-correct, deterministic panel so the entire
#                pipeline + downstream models run end-to-end without the
#                proprietary/registration-gated source files. This is what
#                makes the repository reproducible out-of-the-box and CI-able.
#
# Sources (real mode), all declared in config.yaml -> raw:
#   co2.csv      CEADs city CO2 inventories      (Shan et al., 2018, 2019)
#   yearbook.csv China City Statistical Yearbook (NBS)
#   dfi.csv      PKU-DFIIC                        (Guo et al., 2020)
#   suhii.csv    Global UHII dataset              (Yang, 2024)
#                (annual clear-sky surface daytime UHII; optional robustness
#                 facet columns suhii_terra / suhii_allsky / suhii_summer)
#   cdd.csv      Cooling degree-days from ERA5-Land 2-m AIR temperature
#                (NOT surface LST: CDD is an air-temperature integral, and an
#                 independent source keeps the SUHII->CDD mediation clean)
#   metadata.csv city_id / name / province / lon / lat
#
# Usage
#   python data_pipeline.py --config config.yaml                 # real mode
#   python data_pipeline.py --config config.yaml --synthetic     # synthetic
#   python data_pipeline.py --config config.yaml --synthetic \
#           --out data/panel_yreb.parquet --metadata-out data/city_metadata.csv
#
# Output
#   * a balanced panel parquet (paths.panel_file)
#   * a city metadata csv      (paths.metadata_file)
# =============================================================================
from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger("data_pipeline")

# Indicator / variable groups that every downstream stage relies on. Kept here
# (not duplicated) so a schema change is a one-line edit.
_DEA_POSITIVE = ["labor", "capital", "energy", "gdp", "co2"]


# -----------------------------------------------------------------------------
# Configuration & logging
# -----------------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Load the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> None:
    """Configure root logging to stderr and (optionally) a file."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


# -----------------------------------------------------------------------------
# Real-mode loaders. Each validates its required columns and returns a tidy
# frame keyed by (city_id, year), except metadata which is keyed by city_id.
# -----------------------------------------------------------------------------
def _read_csv_checked(path: str, required: List[str], name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{name}] expected source file not found: {path}. "
            f"Provide it (schema columns: {required}) or run with --synthetic."
        )
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] {path} is missing required columns: {missing}")
    LOGGER.info("loaded %-9s rows=%d cols=%d", name, len(df), df.shape[1])
    return df


def load_sources(cfg: dict) -> Dict[str, pd.DataFrame]:
    """Read every source declared in config -> raw (real mode)."""
    raw_dir = cfg["paths"]["raw_dir"]
    raw = cfg["raw"]
    frames: Dict[str, pd.DataFrame] = {}
    for key in ("metadata", "co2", "yearbook", "dfi", "suhii", "cdd"):
        spec = raw[key]
        frames[key] = _read_csv_checked(
            os.path.join(raw_dir, spec["file"]), spec["columns"], key
        )
    # Optional deflators.
    defl_path = os.path.join(raw_dir, raw["deflators"]["file"])
    if os.path.exists(defl_path):
        frames["deflators"] = _read_csv_checked(
            defl_path, raw["deflators"]["columns"], "deflators"
        )
    else:
        LOGGER.warning("no deflators.csv found; monetary deflation will be skipped")
    return frames


# -----------------------------------------------------------------------------
# Merge + metadata enrichment
# -----------------------------------------------------------------------------
def _province_to_reach(cfg: dict) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for reach, provinces in cfg["reaches"].items():
        for prov in provinces:
            mapping[prov] = reach
    return mapping


def enrich_metadata(metadata: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach the upper/middle/lower reach label to each city via its province."""
    meta = metadata.copy()
    reach_map = _province_to_reach(cfg)
    meta[cfg["panel"]["reach_col"]] = meta["province"].map(reach_map)
    unknown = meta[meta[cfg["panel"]["reach_col"]].isna()]["province"].unique()
    if len(unknown):
        raise ValueError(f"provinces not assigned to any reach: {list(unknown)}")
    return meta


def merge_panel(frames: Dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """Inner-merge all city-year sources, then left-join city metadata."""
    keys = ["city_id", "year"]
    panel = frames["yearbook"]
    for key in ("co2", "dfi", "suhii", "cdd"):
        panel = panel.merge(frames[key], on=keys, how="inner", validate="one_to_one")
    meta = enrich_metadata(frames["metadata"], cfg)
    panel = panel.merge(
        meta[["city_id", "city_name", "province", "lon", "lat",
              cfg["panel"]["reach_col"]]],
        on="city_id", how="left", validate="many_to_one",
    )
    LOGGER.info("merged panel rows=%d unique_cities=%d years=%s",
                len(panel), panel["city_id"].nunique(),
                sorted(panel["year"].unique().tolist()))
    return panel


# -----------------------------------------------------------------------------
# Preprocessing: balance enforcement, deflation, winsorization, positivity
# -----------------------------------------------------------------------------
def enforce_balanced(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Restrict to cities observed in every year of the study window."""
    y0, y1 = cfg["study"]["start_year"], cfg["study"]["end_year"]
    years = list(range(y0, y1 + 1))
    panel = panel[panel["year"].between(y0, y1)].copy()
    counts = panel.groupby("city_id")["year"].nunique()
    complete = counts[counts == len(years)].index
    dropped = sorted(set(panel["city_id"].unique()) - set(complete))
    if dropped and cfg["preprocess"]["drop_incomplete_cities"]:
        LOGGER.warning("dropping %d incomplete cities: %s", len(dropped), dropped[:10])
        panel = panel[panel["city_id"].isin(complete)].copy()
    return panel


def deflate_monetary(panel: pd.DataFrame, deflators: pd.DataFrame | None,
                     cfg: dict) -> pd.DataFrame:
    """Convert nominal monetary columns to constant base-year prices."""
    pp = cfg["preprocess"]
    if not pp["deflate"]:
        return panel
    if deflators is None:
        LOGGER.warning("deflate=true but no deflators provided; leaving nominal")
        return panel
    base = pp["base_year"]
    cpi = deflators.set_index("year")["cpi"]
    if base not in cpi.index:
        raise ValueError(f"base_year {base} absent from deflators.csv")
    factor = (cpi / cpi.loc[base]).to_dict()          # year -> price level
    out = panel.copy()
    f = out["year"].map(factor)
    for col in pp["monetary_cols"]:
        if col in out.columns:
            out[col] = out[col] / f
    LOGGER.info("deflated %d monetary columns to %d prices", len(pp["monetary_cols"]), base)
    return out


def winsorize(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pooled per-column quantile clipping to tame outliers."""
    pp = cfg["preprocess"]
    if not pp["winsorize"]:
        return panel
    lo_q, hi_q = pp["winsor_limits"]
    out = panel.copy()
    for col in pp["winsor_cols"]:
        if col in out.columns:
            lo, hi = out[col].quantile(lo_q), out[col].quantile(hi_q)
            out[col] = out[col].clip(lower=lo, upper=hi)
    LOGGER.info("winsorized %d columns at [%.2f, %.2f]", len(pp["winsor_cols"]), lo_q, hi_q)
    return out


def validate_panel(panel: pd.DataFrame, cfg: dict) -> None:
    """Hard checks: balance, no missing values, strict positivity for DEA."""
    y0, y1 = cfg["study"]["start_year"], cfg["study"]["end_year"]
    n_years = y1 - y0 + 1
    counts = panel.groupby("city_id")["year"].nunique()
    if not (counts == n_years).all():
        raise AssertionError("panel is not balanced across the study window")
    if panel.isna().any().any():
        bad = panel.columns[panel.isna().any()].tolist()
        raise AssertionError(f"panel contains missing values in: {bad}")
    nonpos = [c for c in _DEA_POSITIVE
              if c in panel.columns and (panel[c] <= 0).any()]
    if nonpos:
        raise AssertionError(f"DEA columns must be strictly positive; violated: {nonpos}")
    LOGGER.info("validation OK: balanced, complete, DEA-positive "
                "(cities=%d, years=%d)", panel["city_id"].nunique(), n_years)


# -----------------------------------------------------------------------------
# Synthetic generator (deterministic, schema-correct, economically plausible)
# -----------------------------------------------------------------------------
def make_synthetic(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a reproducible synthetic metadata + panel that satisfies the same
    schema and validation contract as real data. Designed so that:
      * a genuine DEA frontier exists (some efficient, some inefficient DMUs),
      * digital indicators differ systematically across reaches (lower > mid > upper),
      * SUHII drifts upward over time (warming),
      * CDD is an air-temperature-like quantity: correlated with SUHII through a
        shared climate component but carrying its OWN independent variation
        (cdd_noise_sd), so it is not a deterministic function of SUHII,
      * a genuine cooling-tax channel is planted: higher CDD adds a CO2 surcharge
        (cee_cdd_coef), so SUHII -> CDD -> higher CO2 -> lower CEE is recoverable
        by the mediation model (otherwise the synthetic mediation is degenerate),
      * robustness SUHII facets (Terra / all-sky / summer) are provided as extra
        columns so the robustness.suhii_facets sweeps are runnable,
      * all DEA inputs/outputs are strictly positive.
    """
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["study"]["n_cities"]
    y0, y1 = cfg["study"]["start_year"], cfg["study"]["end_year"]
    years = np.arange(y0, y1 + 1)
    syn = cfg["synthetic"]

    # ---- metadata: assign cities to provinces (and thus reaches) round-robin
    reach_provs = [(r, p) for r, ps in cfg["reaches"].items() for p in ps]
    provinces = [reach_provs[i % len(reach_provs)][1] for i in range(n)]
    meta = pd.DataFrame({
        "city_id": np.arange(1, n + 1),
        "city_name": [f"City_{i:03d}" for i in range(1, n + 1)],
        "province": provinces,
        "lon": rng.uniform(102.0, 122.0, n),   # YREB longitudinal span
        "lat": rng.uniform(24.0, 35.0, n),     # YREB latitudinal span
    })
    reach_map = _province_to_reach(cfg)
    reach_of = np.array([reach_map[p] for p in provinces])
    # Reach-level development multiplier (lower reaches more developed).
    dev_mult = np.select(
        [reach_of == "lower", reach_of == "middle", reach_of == "upper"],
        [1.6, 1.1, 0.8], default=1.0,
    )
    # City-level latent efficiency type in (0,1]; a few near-efficient cities.
    eff_type = rng.beta(5.0, 2.0, n)           # right-skewed toward efficient

    def noise(size):
        return np.exp(rng.normal(0.0, syn["noise_sd"], size))

    records = []
    base_suhii = 0.8 + 0.6 * (dev_mult - 1.0)      # hotter in denser/lower reaches
    cdd_noise_sd = float(syn.get("cdd_noise_sd", 30.0))
    cee_cdd_coef = float(syn.get("cee_cdd_coef", 0.0))
    for t, yr in enumerate(years):
        growth = 1.0 + 0.06 * t                # common technological progress

        # --- Heat and cooling demand (built first: CDD feeds the CO2 surcharge) ---
        # SUHII rises over time; hotter in denser/lower reaches.
        suhii = np.clip(base_suhii + syn["suhii_trend_per_year"] * t
                        + rng.normal(0, 0.15, n), 0.01, None)
        # CDD is air-temperature-like: a shared climate component proportional to
        # SUHII PLUS its own independent variation (so CDD != f(SUHII) exactly).
        cdd = np.clip(150.0 + syn["cdd_per_suhii"] * suhii
                      + rng.normal(0, cdd_noise_sd, n), 1.0, None)
        # Standardized cooling load drives the planted cooling-tax channel.
        cdd_z = (cdd - cdd.mean()) / (cdd.std() + 1e-9)

        # --- Inputs (strictly positive) ---
        labor = 50.0 * dev_mult * growth * noise(n)
        capital = 2.0e6 * dev_mult * growth * noise(n)
        energy = 400.0 * dev_mult * growth * noise(n)
        # Desired output rewards efficiency; undesired output penalizes it and
        # carries a COOLING-TAX SURCHARGE: higher cooling load -> more CO2 per
        # unit activity (AC + data-centre cooling on heat-stressed grids). With
        # cee_cdd_coef < 0 this lowers CEE where CDD is high -> the mediation
        # SUHII -> CDD -> CEE is genuine and recoverable.
        gdp = (1.5e6 * dev_mult * growth * eff_type * noise(n))
        co2 = (300.0 * dev_mult * growth * (1.3 - 0.6 * eff_type)
               * (1.0 - cee_cdd_coef * cdd_z)        # surcharge (coef<0 => +CO2 when hot)
               * noise(n))
        co2 = np.clip(co2, 1e-6, None)
        # Digital indicators scale with development; all strictly positive.
        base = dev_mult * growth
        rec = pd.DataFrame({
            "city_id": meta["city_id"].to_numpy(),
            "year": yr,
            "labor": labor, "capital": capital, "energy": energy,
            "gdp": gdp, "co2": co2,
            "mobile_penetration": np.clip(40 * base * noise(n), 1, 150),
            "internet_penetration": np.clip(25 * base * noise(n), 1, 120),
            "postal_per_capita": np.clip(80 * base * noise(n), 1, None),
            "telecom_per_capita": np.clip(600 * base * noise(n), 1, None),
            "ict_employment_share": np.clip(2.5 * base * noise(n), 0.05, 30),
            "scitech_expenditure_share": np.clip(1.5 * base * noise(n), 0.05, 20),
            "patents_5g": np.clip(rng.poisson(20 * base), 1, None),
            "patents_ecommerce": np.clip(rng.poisson(35 * base), 1, None),
            "university_teachers": np.clip(3000 * base * noise(n), 50, None),
            "dfi_breadth": np.clip(120 * base * noise(n), 1, None),
            "dfi_depth": np.clip(130 * base * noise(n), 1, None),
            "dfi_digitization": np.clip(150 * base * noise(n), 1, None),
            "popd": np.clip(400 * dev_mult * noise(n), 10, None),
            "urbanization": np.clip(0.45 * dev_mult * noise(n), 0.05, 0.99),
            "industrial_structure": np.clip(0.46 / np.sqrt(dev_mult) * noise(n), 0.1, 0.9),
            "gov_support": np.clip(1.0e5 * dev_mult * noise(n), 1, None),
            "greenery": np.clip(60 * noise(n), 5, 600),
        })
        rec["suhii"] = suhii
        rec["cdd"] = cdd
        # Robustness SUHII facets (Yang 2024): Terra (~10:30, slightly cooler),
        # all-sky (adds cloudy days, lower), summer (JJA, higher than annual).
        rec["suhii_terra"] = np.clip(suhii * 0.92 + rng.normal(0, 0.05, n), 0.01, None)
        rec["suhii_allsky"] = np.clip(suhii * 0.85 + rng.normal(0, 0.05, n), 0.01, None)
        rec["suhii_summer"] = np.clip(suhii * 1.35 + rng.normal(0, 0.10, n), 0.01, None)
        records.append(rec)

    panel = pd.concat(records, ignore_index=True)
    LOGGER.info("generated synthetic panel rows=%d cities=%d years=%d",
                len(panel), n, len(years))
    return meta, panel


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def build(cfg: dict, synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce (metadata, panel): either from raw sources or synthetically."""
    if synthetic:
        metadata, panel = make_synthetic(cfg)
        panel = panel.merge(
            enrich_metadata(metadata, cfg)[["city_id", "city_name", "province",
                                            "lon", "lat", cfg["panel"]["reach_col"]]],
            on="city_id", how="left", validate="many_to_one",
        )
        deflators = None
    else:
        frames = load_sources(cfg)
        metadata = enrich_metadata(frames["metadata"], cfg)
        panel = merge_panel(frames, cfg)
        deflators = frames.get("deflators")

    panel = enforce_balanced(panel, cfg)
    panel = deflate_monetary(panel, deflators, cfg)
    panel = winsorize(panel, cfg)
    panel = panel.sort_values(["city_id", "year"]).reset_index(drop=True)
    validate_panel(panel, cfg)
    return metadata, panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the balanced YREB panel.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="generate a schema-correct synthetic panel")
    parser.add_argument("--out", default=None, help="override panel output path")
    parser.add_argument("--metadata-out", default=None,
                        help="override metadata output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg["paths"].get("log_file"))

    metadata, panel = build(cfg, synthetic=args.synthetic)

    panel_path = args.out or cfg["paths"]["panel_file"]
    meta_path = args.metadata_out or cfg["paths"]["metadata_file"]
    os.makedirs(os.path.dirname(panel_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    metadata.to_csv(meta_path, index=False)
    LOGGER.info("wrote panel -> %s (%d rows)", panel_path, len(panel))
    LOGGER.info("wrote metadata -> %s (%d cities)", meta_path, len(metadata))


if __name__ == "__main__":
    main()
