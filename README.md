# Cooling first, decarbonizing second? — Replication package

Anonymized replication code for the manuscript

> **Cooling first, decarbonizing second? Urban heat island thresholds in the
> digital productivity–carbon efficiency coupling across China's Yangtze River
> Economic Belt.**

This repository is shared for **double-anonymized peer review**. It contains no
author, institutional, or funding information by design. Please do not add any
identifying material while the manuscript is under review.

Anonymous repository: `https://anonymous.4open.science/r/Cooling_first-ADC0/`

---

## 1. What this code does

The pipeline assembles a strictly balanced panel of **110 prefecture-level
cities** of the Yangtze River Economic Belt (YREB) over **2011–2020**
(1,100 city-years) and estimates whether the link between **digital
productivity (DP)** and the **DP–carbon-emission-efficiency (CEE) coupling
coordination degree (CCD)** is gated by a **surface urban heat island intensity
(SUHII)** threshold. It recovers the threshold twice — parametrically (Hansen
panel threshold regression) and non-parametrically (a double-machine-learning
R-learner) — tests a "cooling-tax" mechanism (heat → cooling degree-days →
CEE) by mediation, and characterizes spatial spillovers with a spatial Durbin
model. Running it reproduces every table and figure in the manuscript.

---

## 2. Repository structure

| File | Role |
| --- | --- |
| `run.py` | Pipeline orchestrator. Runs the full study end-to-end (measurement → threshold → causal ML → spatial/mechanism → figures) and writes all tables and figures to the output directory. |
| `config.yaml` | Single source of truth for paths, the random seed, bootstrap replications, the SUHII/income threshold settings, the CDD base temperatures (18 °C and the 26 °C robustness setpoint), winsorization limits, and spatial-weight options. |
| `data_pipeline.py` | Builds the balanced panel: loads raw sources, harmonizes city codes, deflates monetary series to constant 2011 prices, winsorizes continuous variables at the 1st/99th percentiles, drops cities with incomplete coverage, and constructs the spatial weight matrix **W** over city centroids. |
| `measurement.py` | Construct measurement: the super-SBM (super-efficiency slacks-based) carbon-emission-efficiency score, the entropy-weighted digital-productivity index, and the DP–CEE coupling coordination degree (CCD). |
| `threshold_models.py` | Hansen fixed-effects panel threshold regression: single-threshold estimation, bootstrap threshold tests (SUHII and the income placebo), and likelihood-ratio confidence intervals for γ. |
| `causal_ml.py` | Double-machine-learning R-learner for the heat-varying digital slope θ(SUHII): cross-fitted nuisance estimation, the orthogonalized second stage, cluster-robust inference, the data-driven kink, and the local-overlap diagnostic. |
| `spatial_mechanism.py` | Spatial Durbin model with direct/indirect/total effect decomposition, global Moran's I across years, and the cooling-tax mediation test (SUHII → CDD → CEE) with bootstrap and Sobel inference. |
| `visualize.py` | Regenerates the manuscript figures (study-area map, headline θ(SUHII) curve, regime structure, mediation, and dynamics) from the saved estimation output. |
| `environment.yml` | Conda environment specification pinning Python and all dependencies. |

> Module roles follow the manuscript; consult each file's docstring for the
> exact function-level interface.

---

## 3. Quick start

```bash
# 1. Create and activate the environment (name is defined in environment.yml)
conda env create -f environment.yml
conda activate <env-name-from-environment.yml>

# 2. Place the raw inputs where config.yaml expects them (see Section 4),
#    then review config.yaml (paths, seed, bootstrap reps).

# 3. Run the full pipeline
python run.py
```

All outputs (tables and figures) are written to the directory configured in
`config.yaml`. With the default settings, the run is fully deterministic given
the fixed random seed and bootstrap-replication count.

---

## 4. Data

The study draws on third-party datasets that are **not redistributed here** for
licensing reasons. Obtain each from its provider and place it where
`config.yaml` points; `data_pipeline.py` does the rest.

| Construct | Source |
| --- | --- |
| City CO₂ emissions | CEADs city-level emission inventories |
| Socioeconomic indicators / controls | China City Statistical Yearbook |
| Digital-finance sub-indices (for DP) | Peking University Digital Financial Inclusion Index |
| Surface urban heat island intensity (SUHII) | Global annual clear-sky daytime urban-heat dataset (Aqua overpass) |
| Cooling degree-days (CDD, mediator) | ERA5-Land 2-m **air** temperature |
| Land-surface temperature (robustness) | MODIS LST (held back as a measurement facet) |

**Identification note.** The treatment (SUHII) is a satellite **surface**
measure, while the mediator (CDD) is derived from ERA5-Land **air** temperature.
Drawing treatment and mediator from independent instruments avoids a mechanical
correlation from shared measurement, so an estimated SUHII → CDD → CEE pathway
reflects a physical channel rather than common measurement error. CDD is
accumulated above an 18 °C base, with a 26 °C setpoint re-estimated as a
robustness facet (both configurable in `config.yaml`).

---

## 5. What gets reproduced

Running `run.py` regenerates, in order:

1. The measurement layer — super-SBM CEE, entropy-weighted DP, and CCD, with
   the descriptive statistics.
2. The benchmark two-way fixed-effects estimate and the Hansen single-threshold
   results (γ, its LR and bootstrap intervals, and the regime-specific DP
   slopes), plus the income-threshold placebo.
3. The DML R-learner curve θ(SUHII) with its cluster-robust band, the
   zero-crossing and kink, and the overlap diagnostic.
4. The cooling-tax mediation (indirect effect with bootstrap and Sobel tests)
   and the spatial Durbin effect decomposition and Moran's I series.
5. The figures, written as vector PDF and high-resolution PNG.

---

## 6. Reproducibility notes

- The random seed and the number of bootstrap replications are set in
  `config.yaml`; change them there rather than in code.
- Figures use a Computer-Modern / Latin-Modern fallback so they render with or
  without a full TeX installation; the conceptual-framework diagram is provided
  separately as a TikZ source.
- Map figures carry the boundary disclaimer: *map lines delineate study areas
  and do not necessarily depict accepted national boundaries.*

---

## 7. Citation

Citation details are withheld during anonymized review and will be added on
acceptance. For now, please cite the manuscript by its title and the review
identifier assigned by the journal.

---

## 8. License

A permissive open-source license will be attached to this repository upon
publication. During review the code is provided solely for the purpose of
evaluating the manuscript.

---

## 9. Notes for reviewers

- Some figures shipped in the manuscript (the study-area map and the headline
  θ(SUHII) curve) can be driven either from the saved estimation output or from
  small intermediate CSVs; see the flags at the top of `visualize.py`.
- Portions of the figure-generation and documentation code were prepared with
  the assistance of a generative-AI tool, consistent with the manuscript's
  disclosure statement; all analytical results were produced by the code in
  this repository.
