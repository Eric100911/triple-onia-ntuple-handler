# `multileppat_vertex_batch`

`multileppat_vertex_batch` is a standalone Python repository for the ntuple-based `JpsiJpsiPhi`, `JpsiJpsiUps`, and `JpsiUpsPhi` offline-selection and selector-comparison studies in `HeavyFlavorAnalysis/TPS-Onia2MuMu`.

It replaces notebook-local selection code with a shared workflow that:

- reads `mkcands/X_data` and `mkcands/X_config` from MultiLepPAT ntuples
- auto-detects from `X_config` whether GEN/truth enrichment is available
- builds truth-labeled candidate and event tables when GEN branches exist
- applies vectorized offline selection for `JpsiJpsiPhi`, `JpsiJpsiUps`, and `JpsiUpsPhi`
- picks one best candidate per event with a fixed ranking
- prepares fit frames and compares selectors with RooFit and/or `iminuit`
- models `Ups_mass` with Gaussian `Υ(1S,2S,3S)` peaks plus an adjustable polynomial background of order `1` through `4`
- writes CMS-style fit projection plots with `mplhep`
- exports selector-specific fit-input candidates to a ROOT file for downstream work
- writes reproducible Parquet/JSON/ROOT output bundles with manifests

## Environment

Use `LCG 109a` for the standard notebook and batch workflow:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109a/x86_64-el9-gcc13-opt/setup.sh
```

Recommended runtime settings for this nested repo:

```bash
export MPLCONFIGDIR=/tmp/chiw/mplconfig_multileppat_vertex_batch
export PYTHONPYCACHEPREFIX=/tmp/chiw/pycache_multileppat_vertex_batch
```

Install the package in editable mode from the repository root:

```bash
cd /eos/home-c/chiw/JpsiJpsiPhi/CMSSW_15_0_15_JpsiJpsiPhi_refactor/src/HeavyFlavorAnalysis/TPS-Onia2MuMu/test/multileppat_vertex_batch
python -m pip install --no-deps -e .
```

Avoid relying on `PYTHONPATH=src` in this environment. In the current `LCG 109a` setup it can interfere with `site-packages` resolution for packages such as `pandas`.

Supported analysis modes:

- `JpsiJpsiPhi`
- `JpsiJpsiUps`
- `JpsiUpsPhi`

Standard tree paths:

```text
mkcands/X_config
mkcands/X_data
```

Branch groups are defined in [src/multileppat_vertex_batch/schema.py](./src/multileppat_vertex_batch/schema.py). Treat these as the source of truth rather than re-discovering branches in notebooks:

- `CORE_DATA_BRANCHES`
- `RECO_SELECTION_BRANCHES`
- `MASS_STUDY_BRANCHES`
- `TRUTH_MATCH_BRANCHES`
- `DETAIL_ENTRY_BRANCHES`
- `MASS_BRANCHES_BY_MODE`
- `FIT_BRANCHES_BY_MODE`

## Main Entry Points

Public imports are re-exported from [src/multileppat_vertex_batch/__init__.py](./src/multileppat_vertex_batch/__init__.py). The most important interfaces are:

- Configuration:
  - `StudyConfig`
  - `OfflineSelectionConfig`
  - `MassStudyConfig`
  - `default_mass_windows_from_config_row(...)`
  - `resolve_windows(...)`
- Truth and cache building:
  - `run_batch(...)`
  - `run_truth_batch(...)`
  - `load_or_build_cache(...)`
  - `validate_config_consistency(...)`
- Offline selection and fit preparation:
  - `build_candidate_pool_batch(...)`
  - `run_mass_selection_batch(...)`
  - `run_massfit_prep_batch(...)`
  - `summarize_mass_window_flow(...)`
- Fit comparison:
  - `run_roofit_selector_compare(...)`
  - `run_iminuit_selector_compare(...)`
  - `resolve_jpsi_pdf_config(...)`
  - `JPSI_PDF_PRESETS`
- Output writers:
  - `write_run_metadata(...)`
  - `write_truth_cache_bundle(...)`
  - `write_mass_selection_bundle(...)`
  - `write_fit_compare_bundle(...)`

## Package Layout

- [src/multileppat_vertex_batch/config.py](./src/multileppat_vertex_batch/config.py): dataclasses and active-window helpers
- [src/multileppat_vertex_batch/schema.py](./src/multileppat_vertex_batch/schema.py): branch lists, selector defaults, label maps, cache filenames
- [src/multileppat_vertex_batch/io.py](./src/multileppat_vertex_batch/io.py): ROOT reading, config snapshots, Parquet/JSON helpers
- [src/multileppat_vertex_batch/truth.py](./src/multileppat_vertex_batch/truth.py): truth matching, classifier construction, candidate/event row building
- [src/multileppat_vertex_batch/selection.py](./src/multileppat_vertex_batch/selection.py): vectorized offline selection, selector masks, best-candidate choice, audits
- [src/multileppat_vertex_batch/pipeline.py](./src/multileppat_vertex_batch/pipeline.py): end-to-end batch orchestration and selector comparison
- [src/multileppat_vertex_batch/fit_roofit.py](./src/multileppat_vertex_batch/fit_roofit.py): RooFit 3D mode-aware fit layer and shared J/psi PDF config
- [src/multileppat_vertex_batch/fit_iminuit.py](./src/multileppat_vertex_batch/fit_iminuit.py): `iminuit` implementation aligned to the RooFit signal model
- [src/multileppat_vertex_batch/cache.py](./src/multileppat_vertex_batch/cache.py): reproducible output bundles and manifests
- [src/multileppat_vertex_batch/audit.py](./src/multileppat_vertex_batch/audit.py): per-candidate drill-down tables for failure investigation
- [src/multileppat_vertex_batch/cli_truth_cache.py](./src/multileppat_vertex_batch/cli_truth_cache.py): minimal cache-builder CLI for quick truth/cache inspection
- [src/multileppat_vertex_batch/examples.py](./src/multileppat_vertex_batch/examples.py): helper utilities for choosing representative candidates

## Canonical Workflow

The standard driver is [src/multileppat_vertex_batch/cli_batch.py](./src/multileppat_vertex_batch/cli_batch.py). Install the package in editable mode before using it.

Typical usage:

```bash
cd /eos/home-c/chiw/JpsiJpsiPhi/CMSSW_15_0_15_JpsiJpsiPhi_refactor/src/HeavyFlavorAnalysis/TPS-Onia2MuMu/test/multileppat_vertex_batch
source /cvmfs/sft.cern.ch/lcg/views/LCG_109a/x86_64-el9-gcc13-opt/setup.sh
export MPLCONFIGDIR=/tmp/chiw/mplconfig_multileppat_vertex_batch \
&& export PYTHONPYCACHEPREFIX=/tmp/chiw/pycache_multileppat_vertex_batch \
&& python -m pip install --no-deps -e . \
&& run-multileppat-vertex-batch \
  --analysis-mode JpsiJpsiPhi \
  '/eos/user/c/chiw/JpsiJpsiPhi/MC_samples/Ntuple_refactor/TPS-JpsiJpsiPhi/JJP_TPS_001.root' \
  '/eos/user/c/chiw/JpsiJpsiPhi/MC_samples/Ntuple_refactor/TPS-JpsiJpsiPhi/JJP_TPS_002.root' \
  --output-dir '/eos/user/c/chiw/JpsiJpsiPhi/MC_samples/Ntuple_refactor/TPS-JpsiJpsiPhi/_multileppat_vertex_batch_jjp_tps' \
  --fit-backend roofit \
  --ups-background-order 4 \
  --cms-caption Preliminary \
  --cms-energy 13.6 \
  --cms-lumi 7.98 \
  --cms-era Run2022C
```

Run-3 data examples:

```bash
run-multileppat-vertex-batch \
  --analysis-mode JpsiJpsiUps \
  --output-dir /tmp/chiw/jpsijpsiups_batch \
  --fit-backend both \
  '/eos/user/c/chiw/JpsiJpsiUps/rootNtuple/P_Run20*-refactor_JpsiJpsiUps/*.root'
```

```bash
run-multileppat-vertex-batch \
  --analysis-mode JpsiUpsPhi \
  --output-dir /tmp/chiw/jpsiupsphi_batch \
  --fit-backend roofit \
  --ups-background-order 4 \
  --cms-caption 'Work In Progress' \
  --cms-energy 13.6 \
  --cms-lumi 283.4 \
  --cms-era 'Run 3 (2022-2025)' \
  '/eos/user/c/chiw/JpsiUpsPhi/rootNtuple/P_Run20*-JpsiUpsPhi_refactor_14Apr2026/*.root'
```

The driver performs up to three stages:

1. Truth/cache stage:
   - inspects `mkcands/X_config`
   - auto-enables GEN/truth only when `DoMonteCarloTree=True`
   - builds `candidate_df`, `event_df`, and `config_df`
   - writes the truth-cache bundle
2. Offline-selection stage:
   - builds selector-specific candidate pools
   - selects one best candidate per event
   - writes selection tables and audit summaries
   - writes `fit_input_candidates.root` with selector-specific TTrees for fit-selected candidates
3. Fit-comparison stage:
   - builds `fit_df` from the nominal selector
   - compares selectors with RooFit, `iminuit`, or both
   - for `JpsiJpsiUps` and `JpsiUpsPhi`, fits `Ups_mass` with Gaussian `Υ(1S,2S,3S)` components and a polynomial background of order `1..4`
   - writes per-selector fit summaries and yields
   - writes CMS-style `projection_*.png` plots beside each selector summary

## Minimal Python Usage

```python
import ROOT

from multileppat_vertex_batch.config import (
    CmsPlotStyleConfig,
    MassStudyConfig,
    OfflineSelectionConfig,
    StudyConfig,
    default_mass_windows_from_config_row,
    resolve_windows,
)
from multileppat_vertex_batch.fit_roofit import resolve_jpsi_pdf_config
from multileppat_vertex_batch.io import read_config_snapshot, resolve_input_files
from multileppat_vertex_batch.pipeline import (
    run_massfit_prep_batch,
    run_roofit_selector_compare,
)
from multileppat_vertex_batch.schema import FIT_BRANCHES_BY_MODE

files = resolve_input_files(
    [
        "/path/to/input_1.root",
        "/path/to/input_2.root",
    ]
)
study = StudyConfig(input_files=tuple(str(path) for path in files))
config_row = read_config_snapshot(files[0], study)

offline_cfg = OfflineSelectionConfig()
active_windows = resolve_windows(
    default_mass_windows_from_config_row(config_row),
    {
        "Jpsi_1_mass": offline_cfg.jpsi_mass_window,
        "Ups_mass": offline_cfg.ups_mass_window,
        "Phi_mass": offline_cfg.phi_mass_window,
        "Pri_mass": None,
    },
)
mass_cfg = MassStudyConfig(
    analysis_mode="JpsiUpsPhi",
    active_windows=active_windows,
    selector_name="all6_same_recVtx",
    selectors=("all6_same_recVtx", "Pri_fitValid"),
    fit_branches=tuple(FIT_BRANCHES_BY_MODE["JpsiUpsPhi"]),
)

tables = run_massfit_prep_batch(files, study, offline_cfg, mass_cfg)
roofit_payload = run_roofit_selector_compare(
    root_module=ROOT,
    selected_candidate_df=tables["selected_candidate_df"],
    selection_summary_df=tables["selection_summary_df"],
    study_cfg=mass_cfg,
    jpsi1_pdf_config=resolve_jpsi_pdf_config("small_sample"),
    jpsi2_pdf_config=resolve_jpsi_pdf_config("small_sample"),
    jpsi_pdf_preset="small_sample",
)

plot_style = CmsPlotStyleConfig(
    caption="Work In Progress",
    energy_tev=13.6,
    lumi_fb=283.4,
    era="Run 3 (2022-2025)",
    is_data=True,
)
```

## Standard Outputs

Truth-cache stage:

- `candidate_rows.parquet`
- `event_rows.parquet`
- `config_rows.parquet`
- `truth_summary.json`
- `config_consistency.json`
- `manifest.json`

Mass-selection stage:

- `candidate_pool_df.parquet`
- `selected_candidate_df.parquet`
- `selected_for_selector_df.parquet`
- `fit_df.parquet`
- `fit_input_candidates.root`
- `audit_df.parquet`
- `window_audit_df.parquet`
- `selection_summary_df.parquet`
- `manifest.json`

Fit-comparison stage:

- `selector_compare_df.parquet`
- per-selector `yield_table.parquet`
- per-selector `fit_summary.json`
- per-selector `projection_<fit_branch>.png` files such as `projection_Jpsi_1_mass.png`, `projection_Jpsi_2_mass.png`, `projection_Ups_mass.png`, or `projection_Phi_mass.png`
- RooFit-only `phi_debug_values.json` when present
- stage and per-selector `manifest.json`

## Behavioral Contract

These defaults are relied on by the notebooks and downstream studies and should not change silently:

- `all6_same_recVtx` means the mode-specific muon and kaon `vertexId` values are all equal and non-negative.
- `Pri_fitValid == 1` is the selector for a valid three-body vertex fit in the active analysis mode.
- The default selector comparison uses `("all6_same_recVtx", "Pri_fitValid")`.
- Best-candidate ranking is the sum of squared `pt` values for the three fitted objects in the active analysis mode, with mode-aware tie-breakers from `ANALYSIS_MODE_SPECS`.
- The shared J/psi signal model in both fit backends is `Crystal Ball + Gaussian`.
- `Ups_mass` is modeled with Gaussian `Υ(1S,2S,3S)` peaks plus a polynomial background of order `1..4`.
- J/psi parameter locking should go through `resolve_jpsi_pdf_config(...)` and `JPSI_PDF_PRESETS`, not notebook-local ad hoc parameter fixing.
- CMS-style projection plots should be produced with `mplhep` when fit payloads are non-empty, using run metadata supplied by the driver.
- Data ntuples should remain valid inputs without GEN branches; MC truth is enabled from `X_config`, not by notebook-local assumptions.
- Notebook code should call package entry points rather than re-implement vectorized selection, event-best ranking, or selector-comparison logic inline.

## Related Files

- Package integration notes: [docs/integration.md](./docs/integration.md)
- Reference batch driver: [src/multileppat_vertex_batch/cli_batch.py](./src/multileppat_vertex_batch/cli_batch.py)
