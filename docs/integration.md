# `multileppat_vertex_batch` Integration Status and Interface Notes

Date: 2026-04-10

## Summary

This document records the current integrated state of the `multileppat_vertex_batch` package after the selection, fitting, and notebook refactor work carried out for the `JpsiJpsiPhi` mass-study workflow.

The package now provides:

- a vectorized candidate-selection path for ntuple-based `JpsiJpsiPhi` mass studies
- a common event-best candidate selection strategy
- a standard active-mass-window audit
- selector-by-selector fit comparison for both RooFit and `iminuit`
- a unified J/psi PDF configuration layer, including parameter locking presets

The mass-study notebooks
[study_mass_spectra_roofit_run2023d.ipynb](/eos/home-c/chiw/JpsiJpsiPhi/CMSSW_15_0_15_JpsiJpsiPhi_refactor/src/HeavyFlavorAnalysis/TPS-Onia2MuMu/doc/study_mass_spectra_roofit_run2023d.ipynb)
and
[study_mass_spectra_iminuit_run2023d.ipynb](/eos/home-c/chiw/JpsiJpsiPhi/CMSSW_15_0_15_JpsiJpsiPhi_refactor/src/HeavyFlavorAnalysis/TPS-Onia2MuMu/doc/study_mass_spectra_iminuit_run2023d.ipynb)
now use this package-level workflow instead of implementing their own candidate-selection logic inline.

## Environment and Tree Conventions

Use `LCG 109a` as the default environment:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109a/x86_64-el9-gcc13-opt/setup.sh
```

The standard trees for these ntuple studies are:

```text
mkcands/X_config
mkcands/X_data
```

The source of truth for branch groups is:

- [src/multileppat_vertex_batch/schema.py](../src/multileppat_vertex_batch/schema.py)

Relevant exported groups include:

- `CORE_DATA_BRANCHES`
- `RECO_SELECTION_BRANCHES`
- `MASS_STUDY_BRANCHES`
- `TRUTH_MATCH_BRANCHES`
- `DETAIL_ENTRY_BRANCHES`
- `MASS_BRANCHES_BY_MODE`
- `FIT_BRANCHES_BY_MODE`

## Current Public Interfaces

### Configuration

Defined in
[src/multileppat_vertex_batch/config.py](../src/multileppat_vertex_batch/config.py):

- `StudyConfig`
- `OfflineSelectionConfig`
- `MassStudyConfig`
- `default_mass_windows_from_config_row(...)`
- `resolve_windows(...)`

`MassStudyConfig` is the runtime bridge between selection and fit stages. The main fields are:

- `active_windows`
- `selector_name`
- `selectors`
- `best_candidate_metric`
- `fit_branches`

### Vectorized Selection

Defined in
[src/multileppat_vertex_batch/selection.py](../src/multileppat_vertex_batch/selection.py):

- `build_candidate_pool_for_file(...)`
- `build_candidate_pool_batch(...)`
- `select_best_candidates(...)`
- `summarize_selection(...)`
- `summarize_mass_window_flow(...)`
- `run_mass_selection_batch(...)`

Current selector support:

- `all6_same_recVtx`
- `Pri_fitValid`
- `Pri_fitPass`
- `Pri_passAny`

The canonical selector comparison in the notebooks uses:

```python
SELECTORS = ("all6_same_recVtx", "Pri_fitValid")
```

The best-candidate ranking is fixed to:

```text
Jpsi_1_pt^2 + Jpsi_2_pt^2 + Phi_pt^2
```

with tie-breakers:

- `Pri_VtxProb`
- `Phi_VtxProb`
- `cand_idx`

### Pipeline

Defined in
[src/multileppat_vertex_batch/pipeline.py](../src/multileppat_vertex_batch/pipeline.py):

- `run_truth_batch(...)`
- `run_massfit_prep_batch(...)`
- `run_roofit_selector_compare(...)`
- `run_iminuit_selector_compare(...)`
- legacy truth-cache helpers:
  - `run_batch(...)`
  - `load_or_build_cache(...)`
  - `validate_config_consistency(...)`

`run_massfit_prep_batch(...)` returns:

- `candidate_pool_df`
- `selected_candidate_df`
- `selected_for_selector_df`
- `fit_df`
- `audit_df`
- `window_audit_df`
- `selection_summary_df`

### Fit Backends

Defined in:

- [src/multileppat_vertex_batch/fit_roofit.py](../src/multileppat_vertex_batch/fit_roofit.py)
- [src/multileppat_vertex_batch/fit_iminuit.py](../src/multileppat_vertex_batch/fit_iminuit.py)

Shared fit-preparation contract:

- `build_fit_frame(selected_candidate_df, fit_branches, windows)`

RooFit path:

- `run_roofit_3d_jpsijpsiphi(...)`
- `projection_specs(...)`

iminuit path:

- `run_iminuit_3d_jpsijpsiphi(...)`
- `projection_curves_iminuit(...)`

Both selector-compare helpers return a per-selector summary table with at least:

- `selector`
- `n_candidates_after_full_offline`
- `n_events_with_candidate`
- `n_events_with_multiple_candidates`
- `n_fit_events`
- `fit_status`
- `N_sss`
- `N_sss_err`

The RooFit compare table also carries `covQual`.

### J/psi PDF Configuration

The common J/psi signal model is:

```text
Crystal Ball + Gaussian
```

This is now aligned between RooFit and `iminuit`.

Main interfaces:

- `FitParamSpec`
- `JpsiPdfConfig`
- `DEFAULT_JPSI_PDF_CONFIG`
- `JPSI_PDF_PRESETS`
- `resolve_jpsi_pdf_config(...)`
- `jpsi_pdf_config_table(...)`

Recommended small-sample usage:

```python
JPSI_PDF_PRESET = "small_sample"
JPSI_PDF_OVERRIDES = {}
JPSI1_PDF_CONFIG = resolve_jpsi_pdf_config(
    preset=JPSI_PDF_PRESET,
    overrides=JPSI_PDF_OVERRIDES,
)
JPSI2_PDF_CONFIG = resolve_jpsi_pdf_config(
    preset=JPSI_PDF_PRESET,
    overrides=JPSI_PDF_OVERRIDES,
)
```

## Notebook Usage Contract

The two Run2023D mass-study notebooks should now be treated as thin frontends.

They are expected to:

- read `X_config`
- construct `ACTIVE_WINDOWS`
- build `StudyConfig`, `OfflineSelectionConfig`, and `MassStudyConfig`
- call `run_massfit_prep_batch(...)`
- call one of:
  - `run_roofit_selector_compare(...)`
  - `run_iminuit_selector_compare(...)`
- use `SELECTOR_FOR_PLOTS` to choose the nominal displayed selector

They should not:

- re-implement candidate gathering
- re-implement vectorized muon or kaon selection
- re-implement event-best ranking
- re-implement selector-by-selector audit logic

## Verified Single-File Smoke-Test Results

Validated in `LCG 109a` on:

```text
/eos/user/c/chiw/JpsiJpsiPhi/rootNtuple/JpsiJpsiPhi_refactor_08Apr2026_Run2023Dv1_0_0000.root
```

using active windows:

- `Jpsi_1_mass`: `[2.9, 3.3]`
- `Jpsi_2_mass`: `[2.9, 3.3]`
- `Phi_mass`: `[0.99, 1.07]`

Vectorized selection summary:

- `all6_same_recVtx`
  - `57` candidates after full offline cuts
  - `27` selected events
  - `13` events with multiple passing candidates
- `Pri_fitValid`
  - `85` candidates after full offline cuts
  - `45` selected events
  - `18` events with multiple passing candidates

RooFit selector comparison on that file:

- `all6_same_recVtx`
  - `N_sss ≈ 4.609`
  - `N_sss_err ≈ 2.238`
  - `fit_status = 1`
  - `covQual = 3`
- `Pri_fitValid`
  - `N_sss ≈ 18.946`
  - `N_sss_err ≈ 4.923`
  - `fit_status = 1`
  - `covQual = 2`

iminuit selector comparison on that file:

- `all6_same_recVtx`
  - `N_sss ≈ 10.249`
  - `N_sss_err ≈ 3.281`
  - `fit_status = 1`
- `Pri_fitValid`
  - `N_sss ≈ 16.959`
  - `N_sss_err ≈ 4.693`
  - `fit_status = 1`

## Known Remaining Gap

The MC truth batch path is not yet fully migrated to the new candidate skeleton.

In particular,
[truth.py](/eos/home-c/chiw/JpsiJpsiPhi/CMSSW_15_0_15_JpsiJpsiPhi_refactor/src/HeavyFlavorAnalysis/TPS-Onia2MuMu/test/multileppat_vertex_batch/truth.py)
still retains the `_pythonize_event()`-based per-event Python expansion in `build_file_records(...)`.

That remains the main unfinished refactor target if the package is to become fully consistent between:

- reco/data selection studies
- mass-spectrum studies
- MC truth / vertex-validation studies
