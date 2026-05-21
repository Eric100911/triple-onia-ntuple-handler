# Acceptance and Efficiency Workflow

This note documents the `JpsiJpsiPhi` MC acceptance and efficiency workflow in `multileppat_vertex_batch`.

## Scope

The first supported analysis mode is:

```text
JpsiJpsiPhi
```

The workflow reads MultiLepPAT ntuples from:

```text
mkcands/X_data
```

and builds event-level efficiency numerators from reconstructed candidates that can be matched back to the generated `J/psi + J/psi + phi` system. The denominator is the number of full-GEN ntuple events containing:

```text
J/psi -> mu+ mu-
J/psi -> mu+ mu-
phi   -> K+ K-
```

This denominator is appropriate for MC ntuples produced with:

```text
RequireAcceptedCandidatesForMonteCarloTree = False
```

so events without accepted reconstructed candidates are still present.

## CLI Usage

Use the same `LCG 109a` environment as the rest of the package:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109a/x86_64-el9-gcc13-opt/setup.sh
export MPLCONFIGDIR=/tmp/chiw/mplconfig_multileppat_vertex_batch
export PYTHONPYCACHEPREFIX=/tmp/chiw/pycache_multileppat_vertex_batch
python -m pip install --no-deps -e .
```

Full IHEP XRootD sample discovery:

```bash
run-multileppat-efficiency \
  --analysis-mode JpsiJpsiPhi \
  --xrootd-host root://cceos.ihep.ac.cn// \
  --sample-root /eos/ihep/cms/store/user/xcheng/MC_Production_v3/output \
  --samples JJP_DPS1,JJP_DPS2_CS,JJP_DPS2_G,JJP_SPS_CS,JJP_SPS_G \
  --output-dir /tmp/chiw/jjp_efficiency_v1
```

Single-file smoke test:

```bash
run-multileppat-efficiency \
  --analysis-mode JpsiJpsiPhi \
  --input-files root://cceos.ihep.ac.cn///eos/ihep/cms/store/user/xcheng/MC_Production_v3/output/JJP_DPS1/0/output_ntuple.root \
  --sample-name JJP_DPS1_smoke \
  --output-dir /tmp/chiw/jjp_eff_smoke
```

Useful plot metadata options:

```bash
--cms-caption "Simulation Preliminary"
--cms-energy 13.6
--cms-era "Run 3"
--min-plot-total 1
```

`--min-plot-total` masks bins with fewer denominator events than the threshold.

## Cutflow Definition

The standard cutflow is cumulative:

```text
full_gen
fiducial_acceptance
hlt_muon_matched
single_jpsi_reco
double_jpsi_reco
single_phi_reco
triple_gen_matched_candidate
jpsi_quality
phi_quality
all6_same_recVtx
Pri_fitValid
Pri_fitPass
Pri_assocPVPass
Pri_trackPVPass
final_nominal
```

The main ratio is:

```text
efficiency(step, bin) = N(full-GEN events with >=1 GEN-matched candidate passing step in bin)
                      / N(full-GEN events in bin)
```

`fiducial_acceptance` is GEN-only and applies the current `OfflineSelectionConfig` daughter-level muon and kaon fiducial requirements.

`hlt_muon_matched` is the nominal HLT step. It uses the stored per-muon J/psi trigger/filter matching flags on GEN-matched reconstructed candidates. The event-path OR from `TrigNames` and `TrigRes` is retained as a diagnostic column in `event_step_flags.parquet`, but it is not the nominal HLT numerator.

`final_nominal` currently requires the cumulative path through:

```text
Pri_fitPass
Pri_assocPVPass
Pri_trackPVPass
```

after all earlier stages.

## Map Definition

Generated `J/psi` objects are ordered by generated pT:

```text
jpsi_lead
jpsi_sublead
```

The workflow writes object-level maps for:

```text
jpsi_lead:    pT x |y|
jpsi_sublead: pT x |y|
phi:          pT x |y|
```

For HLT and vertexing-sensitive stages, the correlated map is stored as counts in:

```text
pT(jpsi_lead) x pT(jpsi_sublead) x pT(phi)
```

and plotted as separate `pT(jpsi_lead)` vs `pT(jpsi_sublead)` heatmaps in each `pT(phi)` slice.

The workflow also writes triple-system side-check counts for:

```text
pT(J/psi + J/psi + phi)
|y(J/psi + J/psi + phi)|
M(J/psi + J/psi + phi)
```

## Uncertainties and Plots

Every efficiency bin stores:

```text
total
passed
efficiency
err_low
err_high
err_sym
```

The interval is a binomial Clopper-Pearson interval at 68.27% confidence:

```text
err_low  = efficiency - CP_low(total, passed)
err_high = CP_high(total, passed) - efficiency
err_sym  = max(err_low, err_high)
```

Plots use the existing package CMS style through `mplhep`. Each efficiency map is written as a two-panel figure:

```text
left:  efficiency
right: symmetric Clopper-Pearson uncertainty
```

When the bin grid is readable, the efficiency panel is annotated with:

```text
efficiency
+/- err_sym
passed/total
```

Low-stat bins are masked according to `--min-plot-total`.

## Outputs

Each sample directory contains:

```text
sample_manifest.json
gen_systems.parquet
event_step_flags.parquet
efficiency_counts.parquet
efficiency_maps.parquet
cutflow.csv
plots/*.png
manifest.json
```

The top-level output directory contains:

```text
run_metadata.json
subprocess_summary.parquet
subprocess_summary.csv
subprocess_envelope.parquet
manifest.json
```

`subprocess_summary` reports inclusive full-GEN and final-nominal counts per subprocess. `subprocess_envelope` is an unweighted per-bin envelope across subprocesses; it reports min, max, median, and maximum absolute deviation of the efficiency values.

## Python Interfaces

The main public helpers are:

```python
from multileppat_vertex_batch.efficiency import (
    EfficiencyBinning,
    EfficiencyRunConfig,
    build_cutflow,
    build_efficiency_counts,
    build_subprocess_envelope,
    clopper_pearson_interval,
    discover_xrootd_sample_files,
    find_jpsijpsiphi_gen_system,
    run_efficiency_for_sample,
)
```

Use `run_efficiency_for_sample(...)` for a small programmatic run over an explicit file list. Use `run-multileppat-efficiency` for standard batch output bundles.
