# `multileppat_vertex_batch` Guidelines

## Scope
These instructions apply to `src/HeavyFlavorAnalysis/TPS-Onia2MuMu/test/multileppat_vertex_batch/`. This nested repository owns the pure-Python vectorized candidate-selection, mass-fit preparation, selector comparison, and acceptance/efficiency workflows used by the Run3 `MultiLepPAT` ntuple studies.

Treat `README.md` as the package-level usage guide and `docs/integration.md` as the deeper integration note. Keep notebooks as thin frontends: they may choose inputs, windows, selectors, and plots, but should not re-implement branch lists, vectorized selection, event-best ranking, selector-comparison bookkeeping, or efficiency bookkeeping.

## Environment
Use `LCG 109a` for syntax checks, unit tests, notebooks, and batch runs so `uproot`, `awkward`, `iminuit`, `mplhep`, and plotting dependencies resolve consistently:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_109a/x86_64-el9-gcc13-opt/setup.sh
export MPLCONFIGDIR=/tmp/chiw/mplconfig_multileppat_vertex_batch
export PYTHONPYCACHEPREFIX=/tmp/chiw/pycache_multileppat_vertex_batch
python -m pip install --no-deps -e src/HeavyFlavorAnalysis/TPS-Onia2MuMu/test/multileppat_vertex_batch
```
Avoid `PYTHONPATH=src` as a bootstrap method in the current `LCG 109a` setup; it can interfere with `site-packages` imports such as `pandas`.

## Package Interfaces
Use package APIs and console entrypoints rather than notebook-local workflow forks. The main interfaces are:
- `multileppat_vertex_batch.config.OfflineSelectionConfig`
- `multileppat_vertex_batch.config.MassStudyConfig`
- `multileppat_vertex_batch.pipeline.run_massfit_prep_batch`
- `multileppat_vertex_batch.pipeline.run_roofit_selector_compare`
- `multileppat_vertex_batch.pipeline.run_iminuit_selector_compare`

The canonical mass-study batch driver is `src/multileppat_vertex_batch/cli_batch.py`, exposed as `run-multileppat-vertex-batch` after editable install. The efficiency driver is exposed as `run-multileppat-efficiency`.

Standard output writing should go through the bundle writers in `multileppat_vertex_batch.cache` so runs keep producing Parquet/JSON artifacts plus `manifest.json` files with a stable layout. Mass-selection outputs should include `fit_input_candidates.root` for downstream processing.

## Ntuple Schema
For Run3 ntuple-based studies, the standard trees are:
```text
mkcands/X_config
mkcands/X_data
```
Do not re-discover branch names in notebooks when the package already defines them. Treat `src/multileppat_vertex_batch/schema.py` as the source of truth for:
- `CORE_DATA_BRANCHES`
- `RECO_SELECTION_BRANCHES`
- `MASS_STUDY_BRANCHES`
- `TRUTH_MATCH_BRANCHES`
- `DETAIL_ENTRY_BRANCHES`
- `MASS_BRANCHES_BY_MODE`
- `FIT_BRANCHES_BY_MODE`

Data ntuples are the default supported input. GEN/truth enrichment should be enabled automatically from `mkcands/X_config` when `DoMonteCarloTree=True` and the required truth branches are present, rather than from notebook-local assumptions or a hard-coded MC-only workflow.

## Mass Study Behavior
Preserve these defaults unless the physics model is intentionally changed:
- `all6_same_recVtx` means the mode-specific muon and kaon `vertexId` values are all equal and non-negative.
- `Pri_fitValid == 1` is the selector for a valid three-body vertex fit in the active analysis mode.
- Best-candidate ranking stays the sum of squared `pt` values for the three fitted objects in the active analysis mode, with mode-aware tie-breakers from `ANALYSIS_MODE_SPECS`.
- `J/psi` signal PDFs in RooFit and `iminuit` must stay aligned as `Crystal Ball + Gaussian`.
- `Ups_mass` in `JpsiJpsiUps` and `JpsiUpsPhi` should stay modeled as Gaussian `Upsilon(1S,2S,3S)` peaks plus a polynomial background of order `1..4`.
- J/psi shape locking should go through `resolve_jpsi_pdf_config(...)` and the `JPSI_PDF_PRESETS` mechanism, not notebook-local ad hoc parameter fixing.

Fit-comparison outputs should include selector-local CMS-style projection plots via `mplhep` when a fit payload is non-empty.

## Acceptance And Efficiency
The first supported efficiency target is `JpsiJpsiPhi`. Efficiencies are ratios of GEN-matched candidates satisfying each criterion divided by the number of full GEN events. For a full-GEN denominator, the upstream ntuple production must use `RequireAcceptedCandidatesForMonteCarloTree=False`.

Support processing many independent subprocess samples. Since no reliable theoretical prediction is available for the subprocess fractions, evaluate subprocesses separately and treat their spread as a theory systematic envelope unless an explicit cross-section weighting model is provided.

Efficiency outputs should include stepwise and cut-flow tables for:
- acceptance
- HLT
- single J/psi reconstruction
- single Upsilon reconstruction where relevant
- single phi reconstruction
- vertexing
- individual offline cuts
- cumulative cut flows

Efficiency tables should carry `passed`, `total`, `efficiency`, `err_low`, `err_high`, and `err_sym`. Use Clopper-Pearson intervals at the 68.27% confidence level for binomial uncertainties.

For correlated efficiency stages, especially HLT and vertexing, prefer three-dimensional maps in `pt(Jpsi_lead)`, `pt(Jpsi_sublead)`, and `pt(phi)`. Visualize them as `pt(Jpsi_lead)` versus `pt(Jpsi_sublead)` efficiency maps split into phi-pt bins. Order generated J/psi objects by `pt` for these maps, and include uncertainty information in the plots.

Efficiency and mass-study plots should follow CMS plotting style. Use `mplhep` where practical, include CMS labels, and avoid custom plotting styles that conflict with CMS conventions.

## Inputs
Prefer explicit input file lists. Quoted wildcard tokens are only a convenience layer over the command-line interface. For XRootD sample discovery, use the explicit redirector syntax:
```bash
xrdfs root://cceos.ihep.ac.cn// ls -d /eos/ihep/cms/store/user/xcheng/MC_Production_v3/output/
```

## Planning Style
When asked for an implementation plan, make it decision-complete and include code snippets for crucial functionality. Plans should identify the intended files, interfaces, data flow, tests, and assumptions clearly enough that another engineer or agent can implement without choosing architecture details.

## Testing
Run focused Python checks after code changes under `LCG 109a`. Keep temporary outputs, Python bytecode, and plotting caches under `/tmp/chiw` so tests do not write into EOS/AFS package directories unnecessarily:
```bash
cd /eos/home-c/chiw/JpsiJpsiPhi/CMSSW_15_0_15_JpsiJpsiPhi_refactor/src/HeavyFlavorAnalysis/TPS-Onia2MuMu/test/multileppat_vertex_batch
source /cvmfs/sft.cern.ch/lcg/views/LCG_109a/x86_64-el9-gcc13-opt/setup.sh
export MPLCONFIGDIR=/tmp/chiw/mplconfig_multileppat_vertex_batch
export PYTHONPYCACHEPREFIX=/tmp/chiw/pycache_multileppat_vertex_batch
python -m pip install --no-deps -e .
```

At minimum, syntax-check touched modules and run the relevant tests:
```bash
python -m py_compile src/multileppat_vertex_batch/<touched_module>.py
python tests/<test_file>.py
```

Prefer the direct test-file form for this nested package. The `tests/` directory is not a Python package, so invocations like `python -m unittest tests.test_multileppat_vertex_batch_io.CondorHelpersTest` can fail with `ModuleNotFoundError` even when the tests themselves are valid. To run one class or method, pass the unittest selector to the file directly:
```bash
python tests/test_multileppat_vertex_batch_io.py CondorHelpersTest
python tests/test_multileppat_vertex_batch_io.py CondorHelpersTest.test_condor_dag_records_one_file_jobs_and_merge_dependency
```

For Condor workflow changes, inspect generated submit/DAG text in addition to unit tests. The submit file should keep per-job stdout/stderr but use one HTCondor event log per cluster:
```text
output = logs/$(job).out
error = logs/$(job).err
log = logs/$(Cluster).log
```

Smoke-test CLI changes with a tiny input, manifest, or fixture when available. Do not use `PYTHONPATH=src` to make tests import; use the editable install above instead, because `PYTHONPATH=src` can interfere with `LCG 109a` `site-packages` imports such as `pandas`.
