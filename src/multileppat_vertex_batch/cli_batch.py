#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from multileppat_vertex_batch.cache import (
    load_fit_compare_bundle_if_compatible,
    load_mass_selection_bundle_if_compatible,
    stage_cache_key,
    write_fit_compare_bundle,
    write_mass_selection_bundle,
    write_run_metadata,
    write_truth_cache_bundle,
)
from multileppat_vertex_batch.config import (
    CmsPlotStyleConfig,
    MassStudyConfig,
    OfflineSelectionConfig,
    StudyConfig,
    default_mass_windows_from_config_row,
    resolve_windows,
)
from multileppat_vertex_batch.fit_roofit import JPSI_PDF_PRESETS, resolve_jpsi_pdf_config
from multileppat_vertex_batch.io import (
    ensure_dir,
    find_missing_tree_branches,
    read_config_snapshot,
    resolve_input_files,
    snapshot_input_files,
)
from multileppat_vertex_batch.pipeline import (
    load_or_build_cache,
    run_iminuit_selector_compare,
    run_massfit_prep_batch,
    run_roofit_selector_compare,
    validate_config_consistency,
)
from multileppat_vertex_batch.schema import (
    FIT_BRANCHES_BY_MODE,
    FIT_COMPARE_CACHE_VERSIONS,
    FIT_SIGNIFICANCE_VERSION,
    MASS_SELECTION_CACHE_VERSION,
    MASS_STUDY_BRANCHES,
    PHI_BACKGROUND_MODEL_VERSION,
    TRUTH_MATCH_BRANCHES,
    UPS_SIGNAL_MODEL_VERSION,
    get_analysis_mode_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic batch driver for MultiLepPAT offline-selection and optional fit studies on data or MC ntuples. "
            "The requested --analysis-mode must match mkcands/X_config/AnalysisMode."
        )
    )
    parser.add_argument("input_files", nargs="+", help="Input ROOT files. Wildcard tokens are accepted if quoted.")
    parser.add_argument("--analysis-mode", required=True, choices=("JpsiJpsiPhi", "JpsiJpsiUps", "JpsiUpsPhi"))
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--selectors", default="all6_same_recVtx,Pri_fitValid")
    parser.add_argument("--selector-name", default="all6_same_recVtx")
    parser.add_argument("--fit-backend", choices=("none", "roofit", "iminuit", "both"), default="none")
    parser.add_argument("--phi-background-kind", choices=("polynomial", "chebychev"), default="polynomial")
    parser.add_argument("--ups-background-order", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--jpsi-pdf-preset", choices=tuple(sorted(JPSI_PDF_PRESETS)), default="small_sample")
    parser.add_argument("--skip-truth", action="store_true")
    parser.add_argument("--skip-selection", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    parser.add_argument("--cms-caption", default="Work In Progress")
    parser.add_argument("--cms-energy", type=float, default=13.6)
    parser.add_argument("--cms-lumi", type=float, default=283.4)
    parser.add_argument("--cms-era", default="Run 3 (2022-2025)")
    return parser.parse_args()


def parse_selectors(raw: str) -> tuple[str, ...]:
    selectors = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not selectors:
        raise ValueError("At least one selector must be provided.")
    return selectors


def active_windows_from_offline_defaults(
    analysis_mode: str,
    config_row: dict[str, Any],
    offline_cfg: OfflineSelectionConfig,
) -> dict[str, tuple[float, float]]:
    default_windows = default_mass_windows_from_config_row(config_row)
    overrides: dict[str, tuple[float, float] | None] = {"Pri_mass": None}
    if analysis_mode in {"JpsiJpsiPhi", "JpsiJpsiUps", "JpsiUpsPhi"}:
        overrides["Jpsi_1_mass"] = offline_cfg.jpsi_mass_window
    if analysis_mode == "JpsiJpsiPhi":
        overrides["Jpsi_2_mass"] = offline_cfg.jpsi_mass_window
        overrides["Phi_mass"] = offline_cfg.phi_mass_window
    elif analysis_mode == "JpsiJpsiUps":
        overrides["Jpsi_2_mass"] = offline_cfg.jpsi_mass_window
        overrides["Ups_mass"] = offline_cfg.ups_mass_window
    elif analysis_mode == "JpsiUpsPhi":
        overrides["Ups_mass"] = offline_cfg.ups_mass_window
        overrides["Phi_mass"] = offline_cfg.phi_mass_window
    return resolve_windows(default_windows, overrides)


def _config_bool(value: Any, field_name: str) -> bool:
    if value is None:
        raise RuntimeError(f"X_config field '{field_name}' is missing; cannot determine input-mode behavior.")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def inspect_inputs(files: list[Path], probe_config: StudyConfig) -> dict[str, Any]:
    config_rows = [read_config_snapshot(path, probe_config) for path in files]
    analysis_modes = {str(row.get("AnalysisMode") or "") for row in config_rows}
    if len(analysis_modes) != 1:
        raise RuntimeError(f"Inconsistent AnalysisMode across input files: {sorted(analysis_modes)}")
    analysis_mode = next(iter(analysis_modes))
    if not analysis_mode:
        raise RuntimeError("AnalysisMode is missing from X_config; cannot continue.")

    mc_flags = {_config_bool(row.get("DoMonteCarloTree"), "DoMonteCarloTree") for row in config_rows}
    if len(mc_flags) != 1:
        raise RuntimeError("Mixed data and MC ntuples in one run are not supported.")
    is_mc = next(iter(mc_flags))
    missing_truth_branches = find_missing_tree_branches(files, probe_config.tree_path, TRUTH_MATCH_BRANCHES) if is_mc else {}
    return {
        "analysis_mode": analysis_mode,
        "is_mc": is_mc,
        "config_rows": config_rows,
        "missing_truth_branches": missing_truth_branches,
    }


def format_missing_truth_branch_error(missing_truth_branches: dict[str, list[str]]) -> str:
    lines = ["MC truth was requested automatically from X_config, but required truth branches are missing:"]
    for path, branches in sorted(missing_truth_branches.items()):
        preview = ", ".join(branches[:8])
        suffix = "" if len(branches) <= 8 else f", ... ({len(branches)} total)"
        lines.append(f"- {path}: {preview}{suffix}")
    return "\n".join(lines)


def run_truth_stage(files: list[Path], config: StudyConfig, output_dir: Path) -> dict[str, pd.DataFrame]:
    tables = load_or_build_cache(config)
    consistency_df = validate_config_consistency(tables["config_df"])
    write_truth_cache_bundle(output_dir, files, tables, consistency_df)
    return tables


def build_mass_selection_cache_payload(
    files: list[Path],
    config: StudyConfig,
    study_cfg: MassStudyConfig,
    offline_cfg: OfflineSelectionConfig,
) -> dict[str, Any]:
    return {
        "analysis_mode": study_cfg.analysis_mode,
        "input_files": snapshot_input_files(files),
        "tree_path": config.tree_path,
        "config_tree_path": config.config_tree_path,
        "mass_selection_branches": list(MASS_STUDY_BRANCHES),
        "offline_selection": offline_cfg.__dict__,
        "active_windows": study_cfg.active_windows,
        "selectors": list(study_cfg.selectors),
        "selector_name": study_cfg.selector_name,
        "fit_branches": list(study_cfg.fit_branches),
        "best_candidate_metric": study_cfg.best_candidate_metric,
    }


def build_fit_compare_cache_payload(
    *,
    analysis_mode: str,
    backend: str,
    mass_selection_cache_key: str,
    jpsi_pdf_preset: str,
    plot_style_cfg: CmsPlotStyleConfig,
    phi_background_kind: str | None = None,
    ups_background_order: int = 4,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "analysis_mode": analysis_mode,
        "backend": backend,
        "mass_selection_cache_key": mass_selection_cache_key,
        "jpsi_pdf_preset": jpsi_pdf_preset,
        "ups_background_order": int(ups_background_order),
        "ups_signal_model_version": UPS_SIGNAL_MODEL_VERSION,
        "phi_background_model_version": PHI_BACKGROUND_MODEL_VERSION,
        "cms_plot_style": plot_style_cfg.__dict__,
        "fit_significance_version": FIT_SIGNIFICANCE_VERSION,
    }
    if phi_background_kind is not None:
        payload["phi_background_kind"] = phi_background_kind
    return payload


def run_selection_stage(
    files: list[Path],
    config: StudyConfig,
    offline_cfg: OfflineSelectionConfig,
    study_cfg: MassStudyConfig,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    tables = run_massfit_prep_batch(
        files=files,
        config=config,
        selection_cfg=offline_cfg,
        study_cfg=study_cfg,
        show_progress=config.show_file_progress,
    )
    write_mass_selection_bundle(
        output_dir,
        tables,
        cache_version=MASS_SELECTION_CACHE_VERSION,
        cache_payload=build_mass_selection_cache_payload(files, config, study_cfg, offline_cfg),
    )
    return tables


def run_roofit_stage(
    *,
    selected_candidate_df: pd.DataFrame,
    selection_summary_df: pd.DataFrame,
    study_cfg: MassStudyConfig,
    jpsi_pdf_preset: str,
    phi_background_kind: str,
    ups_background_order: int,
    output_dir: Path,
    plot_style_cfg: CmsPlotStyleConfig,
    mass_selection_cache_key: str,
    write_plots: bool = True,
) -> dict[str, Any]:
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    jpsi1_cfg = resolve_jpsi_pdf_config(preset=jpsi_pdf_preset)
    jpsi2_cfg = resolve_jpsi_pdf_config(preset=jpsi_pdf_preset)
    payload = run_roofit_selector_compare(
        root_module=ROOT,
        selected_candidate_df=selected_candidate_df,
        selection_summary_df=selection_summary_df,
        study_cfg=study_cfg,
        phi_background_kind=phi_background_kind,
        ups_background_order=ups_background_order,
        jpsi1_pdf_config=jpsi1_cfg,
        jpsi2_pdf_config=jpsi2_cfg,
        jpsi_pdf_preset=jpsi_pdf_preset,
    )
    write_fit_compare_bundle(
        output_dir,
        payload,
        backend="roofit",
        extra_metadata={
            "analysis_mode": study_cfg.analysis_mode,
            "jpsi_pdf_preset": jpsi_pdf_preset,
            "phi_background_kind": phi_background_kind,
            "ups_background_order": ups_background_order,
        },
        plot_style_cfg=plot_style_cfg if write_plots else None,
        cache_version=FIT_COMPARE_CACHE_VERSIONS["roofit"],
        cache_payload=build_fit_compare_cache_payload(
            analysis_mode=study_cfg.analysis_mode,
            backend="roofit",
            mass_selection_cache_key=mass_selection_cache_key,
            jpsi_pdf_preset=jpsi_pdf_preset,
            plot_style_cfg=plot_style_cfg,
            phi_background_kind=phi_background_kind,
            ups_background_order=ups_background_order,
        ),
    )
    return payload


def run_iminuit_stage(
    *,
    selected_candidate_df: pd.DataFrame,
    selection_summary_df: pd.DataFrame,
    study_cfg: MassStudyConfig,
    jpsi_pdf_preset: str,
    ups_background_order: int,
    output_dir: Path,
    plot_style_cfg: CmsPlotStyleConfig,
    mass_selection_cache_key: str,
    write_plots: bool = True,
) -> dict[str, Any]:
    from iminuit import Minuit

    jpsi1_cfg = resolve_jpsi_pdf_config(preset=jpsi_pdf_preset)
    jpsi2_cfg = resolve_jpsi_pdf_config(preset=jpsi_pdf_preset)
    payload = run_iminuit_selector_compare(
        minuit_cls=Minuit,
        selected_candidate_df=selected_candidate_df,
        selection_summary_df=selection_summary_df,
        study_cfg=study_cfg,
        ups_background_order=ups_background_order,
        jpsi1_pdf_config=jpsi1_cfg,
        jpsi2_pdf_config=jpsi2_cfg,
        jpsi_pdf_preset=jpsi_pdf_preset,
    )
    write_fit_compare_bundle(
        output_dir,
        payload,
        backend="iminuit",
        extra_metadata={
            "analysis_mode": study_cfg.analysis_mode,
            "jpsi_pdf_preset": jpsi_pdf_preset,
            "ups_background_order": ups_background_order,
        },
        plot_style_cfg=plot_style_cfg if write_plots else None,
        cache_version=FIT_COMPARE_CACHE_VERSIONS["iminuit"],
        cache_payload=build_fit_compare_cache_payload(
            analysis_mode=study_cfg.analysis_mode,
            backend="iminuit",
            mass_selection_cache_key=mass_selection_cache_key,
            jpsi_pdf_preset=jpsi_pdf_preset,
            plot_style_cfg=plot_style_cfg,
            ups_background_order=ups_background_order,
        ),
    )
    return payload


def main() -> None:
    args = parse_args()
    mode_spec = get_analysis_mode_spec(args.analysis_mode)
    selectors = parse_selectors(args.selectors)
    files = resolve_input_files(args.input_files)
    output_dir = ensure_dir(Path(args.output_dir))

    probe_config = StudyConfig(
        input_files=tuple(str(path) for path in files),
        show_file_progress=not args.hide_progress,
        show_event_progress=False,
        progress_backend="terminal",
    )
    inspection = inspect_inputs(files, probe_config)
    if inspection["analysis_mode"] != args.analysis_mode:
        raise RuntimeError(
            f"--analysis-mode {args.analysis_mode} does not match X_config AnalysisMode {inspection['analysis_mode']}."
        )
    analysis_mode = inspection["analysis_mode"]
    is_mc = inspection["is_mc"]
    if inspection["missing_truth_branches"]:
        raise RuntimeError(format_missing_truth_branch_error(inspection["missing_truth_branches"]))

    fit_branches = tuple(FIT_BRANCHES_BY_MODE.get(analysis_mode, ()))
    if args.fit_backend != "none" and not fit_branches:
        raise RuntimeError(f"Fit backend requested, but AnalysisMode '{analysis_mode}' has no supported fit branches.")

    config_row = inspection["config_rows"][0]
    offline_cfg = OfflineSelectionConfig()
    if offline_cfg.ups_vtxprob_min is None:
        offline_cfg = replace(offline_cfg, ups_vtxprob_min=float(config_row["UpsDecayVtxProbCut"]))
    active_windows = active_windows_from_offline_defaults(analysis_mode, config_row, offline_cfg)
    study_cfg = MassStudyConfig(
        analysis_mode=analysis_mode,
        active_windows=active_windows,
        selector_name=args.selector_name,
        selectors=selectors,
        fit_branches=fit_branches,
    )
    truth_enabled = is_mc and not args.skip_truth
    plot_style_cfg = CmsPlotStyleConfig(
        caption=args.cms_caption,
        energy_tev=args.cms_energy,
        lumi_fb=args.cms_lumi,
        era=args.cms_era,
        is_data=not is_mc,
    )

    metadata = {
        "input_files": [str(path) for path in files],
        "n_input_files": len(files),
        "first_file": str(files[0]),
        "analysis_mode": analysis_mode,
        "is_mc": is_mc,
        "truth_enabled": truth_enabled,
        "selectors": selectors,
        "selector_name": args.selector_name,
        "fit_backend": args.fit_backend,
        "jpsi_pdf_preset": args.jpsi_pdf_preset,
        "phi_background_kind": args.phi_background_kind,
        "ups_background_order": args.ups_background_order,
        "offline_selection": offline_cfg.__dict__,
        "active_windows": active_windows,
        "cms_plot_style": plot_style_cfg.__dict__,
    }
    write_run_metadata(output_dir, metadata)

    print(f"Resolved input files: {len(files)}")
    print(f"AnalysisMode: {analysis_mode}")
    print(f"DoMonteCarloTree: {int(is_mc)}")
    print(f"Output directory: {output_dir}")

    truth_config = StudyConfig(
        input_files=tuple(str(path) for path in files),
        cache_dir=output_dir / "truth_cache",
        use_cache=not args.no_cache,
        overwrite_cache=args.overwrite_cache,
        show_file_progress=not args.hide_progress,
        show_event_progress=False,
        progress_backend="terminal",
    )

    if truth_enabled:
        truth_cache_dir = ensure_dir(output_dir / "truth_cache")
        print("Running truth-cache stage...")
        truth_tables = run_truth_stage(files, truth_config, truth_cache_dir)
        print(f"  candidate rows: {len(truth_tables['candidate_df'])}")
        print(f"  event rows: {len(truth_tables['event_df'])}")
        print(f"  config rows: {len(truth_tables['config_df'])}")
    elif is_mc:
        print("Skipping truth-cache stage by request.")
    else:
        print("Skipping truth-cache stage: DoMonteCarloTree=False in X_config.")

    mass_tables: dict[str, Any] | None = None
    mass_selection_cache_payload = build_mass_selection_cache_payload(files, truth_config, study_cfg, offline_cfg)
    mass_selection_cache_key = stage_cache_key("mass_selection", MASS_SELECTION_CACHE_VERSION, mass_selection_cache_payload)
    mass_output_dir = ensure_dir(output_dir / "mass_selection")
    cached_mass_tables = None if args.overwrite_cache else load_mass_selection_bundle_if_compatible(
        mass_output_dir,
        MASS_SELECTION_CACHE_VERSION,
        mass_selection_cache_payload,
    )
    if not args.skip_selection or args.fit_backend != "none":
        if cached_mass_tables is not None:
            print(f"Loading cached mass-selection bundle from {mass_output_dir}")
            mass_tables = cached_mass_tables
        elif args.skip_selection:
            raise RuntimeError(f"--skip-selection was requested, but no compatible mass-selection cache exists in {mass_output_dir}.")
        else:
            print(f"Running offline mass-selection stage for {analysis_mode}...")
            mass_tables = run_selection_stage(
                files=files,
                config=truth_config,
                offline_cfg=offline_cfg,
                study_cfg=study_cfg,
                output_dir=mass_output_dir,
            )
        print(f"  selected best candidates across selectors: {len(mass_tables['selected_candidate_df'])}")
        print(f"  selected best candidates for nominal selector: {len(mass_tables['selected_for_selector_df'])}")
        print(f"  fit-input ROOT file: {output_dir / 'mass_selection' / 'fit_input_candidates.root'}")
    else:
        print("Skipping offline mass-selection stage.")

    if args.fit_backend in ("roofit", "both"):
        if mass_tables is None:
            raise RuntimeError("RooFit stage requested without mass-selection tables.")
        roofit_output_dir = ensure_dir(output_dir / "roofit_compare")
        roofit_cache_payload = build_fit_compare_cache_payload(
            analysis_mode=analysis_mode,
            backend="roofit",
            mass_selection_cache_key=mass_selection_cache_key,
            jpsi_pdf_preset=args.jpsi_pdf_preset,
            plot_style_cfg=plot_style_cfg,
            phi_background_kind=args.phi_background_kind,
            ups_background_order=args.ups_background_order,
        )
        cached_roofit_payload = None if args.overwrite_cache else load_fit_compare_bundle_if_compatible(
            roofit_output_dir,
            backend="roofit",
            cache_version=FIT_COMPARE_CACHE_VERSIONS["roofit"],
            cache_payload=roofit_cache_payload,
        )
        if cached_roofit_payload is not None:
            roofit_payload = cached_roofit_payload
            print(f"Loading cached RooFit compare bundle from {roofit_output_dir}")
        else:
            print("Running RooFit selector compare...")
            roofit_payload = run_roofit_stage(
                selected_candidate_df=mass_tables["selected_candidate_df"],
                selection_summary_df=mass_tables["selection_summary_df"],
                study_cfg=study_cfg,
                jpsi_pdf_preset=args.jpsi_pdf_preset,
                phi_background_kind=args.phi_background_kind,
                ups_background_order=args.ups_background_order,
                output_dir=roofit_output_dir,
                plot_style_cfg=plot_style_cfg,
                mass_selection_cache_key=mass_selection_cache_key,
            )
        print(f"  RooFit selectors processed: {len(roofit_payload['selector_compare_df'])}")

    if args.fit_backend in ("iminuit", "both"):
        if mass_tables is None:
            raise RuntimeError("iminuit stage requested without mass-selection tables.")
        iminuit_output_dir = ensure_dir(output_dir / "iminuit_compare")
        iminuit_cache_payload = build_fit_compare_cache_payload(
            analysis_mode=analysis_mode,
            backend="iminuit",
            mass_selection_cache_key=mass_selection_cache_key,
            jpsi_pdf_preset=args.jpsi_pdf_preset,
            plot_style_cfg=plot_style_cfg,
            ups_background_order=args.ups_background_order,
        )
        cached_iminuit_payload = None if args.overwrite_cache else load_fit_compare_bundle_if_compatible(
            iminuit_output_dir,
            backend="iminuit",
            cache_version=FIT_COMPARE_CACHE_VERSIONS["iminuit"],
            cache_payload=iminuit_cache_payload,
        )
        if cached_iminuit_payload is not None:
            iminuit_payload = cached_iminuit_payload
            print(f"Loading cached iminuit compare bundle from {iminuit_output_dir}")
        else:
            print("Running iminuit selector compare...")
            iminuit_payload = run_iminuit_stage(
                selected_candidate_df=mass_tables["selected_candidate_df"],
                selection_summary_df=mass_tables["selection_summary_df"],
                study_cfg=study_cfg,
                jpsi_pdf_preset=args.jpsi_pdf_preset,
                ups_background_order=args.ups_background_order,
                output_dir=iminuit_output_dir,
                plot_style_cfg=plot_style_cfg,
                mass_selection_cache_key=mass_selection_cache_key,
            )
        print(f"  iminuit selectors processed: {len(iminuit_payload['selector_compare_df'])}")


if __name__ == "__main__":
    main()
