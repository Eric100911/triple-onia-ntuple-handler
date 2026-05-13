from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import stage_cache_matches, write_stage_cache_metadata
from .config import MassStudyConfig, OfflineSelectionConfig, StudyConfig
from .fit_iminuit import run_iminuit_3d_jpsijpsiphi
from .fit_roofit import build_fit_frame, build_fit_mask, run_roofit_3d_jpsijpsiphi
from .io import (
    read_config_snapshot,
    read_ntuple_arrays,
    read_parquet,
    resolve_study_input_files,
    snapshot_input_files,
    write_parquet,
)
from .progress import wrap_iterable
from .schema import CACHE_FILENAMES, CONFIG_BRANCHES, CORE_DATA_BRANCHES, FIT_SIGNIFICANCE_VERSION, TRUTH_CACHE_VERSION
from .selection import run_mass_selection_batch as _run_mass_selection_batch
from .truth import add_classifier_columns, build_file_records


def process_single_file(path: Path, config: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config_row = read_config_snapshot(path, config)
    arrays = read_ntuple_arrays(path, config, CORE_DATA_BRANCHES)
    candidate_rows, event_rows = build_file_records(
        arrays,
        source_file=str(path),
        analysis_mode=str(config_row["AnalysisMode"]),
        show_event_progress=config.show_event_progress,
        progress_backend=config.progress_backend,
    )
    candidate_df = add_classifier_columns(pd.DataFrame(candidate_rows))
    event_df = pd.DataFrame(event_rows)
    return candidate_df, event_df, config_row


def run_batch(config: StudyConfig) -> dict[str, pd.DataFrame]:
    files = resolve_study_input_files(config)

    candidate_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    config_rows: list[dict[str, Any]] = []

    iterator = wrap_iterable(
        files,
        enabled=config.show_file_progress,
        progress_backend=config.progress_backend,
        desc="Ntuple files",
    )
    for path in iterator:
        candidate_df, event_df, config_row = process_single_file(path, config)
        if not candidate_df.empty:
            candidate_parts.append(candidate_df)
        event_parts.append(event_df)
        config_rows.append(config_row)

    all_candidate_df = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    all_event_df = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    all_config_df = pd.DataFrame(config_rows)
    return {
        "candidate_df": all_candidate_df,
        "event_df": all_event_df,
        "config_df": all_config_df,
    }


def run_truth_batch(config: StudyConfig) -> dict[str, pd.DataFrame]:
    return run_batch(config)


def _cache_paths(config: StudyConfig) -> dict[str, Path]:
    return {key: config.cache_dir / filename for key, filename in CACHE_FILENAMES.items()}


def _truth_cache_payload(config: StudyConfig) -> dict[str, Any]:
    files = resolve_study_input_files(config)
    return {
        "input_files": snapshot_input_files(files),
        "tree_path": config.tree_path,
        "config_tree_path": config.config_tree_path,
        "core_data_branches": list(CORE_DATA_BRANCHES),
        "config_branches": list(CONFIG_BRANCHES),
    }


def load_or_build_cache(config: StudyConfig) -> dict[str, pd.DataFrame]:
    cache_paths = _cache_paths(config)
    cache_ready = all(path.exists() for path in cache_paths.values())
    cache_payload = _truth_cache_payload(config)
    if (
        config.use_cache
        and cache_ready
        and not config.overwrite_cache
        and stage_cache_matches(config.cache_dir, "truth_cache", TRUTH_CACHE_VERSION, cache_payload)
    ):
        print(f"Loading existing parquet cache from {config.cache_dir}")
        return {key: read_parquet(path) for key, path in cache_paths.items()}

    print(f"Building parquet cache from ROOT files into {config.cache_dir}")
    tables = run_batch(config)
    for key, path in cache_paths.items():
        write_parquet(tables[key], path)
    write_stage_cache_metadata(config.cache_dir, "truth_cache", TRUTH_CACHE_VERSION, cache_payload)
    return tables


def validate_config_consistency(config_df: pd.DataFrame) -> pd.DataFrame:
    if config_df.empty:
        return pd.DataFrame(columns=["field", "n_unique", "consistent", "example_values"])

    rows: list[dict[str, Any]] = []
    for field in CONFIG_BRANCHES:
        values = config_df[field].dropna().tolist() if field in config_df.columns else []
        unique_values: list[Any] = []
        for value in values:
            if value not in unique_values:
                unique_values.append(value)
        rows.append(
            {
                "field": field,
                "n_unique": len(unique_values),
                "consistent": int(len(unique_values) <= 1),
                "example_values": unique_values[:5],
            }
        )
    return pd.DataFrame(rows).sort_values(["consistent", "field"], ascending=[True, True]).reset_index(drop=True)


def run_massfit_prep_batch(
    files: list[str | Path],
    config: StudyConfig,
    selection_cfg: OfflineSelectionConfig,
    study_cfg: MassStudyConfig,
    show_progress: bool | None = None,
) -> dict[str, pd.DataFrame]:
    tables = _run_mass_selection_batch(
        files=files,
        config=config,
        active_windows=study_cfg.active_windows,
        selection_cfg=selection_cfg,
        analysis_mode=study_cfg.analysis_mode,
        selectors=study_cfg.selectors,
        show_progress=show_progress,
    )

    selected_for_selector_df = tables["selected_candidate_df"].loc[
        tables["selected_candidate_df"]["selector"] == study_cfg.selector_name
    ].reset_index(drop=True)
    fit_df = build_fit_frame(selected_for_selector_df, list(study_cfg.fit_branches), study_cfg.active_windows)
    fit_input_dfs_by_selector: dict[str, pd.DataFrame] = {}
    for selector in study_cfg.selectors:
        selector_selected_df = tables["selected_candidate_df"].loc[
            tables["selected_candidate_df"]["selector"] == selector
        ].reset_index(drop=True)
        if selector_selected_df.empty:
            fit_input_dfs_by_selector[selector] = selector_selected_df.copy()
            continue
        fit_input_mask = build_fit_mask(selector_selected_df, list(study_cfg.fit_branches), study_cfg.active_windows)
        fit_input_dfs_by_selector[selector] = selector_selected_df.loc[fit_input_mask].reset_index(drop=True)

    return {
        **tables,
        "selected_for_selector_df": selected_for_selector_df,
        "fit_df": fit_df,
        "fit_input_dfs_by_selector": fit_input_dfs_by_selector,
    }


def _summary_value(fit_summary: pd.DataFrame, quantity: str) -> Any:
    if fit_summary.empty:
        return np.nan
    match = fit_summary.loc[fit_summary["quantity"] == quantity, "value"]
    return match.iloc[0] if not match.empty else np.nan


def _yield_value(yield_table: pd.DataFrame, yield_name: str, field: str) -> Any:
    if yield_table.empty:
        return np.nan
    match = yield_table.loc[yield_table["yield"] == yield_name, field]
    return match.iloc[0] if not match.empty else np.nan


def _ups_peak_value(ups_peak_significance_table: pd.DataFrame, component: str, field: str) -> Any:
    if ups_peak_significance_table.empty:
        return np.nan
    match = ups_peak_significance_table.loc[ups_peak_significance_table["component"] == component, field]
    return match.iloc[0] if not match.empty else np.nan


def run_roofit_selector_compare(
    root_module,
    selected_candidate_df: pd.DataFrame,
    selection_summary_df: pd.DataFrame,
    study_cfg: MassStudyConfig,
    phi_background_kind: str = "polynomial",
    ups_background_order: int = 4,
    jpsi1_pdf_config: Any = None,
    jpsi2_pdf_config: Any = None,
    jpsi_pdf_preset: str | None = None,
) -> dict[str, Any]:
    compare_rows: list[dict[str, Any]] = []
    fit_payloads: dict[str, dict[str, Any]] = {}

    for selector in study_cfg.selectors:
        selector_summary = selection_summary_df.loc[selection_summary_df["selector"] == selector]
        selector_selected = selected_candidate_df.loc[selected_candidate_df["selector"] == selector].reset_index(drop=True)
        fit_df = build_fit_frame(selector_selected, list(study_cfg.fit_branches), study_cfg.active_windows)

        row: dict[str, Any] = {"selector": selector, "jpsi_pdf_preset": jpsi_pdf_preset or "(custom)"}
        if not selector_summary.empty:
            row.update(selector_summary.iloc[0].to_dict())
        row["n_fit_events"] = int(len(fit_df))

        if fit_df.empty:
            row.update(
                {
                    "fit_status": np.nan,
                    "covQual": np.nan,
                    "N_sss": np.nan,
                    "N_sss_err": np.nan,
                    "N_sss_q0": np.nan,
                    "N_sss_p0": np.nan,
                    "N_sss_significance_sigma": np.nan,
                    "ups_2s_yield": np.nan,
                    "ups_2s_q0": np.nan,
                    "ups_2s_p0": np.nan,
                    "ups_2s_significance_sigma": np.nan,
                    "ups_3s_yield": np.nan,
                    "ups_3s_q0": np.nan,
                    "ups_3s_p0": np.nan,
                    "ups_3s_significance_sigma": np.nan,
                    "ups_excited_yield": np.nan,
                    "ups_excited_q0": np.nan,
                    "ups_excited_p0": np.nan,
                    "ups_excited_significance_sigma": np.nan,
                }
            )
            compare_rows.append(row)
            continue

        fit_payload = run_roofit_3d_jpsijpsiphi(
            root_module,
            fit_df,
            study_cfg.active_windows,
            analysis_mode=study_cfg.analysis_mode,
            phi_background_kind=phi_background_kind,
            ups_background_order=ups_background_order,
            jpsi1_pdf_config=jpsi1_pdf_config,
            jpsi2_pdf_config=jpsi2_pdf_config,
        )
        fit_payload.update(
            {
                "selector": selector,
                "fit_df": fit_df.copy(),
                "active_windows": dict(study_cfg.active_windows),
                "fit_branches": tuple(study_cfg.fit_branches),
                "analysis_mode": study_cfg.analysis_mode,
                "ups_background_order": int(ups_background_order),
            }
        )
        fit_payloads[selector] = fit_payload
        fit_result = fit_payload["fit_result"]
        fit_summary = fit_payload["fit_summary"]
        yield_table = fit_payload["yield_table"]
        ups_peak_significance_table = fit_payload.get("ups_peak_significance_table", pd.DataFrame())
        row.update(
            {
                "fit_status": fit_result.status(),
                "covQual": fit_result.covQual(),
                "N_sss": _yield_value(yield_table, "N_sss", "value"),
                "N_sss_err": _yield_value(yield_table, "N_sss", "error"),
                "N_sss_q0": _summary_value(fit_summary, "q0"),
                "N_sss_p0": _summary_value(fit_summary, "p0"),
                "N_sss_significance_sigma": _summary_value(fit_summary, "N_sss_significance_sigma"),
                "ups_2s_yield": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "signal_yield"),
                "ups_2s_q0": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "q0"),
                "ups_2s_p0": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "p0"),
                "ups_2s_significance_sigma": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "significance_sigma"),
                "ups_3s_yield": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "signal_yield"),
                "ups_3s_q0": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "q0"),
                "ups_3s_p0": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "p0"),
                "ups_3s_significance_sigma": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "significance_sigma"),
                "ups_excited_yield": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "signal_yield"),
                "ups_excited_q0": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "q0"),
                "ups_excited_p0": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "p0"),
                "ups_excited_significance_sigma": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "significance_sigma"),
                "jpsi1_locked_params": _summary_value(fit_summary, "jpsi1_locked_params"),
                "jpsi2_locked_params": _summary_value(fit_summary, "jpsi2_locked_params"),
                "fit_significance_version": FIT_SIGNIFICANCE_VERSION,
            }
        )
        compare_rows.append(row)

    return {"selector_compare_df": pd.DataFrame(compare_rows), "fit_payloads": fit_payloads}


def run_iminuit_selector_compare(
    minuit_cls,
    selected_candidate_df: pd.DataFrame,
    selection_summary_df: pd.DataFrame,
    study_cfg: MassStudyConfig,
    ups_background_order: int = 4,
    jpsi1_pdf_config: Any = None,
    jpsi2_pdf_config: Any = None,
    jpsi_pdf_preset: str | None = None,
) -> dict[str, Any]:
    compare_rows: list[dict[str, Any]] = []
    fit_payloads: dict[str, dict[str, Any]] = {}

    for selector in study_cfg.selectors:
        selector_summary = selection_summary_df.loc[selection_summary_df["selector"] == selector]
        selector_selected = selected_candidate_df.loc[selected_candidate_df["selector"] == selector].reset_index(drop=True)
        fit_df = build_fit_frame(selector_selected, list(study_cfg.fit_branches), study_cfg.active_windows)

        row: dict[str, Any] = {"selector": selector, "jpsi_pdf_preset": jpsi_pdf_preset or "(custom)"}
        if not selector_summary.empty:
            row.update(selector_summary.iloc[0].to_dict())
        row["n_fit_events"] = int(len(fit_df))

        if fit_df.empty:
            row.update(
                {
                    "fit_status": np.nan,
                    "covQual": np.nan,
                    "N_sss": np.nan,
                    "N_sss_err": np.nan,
                    "N_sss_q0": np.nan,
                    "N_sss_p0": np.nan,
                    "N_sss_significance_sigma": np.nan,
                    "ups_2s_yield": np.nan,
                    "ups_2s_q0": np.nan,
                    "ups_2s_p0": np.nan,
                    "ups_2s_significance_sigma": np.nan,
                    "ups_3s_yield": np.nan,
                    "ups_3s_q0": np.nan,
                    "ups_3s_p0": np.nan,
                    "ups_3s_significance_sigma": np.nan,
                    "ups_excited_yield": np.nan,
                    "ups_excited_q0": np.nan,
                    "ups_excited_p0": np.nan,
                    "ups_excited_significance_sigma": np.nan,
                }
            )
            compare_rows.append(row)
            continue

        fit_payload = run_iminuit_3d_jpsijpsiphi(
            minuit_cls,
            fit_df,
            study_cfg.active_windows,
            analysis_mode=study_cfg.analysis_mode,
            ups_background_order=ups_background_order,
            jpsi1_pdf_config=jpsi1_pdf_config,
            jpsi2_pdf_config=jpsi2_pdf_config,
        )
        fit_payload.update(
            {
                "selector": selector,
                "fit_df": fit_df.copy(),
                "active_windows": dict(study_cfg.active_windows),
                "fit_branches": tuple(study_cfg.fit_branches),
                "analysis_mode": study_cfg.analysis_mode,
                "ups_background_order": int(ups_background_order),
            }
        )
        fit_payloads[selector] = fit_payload
        fit_summary = fit_payload["fit_summary"]
        yield_table = fit_payload["yield_table"]
        minuit_obj = fit_payload["minuit"]
        ups_peak_significance_table = fit_payload.get("ups_peak_significance_table", pd.DataFrame())
        row.update(
            {
                "fit_status": int(bool(minuit_obj.fmin.is_valid)),
                "covQual": np.nan,
                "N_sss": _yield_value(yield_table, "N_sss", "value"),
                "N_sss_err": _yield_value(yield_table, "N_sss", "error"),
                "N_sss_q0": _summary_value(fit_summary, "q0"),
                "N_sss_p0": _summary_value(fit_summary, "p0"),
                "N_sss_significance_sigma": _summary_value(fit_summary, "N_sss_significance_sigma"),
                "ups_2s_yield": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "signal_yield"),
                "ups_2s_q0": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "q0"),
                "ups_2s_p0": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "p0"),
                "ups_2s_significance_sigma": _ups_peak_value(ups_peak_significance_table, "Ups_2S", "significance_sigma"),
                "ups_3s_yield": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "signal_yield"),
                "ups_3s_q0": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "q0"),
                "ups_3s_p0": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "p0"),
                "ups_3s_significance_sigma": _ups_peak_value(ups_peak_significance_table, "Ups_3S", "significance_sigma"),
                "ups_excited_yield": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "signal_yield"),
                "ups_excited_q0": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "q0"),
                "ups_excited_p0": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "p0"),
                "ups_excited_significance_sigma": _ups_peak_value(ups_peak_significance_table, "Ups_excited", "significance_sigma"),
                "jpsi1_locked_params": _summary_value(fit_summary, "jpsi1_locked_params"),
                "jpsi2_locked_params": _summary_value(fit_summary, "jpsi2_locked_params"),
                "fit_significance_version": FIT_SIGNIFICANCE_VERSION,
            }
        )
        compare_rows.append(row)

    return {"selector_compare_df": pd.DataFrame(compare_rows), "fit_payloads": fit_payloads}
