from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import CmsPlotStyleConfig
from .io import (
    dataframe_to_json_records,
    ensure_dir,
    read_json,
    read_parquet,
    stable_data_hash,
    to_jsonable,
    write_json,
    write_parquet,
    write_root_trees,
)
from .plotting import write_iminuit_projection_plots, write_roofit_projection_plots
from .schema import CACHE_FILENAMES, CACHE_METADATA_FILENAME, MASS_SELECTION_FILENAMES


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _artifact_entry(path: Path, root: Path, format_name: str, n_rows: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "format": format_name,
        "path": _rel(path, root),
    }
    if n_rows is not None:
        entry["n_rows"] = int(n_rows)
    return entry


def _write_manifest(output_dir: Path, stage: str, artifacts: dict[str, dict[str, Any]], extra: dict[str, Any] | None = None) -> Path:
    manifest = {
        "stage": stage,
        "artifacts": artifacts,
    }
    if extra:
        manifest.update({key: to_jsonable(value) for key, value in extra.items()})
    manifest_path = output_dir / "manifest.json"
    write_json(manifest, manifest_path)
    return manifest_path


def build_stage_cache_metadata(stage: str, cache_version: int, cache_payload: dict[str, Any]) -> dict[str, Any]:
    payload = to_jsonable(cache_payload)
    cache_key = stable_data_hash(
        {
            "stage": stage,
            "cache_version": int(cache_version),
            "payload": payload,
        }
    )
    return {
        "stage": stage,
        "cache_version": int(cache_version),
        "cache_key": cache_key,
        "payload": payload,
    }


def write_stage_cache_metadata(output_dir: Path, stage: str, cache_version: int, cache_payload: dict[str, Any]) -> Path:
    metadata_path = ensure_dir(output_dir) / CACHE_METADATA_FILENAME
    write_json(build_stage_cache_metadata(stage, cache_version, cache_payload), metadata_path)
    return metadata_path


def load_stage_cache_metadata(output_dir: Path) -> dict[str, Any] | None:
    metadata_path = Path(output_dir) / CACHE_METADATA_FILENAME
    if not metadata_path.exists():
        return None
    return read_json(metadata_path)


def stage_cache_matches(output_dir: Path, stage: str, cache_version: int, cache_payload: dict[str, Any]) -> bool:
    current = load_stage_cache_metadata(output_dir)
    if current is None:
        return False
    expected = build_stage_cache_metadata(stage, cache_version, cache_payload)
    return (
        current.get("stage") == expected["stage"]
        and int(current.get("cache_version", -1)) == expected["cache_version"]
        and current.get("cache_key") == expected["cache_key"]
    )


def stage_cache_key(stage: str, cache_version: int, cache_payload: dict[str, Any]) -> str:
    return build_stage_cache_metadata(stage, cache_version, cache_payload)["cache_key"]


def write_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / "run_metadata.json"
    write_json(metadata, path)
    return path


def write_truth_cache_bundle(
    output_dir: Path,
    files: list[str | Path],
    tables: dict[str, pd.DataFrame],
    consistency_df: pd.DataFrame,
    cache_version: int | None = None,
    cache_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    written: dict[str, Path] = {}

    for key in ("candidate_df", "event_df", "config_df"):
        frame = tables[key]
        path = output_dir / CACHE_FILENAMES[key]
        write_parquet(frame, path)
        artifacts[key] = _artifact_entry(path, output_dir, "parquet", len(frame))
        written[key] = path

    truth_summary = {
        "n_input_files": len(files),
        "candidate_rows": int(len(tables["candidate_df"])),
        "event_rows": int(len(tables["event_df"])),
        "config_rows": int(len(tables["config_df"])),
        "truth_triple_strict_candidates": int(tables["candidate_df"]["truth_triple_strict"].sum())
        if not tables["candidate_df"].empty and "truth_triple_strict" in tables["candidate_df"].columns
        else 0,
    }
    truth_summary_path = output_dir / "truth_summary.json"
    write_json(truth_summary, truth_summary_path)
    artifacts["truth_summary"] = _artifact_entry(truth_summary_path, output_dir, "json")
    written["truth_summary"] = truth_summary_path

    consistency_path = output_dir / "config_consistency.json"
    write_json(dataframe_to_json_records(consistency_df), consistency_path)
    artifacts["config_consistency"] = _artifact_entry(consistency_path, output_dir, "json", len(consistency_df))
    written["config_consistency"] = consistency_path

    if cache_version is not None and cache_payload is not None:
        cache_meta_path = write_stage_cache_metadata(output_dir, "truth_cache", cache_version, cache_payload)
        artifacts["cache_meta"] = _artifact_entry(cache_meta_path, output_dir, "json")
        written["cache_meta"] = cache_meta_path
    manifest_path = _write_manifest(
        output_dir,
        stage="truth_cache",
        artifacts=artifacts,
        extra={"n_input_files": len(files)},
    )
    written["manifest"] = manifest_path
    return written


def write_mass_selection_bundle(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    cache_version: int | None = None,
    cache_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    written: dict[str, Path] = {}

    for key, frame in tables.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        path = output_dir / f"{key}.parquet"
        write_parquet(frame, path)
        artifacts[key] = _artifact_entry(path, output_dir, "parquet", len(frame))
        written[key] = path

    fit_input_dfs_by_selector = tables.get("fit_input_dfs_by_selector")
    if isinstance(fit_input_dfs_by_selector, dict):
        fit_root_path = output_dir / "fit_input_candidates.root"
        tree_frames = {
            f"fit_input_{selector}": frame
            for selector, frame in fit_input_dfs_by_selector.items()
            if isinstance(frame, pd.DataFrame)
        }
        write_root_trees(fit_root_path, tree_frames)
        artifacts["fit_input_root"] = _artifact_entry(fit_root_path, output_dir, "root")
        written["fit_input_root"] = fit_root_path

    if cache_version is not None and cache_payload is not None:
        cache_meta_path = write_stage_cache_metadata(output_dir, "mass_selection", cache_version, cache_payload)
        artifacts["cache_meta"] = _artifact_entry(cache_meta_path, output_dir, "json")
        written["cache_meta"] = cache_meta_path
    manifest_path = _write_manifest(output_dir, stage="mass_selection", artifacts=artifacts)
    written["manifest"] = manifest_path
    return written


def load_mass_selection_bundle_if_compatible(
    output_dir: Path,
    cache_version: int,
    cache_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if not stage_cache_matches(output_dir, "mass_selection", cache_version, cache_payload):
        return None
    required_paths = {key: Path(output_dir) / filename for key, filename in MASS_SELECTION_FILENAMES.items()}
    if not all(path.exists() for path in required_paths.values()):
        return None
    tables = {key: read_parquet(path) for key, path in required_paths.items()}
    tables["fit_input_dfs_by_selector"] = {}
    return tables


def write_fit_compare_bundle(
    output_dir: Path,
    payload: dict[str, Any],
    backend: str,
    extra_metadata: dict[str, Any] | None = None,
    plot_style_cfg: CmsPlotStyleConfig | None = None,
    plot_specs: list[dict[str, Any]] | None = None,
    cache_version: int | None = None,
    cache_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir = ensure_dir(output_dir)
    artifacts: dict[str, dict[str, Any]] = {}
    written: dict[str, Path] = {}

    compare_df = payload["selector_compare_df"]
    compare_path = output_dir / "selector_compare_df.parquet"
    write_parquet(compare_df, compare_path)
    artifacts["selector_compare_df"] = _artifact_entry(compare_path, output_dir, "parquet", len(compare_df))
    written["selector_compare_df"] = compare_path

    fit_payloads = payload.get("fit_payloads", {})
    for selector, fit_payload in fit_payloads.items():
        selector_dir = ensure_dir(output_dir / selector)
        selector_artifacts: dict[str, dict[str, Any]] = {}

        yield_table = fit_payload["yield_table"]
        yield_path = selector_dir / "yield_table.parquet"
        write_parquet(yield_table, yield_path)
        selector_artifacts["yield_table"] = _artifact_entry(yield_path, selector_dir, "parquet", len(yield_table))
        written[f"{selector}.yield_table"] = yield_path

        fit_summary = fit_payload["fit_summary"]
        fit_summary_path = selector_dir / "fit_summary.json"
        write_json(dataframe_to_json_records(fit_summary), fit_summary_path)
        selector_artifacts["fit_summary"] = _artifact_entry(fit_summary_path, selector_dir, "json", len(fit_summary))
        written[f"{selector}.fit_summary"] = fit_summary_path

        parameter_table = fit_payload.get("parameter_table")
        if isinstance(parameter_table, pd.DataFrame):
            parameter_path = selector_dir / "parameter_table.parquet"
            write_parquet(parameter_table, parameter_path)
            selector_artifacts["parameter_table"] = _artifact_entry(parameter_path, selector_dir, "parquet", len(parameter_table))
            written[f"{selector}.parameter_table"] = parameter_path

        ups_peak_significance_table = fit_payload.get("ups_peak_significance_table")
        if isinstance(ups_peak_significance_table, pd.DataFrame):
            ups_peak_path = selector_dir / "ups_peak_significance_table.parquet"
            write_parquet(ups_peak_significance_table, ups_peak_path)
            selector_artifacts["ups_peak_significance_table"] = _artifact_entry(
                ups_peak_path,
                selector_dir,
                "parquet",
                len(ups_peak_significance_table),
            )
            written[f"{selector}.ups_peak_significance_table"] = ups_peak_path

        if backend == "roofit" and "phi_debug_values" in fit_payload:
            phi_debug_path = selector_dir / "phi_debug_values.json"
            write_json(to_jsonable(fit_payload["phi_debug_values"]), phi_debug_path)
            selector_artifacts["phi_debug_values"] = _artifact_entry(phi_debug_path, selector_dir, "json")
            written[f"{selector}.phi_debug_values"] = phi_debug_path

        if plot_style_cfg is not None:
            if backend == "roofit":
                plot_paths = write_roofit_projection_plots(selector_dir, fit_payload, plot_style_cfg, plot_specs=plot_specs)
            elif backend == "iminuit":
                plot_paths = write_iminuit_projection_plots(selector_dir, fit_payload, plot_style_cfg, plot_specs=plot_specs)
            else:
                plot_paths = {}
            for branch, plot_path in plot_paths.items():
                key = f"projection_{branch}"
                selector_artifacts[key] = _artifact_entry(plot_path, selector_dir, "png")
                written[f"{selector}.{key}"] = plot_path

        selector_manifest_path = _write_manifest(
            selector_dir,
            stage=f"{backend}_selector",
            artifacts=selector_artifacts,
            extra={"selector": selector, "backend": backend},
        )
        written[f"{selector}.manifest"] = selector_manifest_path

        artifacts[f"{selector}.yield_table"] = _artifact_entry(yield_path, output_dir, "parquet", len(yield_table))
        artifacts[f"{selector}.fit_summary"] = _artifact_entry(fit_summary_path, output_dir, "json", len(fit_summary))
        if isinstance(parameter_table, pd.DataFrame):
            artifacts[f"{selector}.parameter_table"] = _artifact_entry(parameter_path, output_dir, "parquet", len(parameter_table))
        if isinstance(ups_peak_significance_table, pd.DataFrame):
            artifacts[f"{selector}.ups_peak_significance_table"] = _artifact_entry(
                ups_peak_path,
                output_dir,
                "parquet",
                len(ups_peak_significance_table),
            )
        if backend == "roofit" and "phi_debug_values" in fit_payload:
            artifacts[f"{selector}.phi_debug_values"] = _artifact_entry(phi_debug_path, output_dir, "json")
        if plot_style_cfg is not None:
            for branch in fit_payload.get("fit_branches", ()):
                plot_path = selector_dir / f"projection_{branch}.png"
                if plot_path.exists():
                    artifacts[f"{selector}.projection_{branch}"] = _artifact_entry(plot_path, output_dir, "png")

    if cache_version is not None and cache_payload is not None:
        cache_meta_path = write_stage_cache_metadata(output_dir, f"{backend}_compare", cache_version, cache_payload)
        artifacts["cache_meta"] = _artifact_entry(cache_meta_path, output_dir, "json")
        written["cache_meta"] = cache_meta_path
    manifest_path = _write_manifest(
        output_dir,
        stage=f"{backend}_compare",
        artifacts=artifacts,
        extra={"backend": backend, **(extra_metadata or {})},
    )
    written["manifest"] = manifest_path
    return written


def load_fit_compare_bundle_if_compatible(
    output_dir: Path,
    backend: str,
    cache_version: int,
    cache_payload: dict[str, Any],
) -> dict[str, Any] | None:
    stage_name = f"{backend}_compare"
    if not stage_cache_matches(output_dir, stage_name, cache_version, cache_payload):
        return None
    compare_path = Path(output_dir) / "selector_compare_df.parquet"
    if not compare_path.exists():
        return None
    return {
        "selector_compare_df": read_parquet(compare_path),
        "fit_payloads": {},
    }
