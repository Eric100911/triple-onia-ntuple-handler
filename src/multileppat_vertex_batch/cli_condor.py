#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from multileppat_vertex_batch.cache import (
    stage_cache_key,
    write_mass_selection_bundle,
    write_run_metadata,
    write_truth_cache_bundle,
)
from multileppat_vertex_batch.cli_batch import (
    active_windows_from_offline_defaults,
    build_mass_selection_cache_payload,
    format_missing_truth_branch_error,
    inspect_inputs,
    parse_selectors,
    run_iminuit_stage,
    run_roofit_stage,
)
from multileppat_vertex_batch.config import CmsPlotStyleConfig, MassStudyConfig, OfflineSelectionConfig, StudyConfig
from multileppat_vertex_batch.efficiency import (
    EfficiencyBinning,
    EfficiencyRunConfig,
    build_cutflow,
    build_efficiency_counts,
    build_subprocess_envelope,
    discover_xrootd_sample_files,
    load_efficiency_file_manifest,
    process_efficiency_file,
)
from multileppat_vertex_batch.fit_roofit import JPSI_PDF_PRESETS, build_fit_frame, build_fit_mask
from multileppat_vertex_batch.io import ensure_dir, read_json, read_parquet, resolve_input_files, snapshot_input_files, write_json, write_parquet
from multileppat_vertex_batch.pipeline import process_single_file, validate_config_consistency
from multileppat_vertex_batch.schema import (
    CONFIG_BRANCHES,
    CORE_DATA_BRANCHES,
    FIT_BRANCHES_BY_MODE,
    MASS_SELECTION_CACHE_VERSION,
    TRUTH_CACHE_VERSION,
    get_analysis_mode_spec,
)
from multileppat_vertex_batch.selection import (
    run_mass_selection_batch,
    select_best_candidates,
    summarize_mass_window_flow,
    summarize_selection,
)


LCG_SETUP = "/cvmfs/sft.cern.ch/lcg/views/LCG_109a/x86_64-el9-gcc13-opt/setup.sh"


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _progress(message: str) -> None:
    print(f"[condor] {message}", flush=True)


def _job_name(prefix: str, index: int) -> str:
    return f"{prefix}_{index:06d}"


def _safe_job_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return token.strip("_") or "sample"


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _write_worker_script(condor_dir: Path, repo_dir: Path) -> Path:
    path = condor_dir / "worker.sh"
    text = f"""#!/usr/bin/env bash
set -euo pipefail
source {_quote(LCG_SETUP)}
export MPLCONFIGDIR="${{MPLCONFIGDIR:-/tmp/chiw/mplconfig_multileppat_vertex_batch}}"
export PYTHONPYCACHEPREFIX="${{PYTHONPYCACHEPREFIX:-/tmp/chiw/pycache_multileppat_vertex_batch}}"
mkdir -p "$MPLCONFIGDIR" "$PYTHONPYCACHEPREFIX"
cd {_quote(repo_dir)}
python -m multileppat_vertex_batch.cli_condor "$@"
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_submit_file(condor_dir: Path, worker_script: Path) -> Path:
    path = condor_dir / "multileppat.sub"
    text = f"""universe = vanilla
initialdir = {condor_dir}
executable = {worker_script}
arguments = $(cmd) $(args_json)
output = logs/$(job).out
error = logs/$(job).err
log = logs/$(Cluster).log
request_cpus = 1
request_memory = 4 GB
should_transfer_files = NO
getenv = True
queue
"""
    path.write_text(text, encoding="utf-8")
    return path


def _write_dag(condor_dir: Path, submit_file: Path, jobs: list[dict[str, str]], parents: list[tuple[list[str], str]]) -> Path:
    path = condor_dir / "workflow.dag"
    lines: list[str] = []
    submit_path = submit_file.resolve()
    for job in jobs:
        lines.append(f"JOB {job['job']} {submit_path}")
        lines.append(f"VARS {job['job']} job=\"{job['job']}\" cmd=\"{job['cmd']}\" args_json=\"{job['args_json']}\"")
    for parent_jobs, child in parents:
        if parent_jobs:
            lines.append(f"PARENT {' '.join(parent_jobs)} CHILD {child}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_arg_json(args_dir: Path, job: str, payload: dict[str, Any]) -> Path:
    path = args_dir / f"{job}.json"
    write_json(payload, path)
    return path


def _common_mass_config(payload: dict[str, Any]) -> tuple[StudyConfig, OfflineSelectionConfig, MassStudyConfig]:
    config = StudyConfig(
        input_files=tuple(payload.get("input_files", ())),
        tree_path=payload.get("tree_path", "mkcands/X_data"),
        config_tree_path=payload.get("config_tree_path", "mkcands/X_config"),
        cache_dir=Path(payload.get("cache_dir", ".")),
        use_cache=False,
        overwrite_cache=True,
        show_file_progress=False,
        show_event_progress=False,
        progress_backend="terminal",
    )
    offline_cfg = OfflineSelectionConfig(**payload["offline_selection"])
    study_cfg = MassStudyConfig(
        analysis_mode=payload["analysis_mode"],
        active_windows={key: tuple(value) for key, value in payload["active_windows"].items()},
        selector_name=payload["selector_name"],
        selectors=tuple(payload["selectors"]),
        fit_branches=tuple(payload["fit_branches"]),
    )
    return config, offline_cfg, study_cfg


def _concat_existing(paths: list[Path]) -> pd.DataFrame:
    frames = [read_parquet(path) for path in paths if path.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def worker_mass_file(args_json: Path) -> None:
    payload = read_json(args_json)
    output_dir = ensure_dir(Path(payload["output_dir"]))
    source_file = Path(payload["input_file"])
    config, offline_cfg, study_cfg = _common_mass_config({**payload, "input_files": [str(source_file)]})

    if payload.get("truth_enabled", False):
        candidate_df, event_df, config_row = process_single_file(source_file, config)
        truth_dir = ensure_dir(output_dir / "truth")
        write_parquet(candidate_df, truth_dir / "candidate_df.parquet")
        write_parquet(event_df, truth_dir / "event_df.parquet")
        write_parquet(pd.DataFrame([config_row]), truth_dir / "config_df.parquet")

    tables = run_mass_selection_batch(
        files=[source_file],
        config=config,
        active_windows=study_cfg.active_windows,
        selection_cfg=offline_cfg,
        analysis_mode=study_cfg.analysis_mode,
        selectors=study_cfg.selectors,
        show_progress=False,
    )
    selection_dir = ensure_dir(output_dir / "selection")
    for key in ("candidate_pool_df", "audit_df"):
        write_parquet(tables[key], selection_dir / f"{key}.parquet")
    write_json({"input_file": str(source_file), "stage": "mass_file"}, output_dir / "manifest.json")


def worker_mass_merge(args_json: Path) -> None:
    payload = read_json(args_json)
    output_dir = ensure_dir(Path(payload["output_dir"]))
    final_output_dir = ensure_dir(Path(payload["final_output_dir"]))
    file_output_dirs = [Path(item) for item in payload["file_output_dirs"]]
    config, offline_cfg, study_cfg = _common_mass_config(payload)

    if payload.get("truth_enabled", False):
        truth_tables = {
            "candidate_df": _concat_existing([path / "truth" / "candidate_df.parquet" for path in file_output_dirs]),
            "event_df": _concat_existing([path / "truth" / "event_df.parquet" for path in file_output_dirs]),
            "config_df": _concat_existing([path / "truth" / "config_df.parquet" for path in file_output_dirs]),
        }
        consistency_df = validate_config_consistency(truth_tables["config_df"])
        write_truth_cache_bundle(
            final_output_dir / "truth_cache",
            payload["input_files"],
            truth_tables,
            consistency_df,
            cache_version=TRUTH_CACHE_VERSION,
            cache_payload={
                "input_files": snapshot_input_files(payload["input_files"]),
                "tree_path": config.tree_path,
                "config_tree_path": config.config_tree_path,
                "core_data_branches": list(CORE_DATA_BRANCHES),
                "config_branches": list(CONFIG_BRANCHES),
            },
        )

    candidate_pool_df = _concat_existing([path / "selection" / "candidate_pool_df.parquet" for path in file_output_dirs])
    audit_df = _concat_existing([path / "selection" / "audit_df.parquet" for path in file_output_dirs])
    mode_spec = get_analysis_mode_spec(study_cfg.analysis_mode)
    selected_candidate_df = select_best_candidates(candidate_pool_df, mode_spec)
    window_audit_df = summarize_mass_window_flow(audit_df)
    selection_summary_df = summarize_selection(candidate_pool_df, selected_candidate_df, audit_df)
    selected_for_selector_df = selected_candidate_df.loc[selected_candidate_df["selector"] == study_cfg.selector_name].reset_index(drop=True)
    fit_df = build_fit_frame(selected_for_selector_df, list(study_cfg.fit_branches), study_cfg.active_windows)
    fit_input_dfs_by_selector: dict[str, pd.DataFrame] = {}
    for selector in study_cfg.selectors:
        selector_selected = selected_candidate_df.loc[selected_candidate_df["selector"] == selector].reset_index(drop=True)
        if selector_selected.empty:
            fit_input_dfs_by_selector[selector] = selector_selected.copy()
        else:
            fit_mask = build_fit_mask(selector_selected, list(study_cfg.fit_branches), study_cfg.active_windows)
            fit_input_dfs_by_selector[selector] = selector_selected.loc[fit_mask].reset_index(drop=True)

    mass_tables = {
        "candidate_pool_df": candidate_pool_df,
        "selected_candidate_df": selected_candidate_df,
        "audit_df": audit_df,
        "window_audit_df": window_audit_df,
        "selection_summary_df": selection_summary_df,
        "selected_for_selector_df": selected_for_selector_df,
        "fit_df": fit_df,
        "fit_input_dfs_by_selector": fit_input_dfs_by_selector,
    }
    write_mass_selection_bundle(
        final_output_dir / "mass_selection",
        mass_tables,
        cache_version=MASS_SELECTION_CACHE_VERSION,
        cache_payload=build_mass_selection_cache_payload(
            [Path(item) for item in payload["input_files"]],
            config,
            study_cfg,
            offline_cfg,
        ),
    )
    if payload.get("write_merged_parquets", False):
        write_parquet(candidate_pool_df, final_output_dir / "merged_candidate_pool.parquet")
        write_parquet(selected_candidate_df, final_output_dir / "merged_selected_candidates.parquet")
        write_parquet(fit_df, final_output_dir / "merged_fit_input.parquet")
    write_json({"stage": "mass_merge", "n_file_jobs": len(file_output_dirs)}, output_dir / "manifest.json")


def worker_fit_reduce(args_json: Path) -> None:
    payload = read_json(args_json)
    final_output_dir = ensure_dir(Path(payload["final_output_dir"]))
    config, _, study_cfg = _common_mass_config(payload)
    plot_style_cfg = CmsPlotStyleConfig(**payload["cms_plot_style"])
    mass_dir = final_output_dir / "mass_selection"
    mass_tables = {
        "selected_candidate_df": read_parquet(mass_dir / "selected_candidate_df.parquet"),
        "selection_summary_df": read_parquet(mass_dir / "selection_summary_df.parquet"),
    }
    mass_selection_cache_key = stage_cache_key(
        "mass_selection",
        MASS_SELECTION_CACHE_VERSION,
        build_mass_selection_cache_payload(
            [Path(item) for item in payload["input_files"]],
            config,
            study_cfg,
            OfflineSelectionConfig(**payload["offline_selection"]),
        ),
    )
    backend = payload["backend"]
    if backend == "roofit":
        run_roofit_stage(
            selected_candidate_df=mass_tables["selected_candidate_df"],
            selection_summary_df=mass_tables["selection_summary_df"],
            study_cfg=study_cfg,
            jpsi_pdf_preset=payload["jpsi_pdf_preset"],
            phi_background_kind=payload["phi_background_kind"],
            ups_background_order=payload["ups_background_order"],
            output_dir=final_output_dir / "roofit_compare",
            plot_style_cfg=plot_style_cfg,
            mass_selection_cache_key=mass_selection_cache_key,
            write_plots=False,
        )
    elif backend == "iminuit":
        run_iminuit_stage(
            selected_candidate_df=mass_tables["selected_candidate_df"],
            selection_summary_df=mass_tables["selection_summary_df"],
            study_cfg=study_cfg,
            jpsi_pdf_preset=payload["jpsi_pdf_preset"],
            ups_background_order=payload["ups_background_order"],
            output_dir=final_output_dir / "iminuit_compare",
            plot_style_cfg=plot_style_cfg,
            mass_selection_cache_key=mass_selection_cache_key,
            write_plots=False,
        )
    else:
        raise ValueError(f"Unsupported fit backend worker: {backend}")
    write_json({"stage": "fit_reduce", "backend": backend, "cms_plot_style": plot_style_cfg.__dict__}, Path(payload["output_dir"]) / "manifest.json")


def worker_efficiency_file(args_json: Path) -> None:
    payload = read_json(args_json)
    output_dir = ensure_dir(Path(payload["output_dir"]))
    tables = process_efficiency_file(
        path=payload["input_file"],
        sample=payload["sample"],
        cfg=OfflineSelectionConfig(**payload["offline_selection"]),
        tree_path=payload["tree_path"],
    )
    for key, frame in tables.items():
        write_parquet(frame, output_dir / f"{key}.parquet")
    write_json({"stage": "efficiency_file", "sample": payload["sample"], "input_file": payload["input_file"]}, output_dir / "manifest.json")


def worker_efficiency_merge(args_json: Path) -> None:
    payload = read_json(args_json)
    final_output_dir = ensure_dir(Path(payload["final_output_dir"]))
    samples = payload["samples"]
    sample_count_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    gen_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []

    for sample in samples:
        sample_jobs = [Path(item) for item in payload["sample_file_output_dirs"][sample]]
        gen_df = _concat_existing([path / "gen_systems.parquet" for path in sample_jobs])
        event_df = _concat_existing([path / "event_step_flags.parquet" for path in sample_jobs])
        if not gen_df.empty:
            gen_parts.append(gen_df)
        if not event_df.empty:
            event_parts.append(event_df)
        counts_df = build_efficiency_counts(gen_df, event_df, EfficiencyBinning())
        cutflow_df = build_cutflow(event_df)
        sample_count_tables[sample] = counts_df
        sample_dir = ensure_dir(final_output_dir / sample)
        write_json({"sample": sample, "n_input_files": len(sample_jobs), "input_files": payload["files_by_sample"][sample]}, sample_dir / "sample_manifest.json")
        write_parquet(gen_df, sample_dir / "gen_systems.parquet")
        write_parquet(event_df, sample_dir / "event_step_flags.parquet")
        write_parquet(counts_df, sample_dir / "efficiency_counts.parquet")
        write_parquet(counts_df, sample_dir / "efficiency_maps.parquet")
        cutflow_df.to_csv(sample_dir / "cutflow.csv", index=False)
        write_json(
            {
                "stage": "efficiency",
                "sample": sample,
                "artifacts": {
                    "gen_systems": {"path": "gen_systems.parquet", "n_rows": int(len(gen_df))},
                    "event_step_flags": {"path": "event_step_flags.parquet", "n_rows": int(len(event_df))},
                    "efficiency_counts": {"path": "efficiency_counts.parquet", "n_rows": int(len(counts_df))},
                    "efficiency_maps": {"path": "efficiency_maps.parquet", "n_rows": int(len(counts_df))},
                    "cutflow": {"path": "cutflow.csv", "n_rows": int(len(cutflow_df))},
                },
            },
            sample_dir / "manifest.json",
        )
        inclusive_final = cutflow_df.loc[cutflow_df["step"] == "final_nominal"] if not cutflow_df.empty else pd.DataFrame()
        summary_rows.append(
            {
                "sample": sample,
                "n_input_files": len(sample_jobs),
                "n_full_gen": int(event_df["full_gen"].sum()) if not event_df.empty and "full_gen" in event_df else 0,
                "n_final_nominal": int(event_df["final_nominal"].sum()) if not event_df.empty and "final_nominal" in event_df else 0,
                "final_efficiency": float(inclusive_final["efficiency"].iloc[0]) if not inclusive_final.empty else float("nan"),
                "final_err_sym": float(inclusive_final["err_sym"].iloc[0]) if not inclusive_final.empty else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    write_parquet(summary_df, final_output_dir / "subprocess_summary.parquet")
    summary_df.to_csv(final_output_dir / "subprocess_summary.csv", index=False)
    envelope_df = build_subprocess_envelope(sample_count_tables)
    write_parquet(envelope_df, final_output_dir / "subprocess_envelope.parquet")
    if payload.get("write_merged_parquets", False):
        merged_gen_df = pd.concat(gen_parts, ignore_index=True) if gen_parts else pd.DataFrame()
        merged_event_df = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
        write_parquet(merged_gen_df, final_output_dir / "merged_gen_systems.parquet")
        write_parquet(merged_event_df, final_output_dir / "merged_event_step_flags.parquet")
    write_json(
        {
            "stage": "efficiency_summary",
            "artifacts": {
                "run_metadata": "run_metadata.json",
                "subprocess_summary": "subprocess_summary.parquet",
                "subprocess_envelope": "subprocess_envelope.parquet",
                "samples": {sample: f"{sample}/manifest.json" for sample in sample_count_tables},
            },
        },
        final_output_dir / "manifest.json",
    )
    write_json({"stage": "efficiency_merge", "samples": samples}, Path(payload["output_dir"]) / "manifest.json")


def _build_mass_plan(args: argparse.Namespace, condor_dir: Path, repo_dir: Path) -> Path:
    _progress("resolving mass-study input files")
    files = resolve_input_files(args.input_files)
    _progress(f"resolved {len(files)} input files")
    final_output_dir = ensure_dir(Path(args.output_dir))
    selectors = parse_selectors(args.selectors)
    probe_config = StudyConfig(input_files=tuple(str(path) for path in files), show_file_progress=False, progress_backend="terminal")
    _progress("inspecting X_config and truth-branch availability")
    inspection = inspect_inputs(files, probe_config)
    if inspection["analysis_mode"] != args.analysis_mode:
        raise RuntimeError(f"--analysis-mode {args.analysis_mode} does not match X_config AnalysisMode {inspection['analysis_mode']}.")
    if inspection["missing_truth_branches"]:
        raise RuntimeError(format_missing_truth_branch_error(inspection["missing_truth_branches"]))

    analysis_mode = inspection["analysis_mode"]
    is_mc = inspection["is_mc"]
    config_row = inspection["config_rows"][0]
    offline_cfg = OfflineSelectionConfig()
    if offline_cfg.ups_vtxprob_min is None:
        offline_cfg = replace(offline_cfg, ups_vtxprob_min=float(config_row["UpsDecayVtxProbCut"]))
    active_windows = active_windows_from_offline_defaults(analysis_mode, config_row, offline_cfg)
    fit_branches = tuple(FIT_BRANCHES_BY_MODE.get(analysis_mode, ()))
    study_cfg = MassStudyConfig(
        analysis_mode=analysis_mode,
        active_windows=active_windows,
        selector_name=args.selector_name,
        selectors=selectors,
        fit_branches=fit_branches,
    )
    plot_style_cfg = CmsPlotStyleConfig(
        caption=args.cms_caption,
        energy_tev=args.cms_energy,
        lumi_fb=args.cms_lumi,
        era=args.cms_era,
        is_data=not is_mc,
    )
    truth_enabled = bool(is_mc and not args.skip_truth)
    _progress(f"analysis_mode={analysis_mode}, is_mc={int(is_mc)}, truth_enabled={int(truth_enabled)}")

    _progress(f"writing run metadata to {final_output_dir}")
    write_run_metadata(
        final_output_dir,
        {
            "input_files": [str(path) for path in files],
            "n_input_files": len(files),
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
            "condor_dir": str(condor_dir),
        },
    )

    args_dir = ensure_dir(condor_dir / "args")
    file_root = ensure_dir(condor_dir / "mass_files")
    jobs: list[dict[str, str]] = []
    file_job_names: list[str] = []
    file_output_dirs: list[str] = []
    common_payload = {
        "analysis_mode": analysis_mode,
        "tree_path": probe_config.tree_path,
        "config_tree_path": probe_config.config_tree_path,
        "offline_selection": offline_cfg.__dict__,
        "active_windows": active_windows,
        "selector_name": study_cfg.selector_name,
        "selectors": list(study_cfg.selectors),
        "fit_branches": list(study_cfg.fit_branches),
        "input_files": [str(path) for path in files],
        "cache_dir": str(final_output_dir / "truth_cache"),
        "truth_enabled": truth_enabled,
        "write_merged_parquets": bool(args.write_merged_parquets),
    }
    _progress(f"writing {len(files)} one-file mass jobs")
    for index, path in enumerate(files):
        job = _job_name("mass_file", index)
        job_dir = file_root / job
        file_output_dirs.append(str(job_dir))
        arg_path = _write_arg_json(args_dir, job, {**common_payload, "input_file": str(path), "output_dir": str(job_dir)})
        jobs.append({"job": job, "cmd": "worker-mass-file", "args_json": str(arg_path)})
        file_job_names.append(job)
        if (index + 1) % 100 == 0 or index + 1 == len(files):
            _progress(f"prepared {index + 1}/{len(files)} mass file jobs")

    merge_job = "mass_merge"
    _progress("writing mass merge job")
    merge_arg = _write_arg_json(
        args_dir,
        merge_job,
        {**common_payload, "output_dir": str(condor_dir / merge_job), "final_output_dir": str(final_output_dir), "file_output_dirs": file_output_dirs},
    )
    jobs.append({"job": merge_job, "cmd": "worker-mass-merge", "args_json": str(merge_arg)})
    parents: list[tuple[list[str], str]] = [(file_job_names, merge_job)]

    if args.fit_backend in ("roofit", "both"):
        _progress("writing RooFit reduce job")
        job = "fit_roofit"
        arg_path = _write_arg_json(
            args_dir,
            job,
            {
                **common_payload,
                "backend": "roofit",
                "output_dir": str(condor_dir / job),
                "final_output_dir": str(final_output_dir),
                "jpsi_pdf_preset": args.jpsi_pdf_preset,
                "phi_background_kind": args.phi_background_kind,
                "ups_background_order": args.ups_background_order,
                "cms_plot_style": plot_style_cfg.__dict__,
            },
        )
        jobs.append({"job": job, "cmd": "worker-fit-reduce", "args_json": str(arg_path)})
        parents.append(([merge_job], job))
    if args.fit_backend in ("iminuit", "both"):
        _progress("writing iminuit reduce job")
        job = "fit_iminuit"
        arg_path = _write_arg_json(
            args_dir,
            job,
            {
                **common_payload,
                "backend": "iminuit",
                "output_dir": str(condor_dir / job),
                "final_output_dir": str(final_output_dir),
                "jpsi_pdf_preset": args.jpsi_pdf_preset,
                "phi_background_kind": args.phi_background_kind,
                "ups_background_order": args.ups_background_order,
                "cms_plot_style": plot_style_cfg.__dict__,
            },
        )
        jobs.append({"job": job, "cmd": "worker-fit-reduce", "args_json": str(arg_path)})
        parents.append(([merge_job], job))

    worker_script = _write_worker_script(condor_dir, repo_dir)
    submit_file = _write_submit_file(condor_dir, worker_script)
    _progress(f"writing DAG with {len(jobs)} jobs")
    return _write_dag(condor_dir, submit_file, jobs, parents)


def _build_efficiency_plan(args: argparse.Namespace, condor_dir: Path, repo_dir: Path) -> Path:
    final_output_dir = ensure_dir(Path(args.output_dir))
    _progress("building efficiency Condor plan")
    samples_filter = _parse_csv(args.samples) if args.samples is not None else None
    run_samples = samples_filter if samples_filter is not None else EfficiencyRunConfig().samples
    run_cfg = EfficiencyRunConfig(
        analysis_mode=args.analysis_mode,
        tree_path=args.tree_path,
        xrootd_host=args.xrootd_host,
        sample_root=args.sample_root,
        samples=run_samples,
        max_files=args.max_files,
        min_plot_total=args.min_plot_total,
    )
    offline_cfg = OfflineSelectionConfig()
    if args.input_files is not None and args.input_file_manifest:
        raise ValueError("--input-files and --input-file-manifest are mutually exclusive.")
    if args.input_files is not None and not args.input_files:
        raise ValueError("--input-files requires at least one file.")

    if args.input_files is not None:
        input_source = "explicit"
        _progress(f"using {len(args.input_files)} explicit input files for sample {args.sample_name}")
        files_by_sample = {args.sample_name: list(args.input_files)}
    elif args.input_file_manifest:
        input_source = "manifest"
        _progress(f"loading input file manifest {args.input_file_manifest}")
        files_by_sample = load_efficiency_file_manifest(args.input_file_manifest, samples=samples_filter, max_files=run_cfg.max_files)
    else:
        input_source = "xrootd_discovery"
        _progress(f"discovering XRootD sample files under {run_cfg.sample_root}")
        files_by_sample = discover_xrootd_sample_files(
            host=run_cfg.xrootd_host,
            sample_root=run_cfg.sample_root,
            samples=run_cfg.samples,
            max_files=run_cfg.max_files,
        )
    files_by_sample = {sample: files for sample, files in files_by_sample.items() if files}
    total_files = sum(len(files) for files in files_by_sample.values())
    _progress(f"resolved {total_files} files across {len(files_by_sample)} samples")
    for sample, files in files_by_sample.items():
        _progress(f"sample {sample}: {len(files)} files")
    _progress(f"writing run metadata to {final_output_dir}")
    write_json(
        {
            "analysis_mode": run_cfg.analysis_mode,
            "tree_path": run_cfg.tree_path,
            "xrootd_host": run_cfg.xrootd_host,
            "sample_root": run_cfg.sample_root,
            "samples": list(files_by_sample),
            "requested_samples": list(samples_filter or run_cfg.samples),
            "max_files": run_cfg.max_files,
            "min_plot_total": run_cfg.min_plot_total,
            "input_source": input_source,
            "input_file_manifest": str(args.input_file_manifest) if args.input_file_manifest else None,
            "offline_selection": offline_cfg.__dict__,
            "cms_plot_style": CmsPlotStyleConfig(caption=args.cms_caption, energy_tev=args.cms_energy, lumi_fb=args.cms_lumi, era=args.cms_era, is_data=False).__dict__,
            "condor_dir": str(condor_dir),
        },
        final_output_dir / "run_metadata.json",
    )

    args_dir = ensure_dir(condor_dir / "args")
    job_root = ensure_dir(condor_dir / "efficiency_files")
    jobs: list[dict[str, str]] = []
    file_job_names: list[str] = []
    sample_file_output_dirs: dict[str, list[str]] = {}
    prepared = 0
    for sample, files in files_by_sample.items():
        sample_file_output_dirs[sample] = []
        _progress(f"writing one-file jobs for sample {sample}")
        for index, input_file in enumerate(files):
            job = f"eff_{_safe_job_token(sample)}_{index:06d}"
            job_dir = job_root / sample / f"file_{index:06d}"
            sample_file_output_dirs[sample].append(str(job_dir))
            arg_path = _write_arg_json(
                args_dir,
                job,
                {
                    "sample": sample,
                    "input_file": input_file,
                    "output_dir": str(job_dir),
                    "tree_path": run_cfg.tree_path,
                    "offline_selection": offline_cfg.__dict__,
                },
            )
            jobs.append({"job": job, "cmd": "worker-efficiency-file", "args_json": str(arg_path)})
            file_job_names.append(job)
            prepared += 1
            if prepared % 100 == 0 or prepared == total_files:
                _progress(f"prepared {prepared}/{total_files} efficiency file jobs")

    merge_job = "efficiency_merge"
    _progress("writing efficiency merge job")
    merge_arg = _write_arg_json(
        args_dir,
        merge_job,
        {
            "output_dir": str(condor_dir / merge_job),
            "final_output_dir": str(final_output_dir),
            "samples": list(files_by_sample),
            "files_by_sample": files_by_sample,
            "sample_file_output_dirs": sample_file_output_dirs,
            "write_merged_parquets": bool(args.write_merged_parquets),
        },
    )
    jobs.append({"job": merge_job, "cmd": "worker-efficiency-merge", "args_json": str(merge_arg)})
    worker_script = _write_worker_script(condor_dir, repo_dir)
    submit_file = _write_submit_file(condor_dir, worker_script)
    _progress(f"writing DAG with {len(jobs)} jobs")
    return _write_dag(condor_dir, submit_file, jobs, [(file_job_names, merge_job)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create one-file-per-task HTCondor workflows for MultiLepPAT ntuple processing.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mass = subparsers.add_parser("mass", help="Generate a Condor DAG for truth, mass selection, and optional fit-reduce stages.")
    mass.add_argument("input_files", nargs="+")
    mass.add_argument("--analysis-mode", required=True, choices=("JpsiJpsiPhi", "JpsiJpsiUps", "JpsiUpsPhi"))
    mass.add_argument("--output-dir", required=True)
    mass.add_argument("--condor-dir", required=True)
    mass.add_argument("--selectors", default="all6_same_recVtx,Pri_fitValid")
    mass.add_argument("--selector-name", default="all6_same_recVtx")
    mass.add_argument("--fit-backend", choices=("none", "roofit", "iminuit", "both"), default="none")
    mass.add_argument("--phi-background-kind", choices=("polynomial", "chebychev"), default="polynomial")
    mass.add_argument("--ups-background-order", type=int, choices=(1, 2, 3, 4), default=4)
    mass.add_argument("--jpsi-pdf-preset", choices=tuple(sorted(JPSI_PDF_PRESETS)), default="small_sample")
    mass.add_argument("--skip-truth", action="store_true")
    mass.add_argument("--cms-caption", default="Work In Progress")
    mass.add_argument("--cms-energy", type=float, default=13.6)
    mass.add_argument("--cms-lumi", type=float, default=283.4)
    mass.add_argument("--cms-era", default="Run 3 (2022-2025)")
    mass.add_argument("--write-merged-parquets", action="store_true", help="Also write top-level merged Parquet tables for downstream analysis.")
    mass.add_argument("--submit", action="store_true")

    eff = subparsers.add_parser("efficiency", help="Generate a Condor DAG for one-file efficiency processing and sample merge.")
    eff.add_argument("--analysis-mode", default="JpsiJpsiPhi", choices=("JpsiJpsiPhi",))
    eff.add_argument("--output-dir", required=True)
    eff.add_argument("--condor-dir", required=True)
    eff.add_argument("--input-files", nargs="*", default=None)
    eff.add_argument("--input-file-manifest", default=None, help="JSON object mapping sample names to input ROOT files or XRootD URLs.")
    eff.add_argument("--sample-name", default="explicit")
    eff.add_argument("--xrootd-host", default="root://cceos.ihep.ac.cn//")
    eff.add_argument("--sample-root", default="/eos/ihep/cms/store/user/xcheng/MC_Production_v3/output")
    eff.add_argument("--samples", default=None, help="Comma-separated samples for XRootD discovery, or a manifest filter when --input-file-manifest is used.")
    eff.add_argument("--max-files", type=int, default=None)
    eff.add_argument("--tree-path", default="mkcands/X_data")
    eff.add_argument("--min-plot-total", type=int, default=1)
    eff.add_argument("--cms-caption", default="Simulation Preliminary")
    eff.add_argument("--cms-energy", type=float, default=13.6)
    eff.add_argument("--cms-lumi", type=float, default=None)
    eff.add_argument("--cms-era", default="Run 3")
    eff.add_argument("--write-merged-parquets", action="store_true", help="Also write top-level merged GEN/event Parquet tables across all samples.")
    eff.add_argument("--submit", action="store_true")

    for name in ("worker-mass-file", "worker-mass-merge", "worker-fit-reduce", "worker-efficiency-file", "worker-efficiency-merge"):
        worker = subparsers.add_parser(name)
        worker.add_argument("args_json")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    worker_commands = {
        "worker-mass-file": worker_mass_file,
        "worker-mass-merge": worker_mass_merge,
        "worker-fit-reduce": worker_fit_reduce,
        "worker-efficiency-file": worker_efficiency_file,
        "worker-efficiency-merge": worker_efficiency_merge,
    }
    if args.command in worker_commands:
        worker_commands[args.command](Path(args.args_json))
        return

    condor_dir = ensure_dir(Path(args.condor_dir)).resolve()
    ensure_dir(condor_dir / "logs")
    repo_dir = Path(os.getcwd()).resolve()
    if args.command == "mass":
        dag_path = _build_mass_plan(args, condor_dir, repo_dir)
    elif args.command == "efficiency":
        dag_path = _build_efficiency_plan(args, condor_dir, repo_dir)
    else:
        parser.error(f"Unsupported command: {args.command}")
    print(f"Wrote Condor DAG: {dag_path}")
    print(f"Submit command: condor_submit_dag {dag_path}")
    if args.submit:
        _progress("submitting DAG")
        subprocess.run(["condor_submit_dag", str(dag_path)], check=True)


if __name__ == "__main__":
    main()
