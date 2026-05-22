#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from multileppat_vertex_batch.config import CmsPlotStyleConfig, OfflineSelectionConfig
from multileppat_vertex_batch.efficiency import (
    EfficiencyRunConfig,
    build_subprocess_envelope,
    discover_xrootd_sample_files,
    load_efficiency_file_manifest,
    run_efficiency_for_sample,
)
from multileppat_vertex_batch.io import ensure_dir, write_json, write_parquet
from multileppat_vertex_batch.plotting import write_efficiency_plots


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute acceptance and efficiency maps for MultiLepPAT MC ntuples. "
            "The first implemented channel is JpsiJpsiPhi."
        )
    )
    parser.add_argument("--analysis-mode", default="JpsiJpsiPhi", choices=("JpsiJpsiPhi",))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-files", nargs="*", default=None, help="Explicit input ntuple ROOT files or XRootD URLs.")
    parser.add_argument("--input-file-manifest", default=None, help="JSON object mapping sample names to input ROOT files or XRootD URLs.")
    parser.add_argument("--sample-name", default="explicit", help="Sample label used with --input-files.")
    parser.add_argument("--xrootd-host", default="root://cceos.ihep.ac.cn//")
    parser.add_argument("--sample-root", default="/eos/ihep/cms/store/user/xcheng/MC_Production_v3/output")
    parser.add_argument("--samples", default=None, help="Comma-separated samples for XRootD discovery, or a manifest filter when --input-file-manifest is used.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--tree-path", default="mkcands/X_data")
    parser.add_argument("--min-plot-total", type=int, default=1)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--cms-caption", default="Simulation Preliminary")
    parser.add_argument("--cms-energy", type=float, default=13.6)
    parser.add_argument("--cms-lumi", type=float, default=None)
    parser.add_argument("--cms-era", default="Run 3")
    return parser.parse_args(argv)


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _write_sample_bundle(
    sample_dir: Path,
    sample: str,
    files: list[str],
    tables: dict[str, pd.DataFrame],
    plot_style_cfg: CmsPlotStyleConfig,
    min_plot_total: int,
    skip_plots: bool,
) -> dict[str, Any]:
    ensure_dir(sample_dir)
    artifacts: dict[str, Any] = {}
    write_json({"sample": sample, "n_input_files": len(files), "input_files": files}, sample_dir / "sample_manifest.json")
    artifacts["sample_manifest"] = "sample_manifest.json"

    for key in ("gen_systems", "event_step_flags", "efficiency_counts"):
        path = sample_dir / f"{key}.parquet"
        write_parquet(tables[key], path)
        artifacts[key] = {"path": path.name, "n_rows": int(len(tables[key]))}
    maps_path = sample_dir / "efficiency_maps.parquet"
    write_parquet(tables["efficiency_counts"], maps_path)
    artifacts["efficiency_maps"] = {"path": maps_path.name, "n_rows": int(len(tables["efficiency_counts"]))}

    cutflow_path = sample_dir / "cutflow.csv"
    tables["cutflow"].to_csv(cutflow_path, index=False)
    artifacts["cutflow"] = {"path": cutflow_path.name, "n_rows": int(len(tables["cutflow"]))}

    if not skip_plots:
        plot_paths = write_efficiency_plots(
            sample_dir / "plots",
            tables["efficiency_counts"],
            plot_style_cfg=plot_style_cfg,
            min_total=min_plot_total,
        )
        artifacts["plots"] = {key: str(path.relative_to(sample_dir)) for key, path in plot_paths.items()}

    write_json(
        {
            "stage": "efficiency",
            "sample": sample,
            "artifacts": artifacts,
        },
        sample_dir / "manifest.json",
    )
    return artifacts


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
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
    plot_style_cfg = CmsPlotStyleConfig(
        caption=args.cms_caption,
        energy_tev=args.cms_energy,
        lumi_fb=args.cms_lumi,
        era=args.cms_era,
        is_data=False,
    )

    if args.input_files is not None and args.input_file_manifest:
        raise ValueError("--input-files and --input-file-manifest are mutually exclusive.")
    if args.input_files is not None and not args.input_files:
        raise ValueError("--input-files requires at least one file.")

    if args.input_files is not None:
        input_source = "explicit"
        files_by_sample = {args.sample_name: list(args.input_files)}
    elif args.input_file_manifest:
        input_source = "manifest"
        print(f"Loading input file manifest {args.input_file_manifest}")
        files_by_sample = load_efficiency_file_manifest(args.input_file_manifest, samples=samples_filter, max_files=run_cfg.max_files)
    else:
        input_source = "xrootd_discovery"
        print(f"Discovering XRootD samples under {run_cfg.sample_root}")
        files_by_sample = discover_xrootd_sample_files(
            host=run_cfg.xrootd_host,
            sample_root=run_cfg.sample_root,
            samples=run_cfg.samples,
            max_files=run_cfg.max_files,
        )

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
            "cms_plot_style": plot_style_cfg.__dict__,
        },
        output_dir / "run_metadata.json",
    )

    sample_count_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    for sample, files in files_by_sample.items():
        if not files:
            print(f"Skipping {sample}: no input files found")
            continue
        print(f"Running efficiency stage for {sample}: {len(files)} files")
        tables = run_efficiency_for_sample(files, sample, cfg=offline_cfg, tree_path=run_cfg.tree_path)
        sample_count_tables[sample] = tables["efficiency_counts"]
        sample_dir = ensure_dir(output_dir / sample)
        _write_sample_bundle(
            sample_dir,
            sample,
            files,
            tables,
            plot_style_cfg=plot_style_cfg,
            min_plot_total=run_cfg.min_plot_total,
            skip_plots=args.skip_plots,
        )
        inclusive_final = tables["cutflow"].loc[tables["cutflow"]["step"] == "final_nominal"]
        summary_rows.append(
            {
                "sample": sample,
                "n_input_files": len(files),
                "n_full_gen": int(tables["event_step_flags"]["full_gen"].sum()) if not tables["event_step_flags"].empty else 0,
                "n_final_nominal": int(tables["event_step_flags"]["final_nominal"].sum()) if not tables["event_step_flags"].empty else 0,
                "final_efficiency": float(inclusive_final["efficiency"].iloc[0]) if not inclusive_final.empty else float("nan"),
                "final_err_sym": float(inclusive_final["err_sym"].iloc[0]) if not inclusive_final.empty else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    write_parquet(summary_df, output_dir / "subprocess_summary.parquet")
    summary_df.to_csv(output_dir / "subprocess_summary.csv", index=False)

    envelope_df = build_subprocess_envelope(sample_count_tables)
    write_parquet(envelope_df, output_dir / "subprocess_envelope.parquet")
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
        output_dir / "manifest.json",
    )
    print(f"Wrote efficiency outputs to {output_dir}")


if __name__ == "__main__":
    main()
