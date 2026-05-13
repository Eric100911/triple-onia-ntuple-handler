from __future__ import annotations

import argparse

from .config import StudyConfig
from .io import resolve_input_files
from .pipeline import load_or_build_cache, validate_config_consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch vertex-truth analysis for MultiLepPAT ntuples.")
    parser.add_argument("input_files", nargs="*", help="Input ROOT ntuples. Wildcard tokens are accepted if quoted.")
    parser.add_argument("--input-glob", help="Deprecated fallback for a single glob pattern.")
    parser.add_argument("--cache-dir", required=True, help="Directory where Parquet cache files will be written.")
    parser.add_argument("--overwrite-cache", action="store_true", help="Rebuild cache even if Parquet files exist.")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache reuse and always rebuild in memory.")
    parser.add_argument("--hide-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    if args.input_files and args.input_glob:
        parser.error("Use either positional input_files or --input-glob, not both.")
    if not args.input_files and not args.input_glob:
        parser.error("Provide at least one input file or --input-glob.")
    return args


def main() -> None:
    args = parse_args()
    resolved_files = resolve_input_files(args.input_files) if args.input_files else ()
    config = StudyConfig(
        input_files=tuple(str(path) for path in resolved_files),
        input_glob=args.input_glob,
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        use_cache=not args.no_cache,
        show_file_progress=not args.hide_progress,
        progress_backend="terminal",
    )
    tables = load_or_build_cache(config)
    candidate_df = tables["candidate_df"]
    event_df = tables["event_df"]
    config_df = tables["config_df"]

    print(f"candidate rows: {len(candidate_df)}")
    print(f"event rows: {len(event_df)}")
    print(f"config rows: {len(config_df)}")
    if not candidate_df.empty and "truth_triple_strict" in candidate_df.columns:
        print(f"truth_triple_strict candidates: {int(candidate_df['truth_triple_strict'].sum())}")

    consistency_df = validate_config_consistency(config_df)
    inconsistent = consistency_df[consistency_df["consistent"] == 0]
    print(f"inconsistent config fields: {len(inconsistent)}")
    if not inconsistent.empty:
        print(inconsistent[["field", "n_unique", "example_values"]].to_string(index=False))


if __name__ == "__main__":
    main()
