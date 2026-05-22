from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from glob import glob
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from .config import StudyConfig
from .schema import CONFIG_BRANCHES


def natural_sample_sort_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    key: list[Any] = []
    for part in parts:
        if not part:
            continue
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def discover_ntuple_files(glob_pattern: str) -> list[Path]:
    paths = [Path(item) for item in glob(glob_pattern, recursive=True)]
    return sorted(paths, key=natural_sample_sort_key)


def resolve_input_files(inputs: Sequence[str | Path]) -> list[Path]:
    resolved: list[Path] = []
    for item in inputs:
        raw = str(item)
        if any(char in raw for char in "*?[]"):
            matches = discover_ntuple_files(raw)
            if not matches:
                raise FileNotFoundError(f"No input ROOT files matched wildcard token: {raw}")
            resolved.extend(matches)
            continue

        path = Path(raw)
        if not path.exists():
            raise FileNotFoundError(f"Input ROOT file does not exist: {path}")
        resolved.append(path)

    deduped: dict[str, Path] = {}
    for path in resolved:
        key = str(path.resolve())
        deduped.setdefault(key, path)
    return sorted(deduped.values(), key=natural_sample_sort_key)


def resolve_study_input_files(config: StudyConfig) -> list[Path]:
    if config.input_files:
        return resolve_input_files(config.input_files)
    if config.input_glob:
        files = discover_ntuple_files(config.input_glob)
        if files:
            return files
        raise FileNotFoundError(f"No input ROOT files matched: {config.input_glob}")
    raise ValueError("StudyConfig requires either input_files or input_glob.")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def read_config_snapshot(path: Path, config: StudyConfig) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"source_file": str(path)}
    with uproot.open(path) as root_file:
        if config.config_tree_path not in root_file:
            for branch in CONFIG_BRANCHES:
                snapshot[branch] = None
            return snapshot
        tree = root_file[config.config_tree_path]
        arrays = tree.arrays(CONFIG_BRANCHES, entry_start=0, entry_stop=1, library="ak")

    fields = set(arrays.fields)
    for branch in CONFIG_BRANCHES:
        branch_values = arrays[branch] if branch in fields else None
        if branch_values is None or len(branch_values) == 0:
            snapshot[branch] = None
            continue
        snapshot[branch] = _normalize_value(ak.to_list(branch_values[0]))
    return snapshot


def read_ntuple_arrays(path: Path, config: StudyConfig, branches: list[str]) -> ak.Array:
    with uproot.open(path) as root_file:
        tree = root_file[config.tree_path]
        return tree.arrays(branches, library="ak")


def read_ntuple_entry(path: Path, config: StudyConfig, branches: list[str], entry: int) -> dict[str, Any]:
    with uproot.open(path) as root_file:
        tree = root_file[config.tree_path]
        arrays = tree.arrays(branches, entry_start=entry, entry_stop=entry + 1, library="ak")
    return {name: _normalize_value(ak.to_list(arrays[name][0])) for name in arrays.fields}


def missing_tree_branches(path: Path, tree_path: str, branches: Sequence[str]) -> list[str]:
    with uproot.open(path) as root_file:
        if tree_path not in root_file:
            return list(branches)
        tree = root_file[tree_path]
        available = set(tree.keys())
    return [branch for branch in branches if branch not in available]


def find_missing_tree_branches(files: Sequence[Path], tree_path: str, branches: Sequence[str]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for path in files:
        missing_branches = missing_tree_branches(path, tree_path, branches)
        if missing_branches:
            missing[str(path)] = missing_branches
    return missing


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def stable_data_hash(data: Any) -> str:
    payload = json.dumps(to_jsonable(data), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_snapshot(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def snapshot_input_files(files: Sequence[str | Path]) -> list[dict[str, Any]]:
    return [file_snapshot(path) for path in files]


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{column: to_jsonable(value) for column, value in row.items()} for row in df.to_dict(orient="records")]


def dataframe_to_root_arrays(df: pd.DataFrame) -> dict[str, Any]:
    arrays: dict[str, Any] = {}
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series):
            arrays[column] = series.astype(np.int8).to_numpy()
        elif pd.api.types.is_integer_dtype(series):
            arrays[column] = series.to_numpy()
        elif pd.api.types.is_float_dtype(series):
            arrays[column] = series.to_numpy(dtype=np.float64, copy=False)
        else:
            arrays[column] = ak.Array(series.fillna("").astype(str).tolist())
    return arrays


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_root_trees(path: Path, tree_frames: dict[str, pd.DataFrame]) -> None:
    ensure_dir(path.parent)
    with uproot.recreate(path) as root_file:
        for tree_name, frame in tree_frames.items():
            if frame.empty:
                continue
            root_file[tree_name] = dataframe_to_root_arrays(frame)
