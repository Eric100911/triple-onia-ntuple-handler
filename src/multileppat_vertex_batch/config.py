from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StudyConfig:
    input_files: tuple[str, ...] = ()
    input_glob: str | None = None
    tree_path: str = "mkcands/X_data"
    config_tree_path: str = "mkcands/X_config"
    cache_dir: Path = Path(
        "/eos/user/c/chiw/JpsiJpsiPhi/MC_samples/Ntuple_refactor/TPS-JpsiJpsiPhi/_vertex_truth_cache"
    )
    use_cache: bool = True
    overwrite_cache: bool = False
    show_file_progress: bool = True
    show_event_progress: bool = False
    progress_backend: str = "notebook"
    phi_vtxprob_scan: tuple[float, ...] = (
        0.0,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
    )


@dataclass(frozen=True)
class OfflineSelectionConfig:
    mu_pt_barrel_min: float = 3.5
    mu_pt_endcap_min: float = 2.5
    mu_abs_eta_max: float = 2.4
    track_pt_min: float = 2.0
    track_abs_eta_max: float = 2.5
    jpsi_mass_window: tuple[float, float] = (2.9, 3.3)
    jpsi_pt_min: float = 6.0
    jpsi_abs_y_max: float = 2.5
    jpsi_vtxprob_min: float = 0.01
    ups_mass_window: tuple[float, float] = (8.0, 12.0)
    ups_pt_min: float = 6.0
    ups_abs_y_max: float = 2.5
    ups_vtxprob_min: float | None = None
    phi_mass_window: tuple[float, float] = (0.99, 1.07)
    phi_pt_min: float = 4.0
    phi_vtxprob_min: float = 0.01


@dataclass(frozen=True)
class MassStudyConfig:
    analysis_mode: str
    active_windows: dict[str, tuple[float, float]]
    selector_name: str = "all6_same_recVtx"
    selectors: tuple[str, ...] = ("all6_same_recVtx", "Pri_fitValid")
    best_candidate_metric: str = "triple_pt2_sum"
    fit_branches: tuple[str, ...] = ("Jpsi_1_mass", "Jpsi_2_mass", "Phi_mass")


@dataclass(frozen=True)
class CmsPlotStyleConfig:
    caption: str | None = None
    energy_tev: float = 13.6
    lumi_fb: float | None = None
    era: str | None = None
    is_data: bool = True


def resolve_windows(
    defaults: dict[str, tuple[float, float]],
    overrides: dict[str, tuple[float, float] | None] | None,
) -> dict[str, tuple[float, float]]:
    active: dict[str, tuple[float, float]] = {}
    overrides = overrides or {}
    for key, default_window in defaults.items():
        override = overrides.get(key)
        active[key] = default_window if override is None else tuple(map(float, override))
    return active


def default_mass_windows_from_config_row(config_row: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {
        "Jpsi_1_mass": (float(config_row["JpsiMassMin"]), float(config_row["JpsiMassMax"])),
        "Jpsi_2_mass": (float(config_row["JpsiMassMin"]), float(config_row["JpsiMassMax"])),
        "Ups_mass": (float(config_row["UpsMassMin"]), float(config_row["UpsMassMax"])),
        "Phi_mass": (float(config_row["PhiMassMin"]), float(config_row["PhiMassMax"])),
        "Pri_mass": (0.0, 100.0),
    }
