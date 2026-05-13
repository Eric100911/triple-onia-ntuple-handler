from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from .config import OfflineSelectionConfig, StudyConfig
from .io import read_ntuple_arrays
from .kinematics import eta_from_pxyz, flatten_candidate_field, in_window, muon_pass_mask, rapidity_from_pxyzm
from .progress import wrap_iterable
from .schema import (
    FIT_INPUT_CANDIDATE_BRANCHES,
    FIT_INPUT_MUON_VECTOR_BRANCH_OUTPUTS,
    MASS_STUDY_BRANCHES,
    AnalysisModeSpec,
    get_analysis_mode_spec,
)


SELECTOR_LABELS = {
    "all6_same_recVtx": "all6_same_recVtx",
    "Pri_fitValid": "Pri_fitValid",
    "Pri_fitPass": "Pri_fitPass",
    "Pri_passAny": "Pri_passAny",
}

SELECTION_SUMMARY_COLUMNS = [
    "selector",
    "n_total_candidates",
    "n_candidates_in_active_windows",
    "n_candidates_passing_selector",
    "n_candidates_passing_selector_in_active_windows",
    "n_candidates_after_full_offline",
    "n_events_with_candidate",
    "n_events_with_multiple_candidates",
    "n_selected_best_candidates",
]


def _mode_candidate_branches(mode_spec: AnalysisModeSpec) -> list[str]:
    include_phi = bool(mode_spec.phi_objects)
    include_jpsi2 = "Jpsi_2" in mode_spec.jpsi_objects
    include_ups = bool(mode_spec.ups_objects)

    def keep(branch: str) -> bool:
        if branch.startswith("Jpsi_1_"):
            return True
        if branch.startswith("Jpsi_2_"):
            return include_jpsi2
        if branch.startswith("Ups_"):
            return include_ups
        if branch.startswith("Phi_"):
            return include_phi
        if branch.startswith("Pri_") or branch.startswith("DiOnia_"):
            return True
        return False

    return [branch for branch in FIT_INPUT_CANDIDATE_BRANCHES if keep(branch)]


def _output_index_branches(mode_spec: AnalysisModeSpec) -> list[str]:
    index_branches = [index_branch for _, index_branch in mode_spec.muon_slots]
    index_branches.extend(
        branch.replace("vertexId", "Idx")
        for branch in mode_spec.same_vertex_track_branches
        if branch.endswith("vertexId")
    )
    return index_branches


def _muon_output_columns(mode_spec: AnalysisModeSpec) -> list[str]:
    columns: list[str] = []
    for slot_name, _ in mode_spec.muon_slots:
        columns.append(f"{slot_name}_pt")
        columns.append(f"{slot_name}_eta")
        for suffix in FIT_INPUT_MUON_VECTOR_BRANCH_OUTPUTS.values():
            columns.append(f"{slot_name}_{suffix}")
    return columns


def _candidate_pool_columns(mode_spec: AnalysisModeSpec) -> list[str]:
    rapidity_columns = [f"{name}_y" for name in mode_spec.rapidity_objects]
    return [
        "selector",
        "source_file",
        "entry",
        "run",
        "lumi",
        "event",
        "cand_idx",
        "n_passing_candidates",
        "triple_pt2_sum",
        *rapidity_columns,
        *_output_index_branches(mode_spec),
        *_mode_candidate_branches(mode_spec),
        *_muon_output_columns(mode_spec),
        "all6_same_recVtx",
    ]


def _active_mass_mask(arrays, active_windows: dict[str, tuple[float, float]], mode_spec: AnalysisModeSpec):
    mask = in_window(arrays[mode_spec.fit_branches[0]], active_windows[mode_spec.fit_branches[0]])
    for branch in mode_spec.fit_branches[1:]:
        mask = mask & in_window(arrays[branch], active_windows[branch])
    if "Pri_mass" in active_windows and "Pri_mass" in arrays.fields:
        mask = mask & in_window(arrays["Pri_mass"], active_windows["Pri_mass"])
    return mask


def _selector_masks(arrays, all6_same_recVtx):
    return {
        "all6_same_recVtx": all6_same_recVtx,
        "Pri_fitValid": arrays["Pri_fitValid"] == 1,
        "Pri_fitPass": arrays["Pri_fitPass"] == 1,
        "Pri_passAny": arrays["Pri_passAny"] == 1,
    }


def _flatten_branch(arrays, branch: str, mask) -> np.ndarray:
    return flatten_candidate_field(arrays[branch][mask])


def _muon_branch_default(arrays, branch: str):
    type_text = str(ak.type(arrays[branch])).lower()
    if "float" in type_text:
        return np.nan
    if "bool" in type_text:
        return False
    return -1


def _candidate_muon_values(arrays, branch: str, safe_idx, default=None):
    default = _muon_branch_default(arrays, branch) if default is None else default
    branch_array = arrays[branch]
    branch_counts = ak.num(branch_array, axis=1)
    branch_count_values = np.asarray(ak.to_numpy(branch_counts), dtype=np.int64)
    flat_idx = ak.flatten(safe_idx, axis=None)
    max_idx = int(ak.max(flat_idx)) if len(flat_idx) else -1
    branch_count_max = int(branch_count_values.max()) if branch_count_values.size else 0
    target = max(branch_count_max, max_idx + 1)
    padded = ak.pad_none(branch_array, target, axis=1, clip=False)
    return ak.fill_none(padded[safe_idx], default)


def _flatten_muon_branch(arrays, branch: str, safe_idx, mask) -> np.ndarray:
    return flatten_candidate_field(_candidate_muon_values(arrays, branch, safe_idx)[mask])


def _slot_object_name(slot_name: str) -> str:
    return slot_name.split("_mu_")[0]


def _object_slot_groups(mode_spec: AnalysisModeSpec) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for slot_name, _ in mode_spec.muon_slots:
        grouped[_slot_object_name(slot_name)].append(slot_name)
    return dict(grouped)


def build_candidate_pool_for_file(
    path: str | Path,
    config: StudyConfig,
    active_windows: dict[str, tuple[float, float]],
    selection_cfg: OfflineSelectionConfig,
    analysis_mode: str,
    selectors: tuple[str, ...] = ("all6_same_recVtx", "Pri_fitValid"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode_spec = get_analysis_mode_spec(analysis_mode)
    candidate_columns = _candidate_pool_columns(mode_spec)
    candidate_branches = _mode_candidate_branches(mode_spec)
    arrays = read_ntuple_arrays(Path(path), config, MASS_STUDY_BRANCHES)

    cand_ref = arrays[mode_spec.fit_branches[0]]
    cand_idx = ak.local_index(cand_ref)
    entry_bc = ak.broadcast_arrays(ak.Array(np.arange(len(cand_ref), dtype=np.int64)), cand_ref)[0]
    run_bc = ak.broadcast_arrays(arrays["runNum"], cand_ref)[0]
    lumi_bc = ak.broadcast_arrays(arrays["lumiNum"], cand_ref)[0]
    evt_bc = ak.broadcast_arrays(arrays["evtNum"], cand_ref)[0]

    total_input_candidates = int(ak.sum(ak.num(cand_ref, axis=1)))
    n_mu = ak.num(arrays["muPx"], axis=1)

    raw_mu_slot_indices: dict[str, ak.Array] = {}
    valid_slot_mask = ak.ones_like(cand_ref, dtype=bool)
    for slot_name, index_branch in mode_spec.muon_slots:
        slot_indices = ak.values_astype(arrays[index_branch], np.int64)
        raw_mu_slot_indices[slot_name] = slot_indices
        valid_slot_mask = valid_slot_mask & (slot_indices >= 0) & (slot_indices < n_mu[:, None])

    mu_slot_indices = {
        slot_name: ak.where(valid_slot_mask, slot_indices, 0)
        for slot_name, slot_indices in raw_mu_slot_indices.items()
    }

    mu_slot_values: dict[str, dict[str, ak.Array]] = {}
    mu_slot_derived: dict[str, dict[str, ak.Array]] = {}
    for slot_name, safe_idx in mu_slot_indices.items():
        px = _candidate_muon_values(arrays, "muPx", safe_idx, 0.0)
        py = _candidate_muon_values(arrays, "muPy", safe_idx, 0.0)
        pz = _candidate_muon_values(arrays, "muPz", safe_idx, 0.0)
        charge = _candidate_muon_values(arrays, "muCharge", safe_idx, 0)
        soft = _candidate_muon_values(arrays, "muIsPatSoftMuon", safe_idx, 0)
        vertex_id = _candidate_muon_values(arrays, "muVertexId", safe_idx, -1)
        mu_slot_values[slot_name] = {
            "px": px,
            "py": py,
            "pz": pz,
            "charge": charge,
            "soft": soft,
            "vertexId": vertex_id,
        }
        mu_slot_derived[slot_name] = {
            "pt": np.hypot(px, py),
            "eta": eta_from_pxyz(px, py, pz),
        }

    all6_same_recVtx = valid_slot_mask
    reference_vertex = mu_slot_values[mode_spec.same_vertex_muon_slots[0]]["vertexId"]
    all6_same_recVtx = all6_same_recVtx & (reference_vertex >= 0)
    for slot_name in mode_spec.same_vertex_muon_slots[1:]:
        all6_same_recVtx = all6_same_recVtx & (mu_slot_values[slot_name]["vertexId"] == reference_vertex)
    for branch in mode_spec.same_vertex_track_branches:
        all6_same_recVtx = all6_same_recVtx & (arrays[branch] == reference_vertex)

    mass_window_mask = _active_mass_mask(arrays, active_windows, mode_spec)

    all_mu_pass = valid_slot_mask
    for slot_name, _ in mode_spec.muon_slots:
        all_mu_pass = all_mu_pass & muon_pass_mask(
            mu_slot_derived[slot_name]["pt"],
            mu_slot_derived[slot_name]["eta"],
            mu_slot_values[slot_name]["soft"],
            selection_cfg,
        )

    mu_os = valid_slot_mask
    for object_name, slot_names in _object_slot_groups(mode_spec).items():
        if len(slot_names) != 2:
            raise RuntimeError(f"Expected exactly two muon slots for object '{object_name}', found {slot_names}.")
        mu_os = mu_os & (
            (mu_slot_values[slot_names[0]]["charge"] + mu_slot_values[slot_names[1]]["charge"]) == 0
        )

    rapidity_by_object = {
        object_name: rapidity_from_pxyzm(
            arrays[f"{object_name}_px"],
            arrays[f"{object_name}_py"],
            arrays[f"{object_name}_pz"],
            arrays[f"{object_name}_mass"],
        )
        for object_name in mode_spec.rapidity_objects
    }

    jpsi_pass = ak.ones_like(cand_ref, dtype=bool)
    for object_name in mode_spec.jpsi_objects:
        jpsi_pass = jpsi_pass & (
            in_window(arrays[f"{object_name}_mass"], active_windows[f"{object_name}_mass"])
            & (arrays[f"{object_name}_pt"] > selection_cfg.jpsi_pt_min)
            & (np.abs(rapidity_by_object[object_name]) < selection_cfg.jpsi_abs_y_max)
            & (arrays[f"{object_name}_VtxProb"] > selection_cfg.jpsi_vtxprob_min)
        )

    resolved_ups_vtxprob_min = 0.0 if selection_cfg.ups_vtxprob_min is None else float(selection_cfg.ups_vtxprob_min)
    ups_pass = ak.ones_like(cand_ref, dtype=bool)
    for object_name in mode_spec.ups_objects:
        ups_pass = ups_pass & (
            in_window(arrays[f"{object_name}_mass"], active_windows[f"{object_name}_mass"])
            & (arrays[f"{object_name}_pt"] > selection_cfg.ups_pt_min)
            & (np.abs(rapidity_by_object[object_name]) < selection_cfg.ups_abs_y_max)
            & (arrays[f"{object_name}_VtxProb"] > resolved_ups_vtxprob_min)
        )

    if mode_spec.phi_objects:
        track_pass = (
            (arrays["Phi_K_1_pt"] > selection_cfg.track_pt_min)
            & (arrays["Phi_K_2_pt"] > selection_cfg.track_pt_min)
            & (np.abs(arrays["Phi_K_1_eta"]) < selection_cfg.track_abs_eta_max)
            & (np.abs(arrays["Phi_K_2_eta"]) < selection_cfg.track_abs_eta_max)
        )
        phi_pass = (
            in_window(arrays["Phi_mass"], active_windows["Phi_mass"])
            & (arrays["Phi_pt"] > selection_cfg.phi_pt_min)
            & (arrays["Phi_VtxProb"] > selection_cfg.phi_vtxprob_min)
        )
    else:
        track_pass = ak.ones_like(cand_ref, dtype=bool)
        phi_pass = ak.ones_like(cand_ref, dtype=bool)

    offline_mask = valid_slot_mask & all_mu_pass & mu_os & jpsi_pass & ups_pass & track_pass & phi_pass

    triple_pt2_sum = ak.zeros_like(arrays[mode_spec.ranking_pt_branches[0]], dtype=np.float64)
    for branch in mode_spec.ranking_pt_branches:
        triple_pt2_sum = triple_pt2_sum + arrays[branch] ** 2

    selector_masks = _selector_masks(arrays, all6_same_recVtx)
    audit_rows = [
        {"source_file": str(path), "stage": "initial_candidates", "selector": "all", "count": total_input_candidates},
        {
            "source_file": str(path),
            "stage": "initial_candidates_in_active_windows",
            "selector": "all",
            "count": int(ak.sum(mass_window_mask)),
        },
    ]

    selector_frames: list[pd.DataFrame] = []
    for selector in selectors:
        selector_mask = selector_masks[selector]
        final_mask = selector_mask & offline_mask
        audit_rows.extend(
            [
                {
                    "source_file": str(path),
                    "stage": "selector_candidates",
                    "selector": selector,
                    "count": int(ak.sum(selector_mask)),
                },
                {
                    "source_file": str(path),
                    "stage": "selector_candidates_in_active_windows",
                    "selector": selector,
                    "count": int(ak.sum(selector_mask & mass_window_mask)),
                },
                {
                    "source_file": str(path),
                    "stage": "selector_candidates_after_full_offline",
                    "selector": selector,
                    "count": int(ak.sum(final_mask)),
                },
            ]
        )

        if int(ak.sum(final_mask)) == 0:
            continue

        row_count = int(ak.sum(final_mask))
        n_passing_candidates = ak.sum(final_mask, axis=1)
        row_data: dict[str, np.ndarray] = {
            "selector": np.full(row_count, selector, dtype=object),
            "source_file": np.full(row_count, str(path), dtype=object),
            "entry": flatten_candidate_field(entry_bc[final_mask]).astype(np.int64),
            "run": flatten_candidate_field(run_bc[final_mask]).astype(np.int64),
            "lumi": flatten_candidate_field(lumi_bc[final_mask]).astype(np.int64),
            "event": flatten_candidate_field(evt_bc[final_mask]).astype(np.int64),
            "cand_idx": flatten_candidate_field(cand_idx[final_mask]).astype(np.int64),
            "n_passing_candidates": flatten_candidate_field(
                ak.broadcast_arrays(n_passing_candidates, cand_ref)[0][final_mask]
            ).astype(np.int64),
            "triple_pt2_sum": flatten_candidate_field(triple_pt2_sum[final_mask]),
            "all6_same_recVtx": flatten_candidate_field(all6_same_recVtx[final_mask]).astype(np.int64),
        }
        for object_name in mode_spec.rapidity_objects:
            row_data[f"{object_name}_y"] = flatten_candidate_field(rapidity_by_object[object_name][final_mask])
        for _, index_branch in mode_spec.muon_slots:
            row_data[index_branch] = flatten_candidate_field(arrays[index_branch][final_mask]).astype(np.int64)
        for branch in mode_spec.same_vertex_track_branches:
            idx_branch = branch.replace("vertexId", "Idx")
            row_data[idx_branch] = flatten_candidate_field(arrays[idx_branch][final_mask]).astype(np.int64)
        for branch in candidate_branches:
            row_data[branch] = _flatten_branch(arrays, branch, final_mask)
        for slot_name, _ in mode_spec.muon_slots:
            row_data[f"{slot_name}_pt"] = flatten_candidate_field(mu_slot_derived[slot_name]["pt"][final_mask])
            row_data[f"{slot_name}_eta"] = flatten_candidate_field(mu_slot_derived[slot_name]["eta"][final_mask])
            safe_idx = mu_slot_indices[slot_name]
            for source_branch, suffix in FIT_INPUT_MUON_VECTOR_BRANCH_OUTPUTS.items():
                row_data[f"{slot_name}_{suffix}"] = _flatten_muon_branch(arrays, source_branch, safe_idx, final_mask)
        frame = pd.DataFrame(row_data, columns=candidate_columns)
        selector_frames.append(frame)

    candidate_pool_df = (
        pd.concat(selector_frames, ignore_index=True)
        if selector_frames
        else pd.DataFrame(columns=candidate_columns)
    )
    audit_df = pd.DataFrame(audit_rows)
    return candidate_pool_df, audit_df


def select_best_candidates(candidate_pool_df: pd.DataFrame, mode_spec: AnalysisModeSpec) -> pd.DataFrame:
    if candidate_pool_df.empty:
        return candidate_pool_df.copy()

    event_key_cols = ["selector", "source_file", "run", "lumi", "event"]
    tie_break_cols = [column for column in mode_spec.ranking_tiebreak_branches if column in candidate_pool_df.columns]
    sort_columns = event_key_cols + ["triple_pt2_sum", *tie_break_cols]
    ascending = [True, True, True, True, True, False] + [False] * max(len(tie_break_cols) - 1, 0)
    if tie_break_cols:
        ascending.append(True if tie_break_cols[-1] == "cand_idx" else False)
    selected_candidate_df = (
        candidate_pool_df
        .sort_values(
            sort_columns,
            ascending=ascending,
        )
        .drop_duplicates(event_key_cols, keep="first")
        .sort_values(event_key_cols + ["cand_idx"])
        .reset_index(drop=True)
    )

    multiplicity = candidate_pool_df.groupby(event_key_cols).size().rename("n_passing_candidates").reset_index()
    selected_candidate_df = selected_candidate_df.drop(columns=["n_passing_candidates"], errors="ignore").merge(
        multiplicity,
        on=event_key_cols,
        how="left",
    )
    return selected_candidate_df


def summarize_mass_window_flow(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame(columns=["selector", "stage", "count"])

    summary = (
        audit_df.groupby(["selector", "stage"], dropna=False)["count"]
        .sum()
        .reset_index()
        .sort_values(["selector", "stage"])
        .reset_index(drop=True)
    )
    return summary


def summarize_selection(candidate_pool_df: pd.DataFrame, selected_candidate_df: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame(columns=SELECTION_SUMMARY_COLUMNS)

    rows = []
    selectors = sorted(audit_df.loc[audit_df["selector"] != "all", "selector"].unique().tolist())
    for selector in selectors:
        subset = candidate_pool_df.loc[candidate_pool_df["selector"] == selector]
        selected = selected_candidate_df.loc[selected_candidate_df["selector"] == selector]
        multiplicity = (
            subset.groupby(["selector", "source_file", "run", "lumi", "event"]).size()
            if not subset.empty
            else pd.Series(dtype=int)
        )
        rows.append(
            {
                "selector": selector,
                "n_total_candidates": int(audit_df.loc[(audit_df["selector"] == "all") & (audit_df["stage"] == "initial_candidates"), "count"].sum()),
                "n_candidates_in_active_windows": int(audit_df.loc[(audit_df["selector"] == "all") & (audit_df["stage"] == "initial_candidates_in_active_windows"), "count"].sum()),
                "n_candidates_passing_selector": int(audit_df.loc[(audit_df["selector"] == selector) & (audit_df["stage"] == "selector_candidates"), "count"].sum()),
                "n_candidates_passing_selector_in_active_windows": int(
                    audit_df.loc[
                        (audit_df["selector"] == selector) & (audit_df["stage"] == "selector_candidates_in_active_windows"),
                        "count",
                    ].sum()
                ),
                "n_candidates_after_full_offline": int(audit_df.loc[(audit_df["selector"] == selector) & (audit_df["stage"] == "selector_candidates_after_full_offline"), "count"].sum()),
                "n_events_with_candidate": int(selected.shape[0]),
                "n_events_with_multiple_candidates": int((multiplicity > 1).sum()) if not multiplicity.empty else 0,
                "n_selected_best_candidates": int(selected.shape[0]),
            }
        )
    return pd.DataFrame(rows, columns=SELECTION_SUMMARY_COLUMNS)


def build_candidate_pool_batch(
    files: list[str | Path],
    config: StudyConfig,
    active_windows: dict[str, tuple[float, float]],
    selection_cfg: OfflineSelectionConfig,
    analysis_mode: str,
    selectors: tuple[str, ...] = ("all6_same_recVtx", "Pri_fitValid"),
    show_progress: bool | None = None,
) -> dict[str, pd.DataFrame]:
    return run_mass_selection_batch(
        files=files,
        config=config,
        active_windows=active_windows,
        selection_cfg=selection_cfg,
        analysis_mode=analysis_mode,
        selectors=selectors,
        show_progress=show_progress,
    )


def run_mass_selection_batch(
    files: list[str | Path],
    config: StudyConfig,
    active_windows: dict[str, tuple[float, float]],
    selection_cfg: OfflineSelectionConfig,
    analysis_mode: str,
    selectors: tuple[str, ...] = ("all6_same_recVtx", "Pri_fitValid"),
    show_progress: bool | None = None,
) -> dict[str, pd.DataFrame]:
    mode_spec = get_analysis_mode_spec(analysis_mode)
    candidate_columns = _candidate_pool_columns(mode_spec)
    candidate_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []

    iterator = wrap_iterable(
        files,
        enabled=config.show_file_progress if show_progress is None else show_progress,
        progress_backend=config.progress_backend,
        desc="Selection files",
    )
    for path in iterator:
        candidate_df, audit_df = build_candidate_pool_for_file(
            path=path,
            config=config,
            active_windows=active_windows,
            selection_cfg=selection_cfg,
            analysis_mode=analysis_mode,
            selectors=selectors,
        )
        if not candidate_df.empty:
            candidate_parts.append(candidate_df)
        audit_parts.append(audit_df)

    candidate_pool_df = (
        pd.concat(candidate_parts, ignore_index=True)
        if candidate_parts
        else pd.DataFrame(columns=candidate_columns)
    )
    audit_df = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    window_audit_df = summarize_mass_window_flow(audit_df)
    selected_candidate_df = select_best_candidates(candidate_pool_df, mode_spec)
    selection_summary_df = summarize_selection(candidate_pool_df, selected_candidate_df, audit_df)
    return {
        "candidate_pool_df": candidate_pool_df,
        "selected_candidate_df": selected_candidate_df,
        "audit_df": audit_df,
        "window_audit_df": window_audit_df,
        "selection_summary_df": selection_summary_df,
    }
