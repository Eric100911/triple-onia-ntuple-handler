from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import awkward as ak

from .progress import wrap_iterable
from .schema import AnalysisModeSpec, get_analysis_mode_spec


def to_int_idx(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(round(float(value)))
    except Exception:
        return default


def label_value(mapping: Mapping[int, Any], value: Any) -> Any:
    return mapping.get(int(value), int(value))


def pdg_label(pdg_id: int, labels: Mapping[int, str]) -> str:
    return labels.get(int(pdg_id), str(int(pdg_id)))


def first_ancestor_idx(
    gen_pdg: list[int],
    gen_mother_idx: list[int],
    start_idx: Any,
    target_abs_pdg: int | tuple[int, ...] | list[int] | set[int],
) -> int:
    target_abs_pdgs = {abs(int(target_abs_pdg))} if isinstance(target_abs_pdg, int) else {abs(int(value)) for value in target_abs_pdg}
    idx = to_int_idx(start_idx, -1)
    seen: set[int] = set()
    while 0 <= idx < len(gen_pdg) and idx not in seen:
        seen.add(idx)
        if abs(int(gen_pdg[idx])) in target_abs_pdgs:
            return idx
        idx = to_int_idx(gen_mother_idx[idx], -1)
    return -1


def mother_chain(gen_rows: list[dict[str, Any]], start_idx: Any, max_depth: int = 20) -> str:
    idx = to_int_idx(start_idx, -1)
    chain: list[str] = []
    seen: set[int] = set()
    while 0 <= idx < len(gen_rows) and idx not in seen and len(chain) < max_depth:
        seen.add(idx)
        row = gen_rows[idx]
        chain.append(f"{idx}:{row['particle']}")
        idx = to_int_idx(row["mother_idx"], -1)
    return " <- ".join(chain) if chain else "unmatched"


def same_nonnegative_vertex(values: list[Any]) -> bool:
    vals = [to_int_idx(value, -1) for value in values]
    return bool(vals) and min(vals) >= 0 and len(set(vals)) == 1


def safe_rank_score(*probs: Any) -> float:
    for value in probs:
        if value is not None and float(value) >= 0.0:
            return float(value)
    return -1.0


def classify_failure_mode(row: Mapping[str, Any]) -> str:
    if row["truth_triple_strict"] == 1:
        return "truth_positive"
    if row["all_mu_gen_ok"] == 0 and row["all_k_gen_ok"] == 1:
        return "muon_side_gen_loss"
    if row["all_mu_gen_ok"] == 1 and row["all_k_gen_ok"] == 0:
        return "kaon_side_gen_loss"
    if row["all_mu_gen_ok"] == 0 and row["all_k_gen_ok"] == 0:
        return "muon_and_kaon_gen_loss"
    if row["jpsi1_pair_consistent"] == 0 or row["jpsi2_pair_consistent"] == 0:
        return "jpsi_pairing_failure"
    if row.get("ups_pair_consistent", 1) == 0:
        return "ups_pairing_failure"
    if row["phi_pair_consistent"] == 0:
        return "phi_pairing_failure"
    if row["two_jpsi_distinct"] == 0:
        return "two_jpsi_not_distinct"
    return "mixed_failure"


def _record_fields(arrays: Any) -> list[str]:
    if isinstance(arrays, Mapping):
        return list(arrays.keys())
    return list(arrays.fields)


def _record_field(arrays: Any, name: str) -> ak.Array:
    if isinstance(arrays, Mapping):
        return arrays[name]
    return arrays[name]


def _pythonize_event(arrays: Any, entry: int) -> dict[str, Any]:
    return {name: ak.to_list(_record_field(arrays, name)[entry]) for name in _record_fields(arrays)}


def _event_field_value(event: Mapping[str, Any], field: str, index: int, default: Any = None) -> Any:
    values = event.get(field)
    if values is None:
        return default
    if index < 0 or index >= len(values):
        return default
    return values[index]


def _mode_secondary_rank_values(event: Mapping[str, Any], cand_idx: int, mode_spec: AnalysisModeSpec) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in mode_spec.ranking_tiebreak_branches:
        if field in {"Phi_VtxProb", "Ups_VtxProb", "DiOnia_VtxProb"}:
            values.append(_event_field_value(event, field, cand_idx, None))
    return tuple(values)


def _daughter_alias(slot_key: str, muon_order_map: Mapping[str, int]) -> str:
    if slot_key in muon_order_map:
        return f"mu{muon_order_map[slot_key]}"
    if slot_key == "Phi_K_1":
        return "k1"
    if slot_key == "Phi_K_2":
        return "k2"
    raise KeyError(f"Unknown daughter slot key '{slot_key}'.")


def build_file_records(
    arrays: Any,
    source_file: str,
    analysis_mode: str,
    show_event_progress: bool = False,
    progress_backend: str = "auto",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    mode_spec = get_analysis_mode_spec(analysis_mode)
    muon_order_map = {slot_name: idx + 1 for idx, (slot_name, _) in enumerate(mode_spec.muon_slots)}

    n_entries = len(_record_field(arrays, "evtNum"))
    iterator = wrap_iterable(
        range(n_entries),
        enabled=show_event_progress,
        progress_backend=progress_backend,
        desc=f"Events in {source_file.rsplit('/', 1)[-1]}",
        leave=False,
    )

    for entry in iterator:
        event = _pythonize_event(arrays, entry)
        run = int(event["runNum"])
        lumi = int(event["lumiNum"])
        evt = int(event["evtNum"])

        gen_pdg = event["MC_GenPart_pdgId"]
        gen_mother_idx = event["MC_GenPart_motherGenIdx"]

        n_candidates = len(event[mode_spec.muon_slots[0][1]])
        event_truth_any = 0
        event_all6_any = 0

        for cand_idx in range(n_candidates):
            mu_slot_indices = {
                slot_name: to_int_idx(_event_field_value(event, index_branch, cand_idx, -1), -1)
                for slot_name, index_branch in mode_spec.muon_slots
            }
            mu_gen_indices = {
                slot_name: (
                    to_int_idx(event["muGenMatchIdx"][mu_idx], -1)
                    if 0 <= mu_idx < len(event["muGenMatchIdx"])
                    else -1
                )
                for slot_name, mu_idx in mu_slot_indices.items()
            }
            mu_gen_sources = {
                slot_name: (
                    to_int_idx(event["muGenMatchSource"][mu_idx], 0)
                    if 0 <= mu_idx < len(event["muGenMatchSource"])
                    else 0
                )
                for slot_name, mu_idx in mu_slot_indices.items()
            }
            mu_vertex_ids = {
                slot_name: (
                    to_int_idx(event["muVertexId"][mu_idx], -1)
                    if 0 <= mu_idx < len(event["muVertexId"])
                    else -1
                )
                for slot_name, mu_idx in mu_slot_indices.items()
            }
            mu_packed_methods = {
                slot_name: (
                    to_int_idx(event["muPackedMatchMethod"][mu_idx], 0)
                    if 0 <= mu_idx < len(event["muPackedMatchMethod"])
                    else 0
                )
                for slot_name, mu_idx in mu_slot_indices.items()
            }

            kaon_track_indices = {
                "Phi_K_1": to_int_idx(_event_field_value(event, "Phi_K_1_Idx", cand_idx, -1), -1),
                "Phi_K_2": to_int_idx(_event_field_value(event, "Phi_K_2_Idx", cand_idx, -1), -1),
            }
            kaon_vertex_ids = {
                "Phi_K_1": to_int_idx(_event_field_value(event, "Phi_K_1_vertexId", cand_idx, -1), -1),
                "Phi_K_2": to_int_idx(_event_field_value(event, "Phi_K_2_vertexId", cand_idx, -1), -1),
            }
            kaon_gen_indices = {
                "Phi_K_1": to_int_idx(_event_field_value(event, "Phi_K_1_genMatchIdx", cand_idx, -1), -1),
                "Phi_K_2": to_int_idx(_event_field_value(event, "Phi_K_2_genMatchIdx", cand_idx, -1), -1),
            }
            kaon_gen_sources = {
                "Phi_K_1": to_int_idx(_event_field_value(event, "Phi_K_1_genMatchSource", cand_idx, 0), 0),
                "Phi_K_2": to_int_idx(_event_field_value(event, "Phi_K_2_genMatchSource", cand_idx, 0), 0),
            }

            all_mu_gen_ok = int(all(idx >= 0 for idx in mu_gen_indices.values()))
            active_kaon_keys = ["Phi_K_1", "Phi_K_2"] if mode_spec.phi_objects else []
            all_k_gen_ok = int(all(kaon_gen_indices[key] >= 0 for key in active_kaon_keys)) if active_kaon_keys else 1
            truth_daughters_all_matched = int(all_mu_gen_ok and all_k_gen_ok)

            leg_pair_consistent: dict[str, int] = {}
            leg_mother_indices: dict[str, int] = {}
            for leg in mode_spec.truth_leg_specs:
                daughter_ancestors: list[int] = []
                for daughter_key in leg.daughter_keys:
                    if daughter_key.startswith("Phi_K_"):
                        gen_idx = kaon_gen_indices[daughter_key]
                    else:
                        gen_idx = mu_gen_indices[daughter_key]
                    daughter_ancestors.append(first_ancestor_idx(gen_pdg, gen_mother_idx, gen_idx, leg.ancestor_abs_pdgs))
                pair_consistent = int(bool(daughter_ancestors) and daughter_ancestors[0] >= 0 and all(idx == daughter_ancestors[0] for idx in daughter_ancestors))
                leg_pair_consistent[leg.object_name] = pair_consistent
                leg_mother_indices[leg.object_name] = daughter_ancestors[0] if pair_consistent else -1

            jpsi_mothers = [leg_mother_indices[name] for name in ("Jpsi_1", "Jpsi_2") if name in leg_mother_indices]
            two_jpsi_distinct = (
                int(len(jpsi_mothers) < 2 or (jpsi_mothers[0] >= 0 and jpsi_mothers[1] >= 0 and jpsi_mothers[0] != jpsi_mothers[1]))
            )
            distinct_truth_mothers = int(
                len(
                    {
                        leg_mother_indices[leg.object_name]
                        for leg in mode_spec.truth_leg_specs
                        if leg.requires_distinct_mother and leg_mother_indices[leg.object_name] >= 0
                    }
                )
                == len(
                    [
                        leg
                        for leg in mode_spec.truth_leg_specs
                        if leg.requires_distinct_mother and leg_mother_indices[leg.object_name] >= 0
                    ]
                )
            )
            truth_triple_strict = int(
                truth_daughters_all_matched
                and all(leg_pair_consistent.values())
                and distinct_truth_mothers
            )

            muons_same_recVtx = int(same_nonnegative_vertex([mu_vertex_ids[slot_name] for slot_name, _ in mode_spec.muon_slots]))
            same_vertex_values = [mu_vertex_ids[slot_name] for slot_name in mode_spec.same_vertex_muon_slots]
            same_vertex_values.extend(
                to_int_idx(_event_field_value(event, branch, cand_idx, -1), -1)
                for branch in mode_spec.same_vertex_track_branches
            )
            all6_same_recVtx = int(same_nonnegative_vertex(same_vertex_values))
            phi_same_recVtx_as_all_muons = int(
                bool(mode_spec.phi_objects)
                and muons_same_recVtx
                and to_int_idx(_event_field_value(event, "Phi_commonAssocPVPass", cand_idx, 0), 0) == 1
                and kaon_vertex_ids["Phi_K_1"] == next(iter(mu_vertex_ids.values()), -1) == kaon_vertex_ids["Phi_K_2"]
            )

            row = {
                "source_file": source_file,
                "analysis_mode": analysis_mode,
                "entry": int(entry),
                "run": run,
                "lumi": lumi,
                "event": evt,
                "cand_idx": int(cand_idx),
                "all_mu_gen_ok": all_mu_gen_ok,
                "all_k_gen_ok": all_k_gen_ok,
                "truth_daughters_all_matched": truth_daughters_all_matched,
                "jpsi1_pair_consistent": leg_pair_consistent.get("Jpsi_1", 1),
                "jpsi2_pair_consistent": leg_pair_consistent.get("Jpsi_2", 1),
                "ups_pair_consistent": leg_pair_consistent.get("Ups", 1),
                "phi_pair_consistent": leg_pair_consistent.get("Phi", 1),
                "two_jpsi_distinct": two_jpsi_distinct,
                "distinct_truth_mothers": distinct_truth_mothers,
                "truth_triple_strict": truth_triple_strict,
                "muons_same_recVtx": muons_same_recVtx,
                "all6_same_recVtx": all6_same_recVtx,
                "phi_same_recVtx_as_all_muons": phi_same_recVtx_as_all_muons,
                "Pri_fitValid": to_int_idx(_event_field_value(event, "Pri_fitValid", cand_idx, 0), 0),
                "Pri_fitPass": to_int_idx(_event_field_value(event, "Pri_fitPass", cand_idx, 0), 0),
                "Pri_assocPVPass": to_int_idx(_event_field_value(event, "Pri_assocPVPass", cand_idx, 0), 0),
                "Pri_trackPVPass": to_int_idx(_event_field_value(event, "Pri_trackPVPass", cand_idx, 0), 0),
                "Pri_passAny": to_int_idx(_event_field_value(event, "Pri_passAny", cand_idx, 0), 0),
                "Pri_VtxProb": float(_event_field_value(event, "Pri_VtxProb", cand_idx, -1.0)),
                "Pri_maxAbsDzPV": float(_event_field_value(event, "Pri_maxAbsDzPV", cand_idx, -1.0)),
                "Pri_maxAbsDxyPV": float(_event_field_value(event, "Pri_maxAbsDxyPV", cand_idx, -1.0)),
                "Phi_fitPass": to_int_idx(_event_field_value(event, "Phi_fitPass", cand_idx, 0), 0),
                "Phi_commonAssocPVPass": to_int_idx(_event_field_value(event, "Phi_commonAssocPVPass", cand_idx, 0), 0),
                "Phi_commonAssocPVIdx": to_int_idx(_event_field_value(event, "Phi_commonAssocPVIdx", cand_idx, -1), -1),
                "Phi_trackPVPass": to_int_idx(_event_field_value(event, "Phi_trackPVPass", cand_idx, 0), 0),
                "Phi_vertexCriteriaPass": to_int_idx(_event_field_value(event, "Phi_vertexCriteriaPass", cand_idx, 0), 0),
                "Phi_VtxProb": float(_event_field_value(event, "Phi_VtxProb", cand_idx, -1.0)),
                "Phi_maxAbsDzPV": float(_event_field_value(event, "Phi_maxAbsDzPV", cand_idx, -1.0)),
                "Phi_maxAbsDxyPV": float(_event_field_value(event, "Phi_maxAbsDxyPV", cand_idx, -1.0)),
                "DiOnia_fitValid": to_int_idx(_event_field_value(event, "DiOnia_fitValid", cand_idx, 0), 0),
                "DiOnia_fitPass": to_int_idx(_event_field_value(event, "DiOnia_fitPass", cand_idx, 0), 0),
                "DiOnia_commonRecVtxPass": to_int_idx(_event_field_value(event, "DiOnia_commonRecVtxPass", cand_idx, 0), 0),
                "DiOnia_passAny": to_int_idx(_event_field_value(event, "DiOnia_passAny", cand_idx, 0), 0),
                "DiOnia_VtxProb": float(_event_field_value(event, "DiOnia_VtxProb", cand_idx, -1.0)),
                "Jpsi_1_VtxProb": float(_event_field_value(event, "Jpsi_1_VtxProb", cand_idx, -1.0)),
                "Jpsi_2_VtxProb": float(_event_field_value(event, "Jpsi_2_VtxProb", cand_idx, -1.0)),
                "Ups_VtxProb": float(_event_field_value(event, "Ups_VtxProb", cand_idx, -1.0)),
                "all_mu_gen_source1": int(all(source == 1 for source in mu_gen_sources.values())),
                "any_mu_gen_source2": int(any(source == 2 for source in mu_gen_sources.values())),
                "all_kaons_gen_source1": int(all(kaon_gen_sources[key] == 1 for key in active_kaon_keys)) if active_kaon_keys else 1,
            }
            for alias_idx in range(1, 7):
                row[f"mu{alias_idx}_idx"] = -1
                row[f"mu{alias_idx}_vertex_id"] = -1
                row[f"mu{alias_idx}_gen_idx"] = -1
                row[f"mu{alias_idx}_gen_source"] = 0
                row[f"mu{alias_idx}_packed_method"] = 0
            for slot_name, _ in mode_spec.muon_slots:
                alias = _daughter_alias(slot_name, muon_order_map)
                row[f"{alias}_idx"] = mu_slot_indices[slot_name]
                row[f"{alias}_vertex_id"] = mu_vertex_ids[slot_name]
                row[f"{alias}_gen_idx"] = mu_gen_indices[slot_name]
                row[f"{alias}_gen_source"] = mu_gen_sources[slot_name]
                row[f"{alias}_packed_method"] = mu_packed_methods[slot_name]
            row["k1_track_idx"] = kaon_track_indices["Phi_K_1"] if mode_spec.phi_objects else -1
            row["k2_track_idx"] = kaon_track_indices["Phi_K_2"] if mode_spec.phi_objects else -1
            row["k1_vertex_id"] = kaon_vertex_ids["Phi_K_1"] if mode_spec.phi_objects else -1
            row["k2_vertex_id"] = kaon_vertex_ids["Phi_K_2"] if mode_spec.phi_objects else -1
            row["k1_gen_idx"] = kaon_gen_indices["Phi_K_1"] if mode_spec.phi_objects else -1
            row["k2_gen_idx"] = kaon_gen_indices["Phi_K_2"] if mode_spec.phi_objects else -1
            row["k1_gen_source"] = kaon_gen_sources["Phi_K_1"] if mode_spec.phi_objects else 0
            row["k2_gen_source"] = kaon_gen_sources["Phi_K_2"] if mode_spec.phi_objects else 0
            row["ranking_score"] = safe_rank_score(row["Pri_VtxProb"], *_mode_secondary_rank_values(event, cand_idx, mode_spec))
            row["pileup_like_proxy"] = int((truth_triple_strict == 0) and (all6_same_recVtx == 0))
            row["failure_mode"] = classify_failure_mode(row)
            candidate_rows.append(row)

            event_truth_any = max(event_truth_any, truth_triple_strict)
            event_all6_any = max(event_all6_any, truth_daughters_all_matched)

        event_rows.append(
            {
                "source_file": source_file,
                "entry": int(entry),
                "run": run,
                "lumi": lumi,
                "event": evt,
                "n_candidates": int(n_candidates),
                "has_candidate": int(n_candidates > 0),
                "has_truth_daughters_all_matched": int(event_all6_any),
                "has_truth_triple_strict": int(event_truth_any),
            }
        )

    return candidate_rows, event_rows


def add_classifier_columns(df):
    if df.empty:
        return df
    df = df.copy()
    df["clf_phi_and_pri_fit"] = ((df["Phi_vertexCriteriaPass"] == 1) & (df["Pri_fitPass"] == 1)).astype(int)
    df["clf_strict_hybrid"] = (
        (df["Phi_vertexCriteriaPass"] == 1)
        & (df["Pri_fitPass"] == 1)
        & (df["all6_same_recVtx"] == 1)
    ).astype(int)
    return df
