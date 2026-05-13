from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import StudyConfig
from .io import read_ntuple_entry
from .schema import DETAIL_DATA_BRANCHES, KAON_GEN_SOURCE_LABELS, MU_GEN_SOURCE_LABELS, MU_PACKED_METHOD_LABELS, PDG_LABELS
from .truth import first_ancestor_idx, label_value, mother_chain, pdg_label


def first_row(df: pd.DataFrame, mask: pd.Series) -> dict[str, Any] | None:
    subset = df.loc[mask]
    return subset.iloc[0].to_dict() if not subset.empty else None


def first_sorted_row(df: pd.DataFrame, mask: pd.Series, sort_by: str, ascending: bool = True) -> dict[str, Any] | None:
    subset = df.loc[mask].sort_values(sort_by, ascending=ascending)
    return subset.iloc[0].to_dict() if not subset.empty else None


def select_representatives(df: pd.DataFrame) -> dict[str, dict[str, Any] | None]:
    return {
        "truth_positive": first_row(df, df["truth_triple_strict"] == 1),
        "phi_vertex_false_positive": first_row(
            df, (df["Phi_vertexCriteriaPass"] == 1) & (df["truth_triple_strict"] == 0)
        ),
        "pri_passAny_false_positive": first_row(
            df, (df["Pri_passAny"] == 1) & (df["truth_triple_strict"] == 0)
        ),
        "strict_hybrid_false_negative": first_row(
            df, (df["clf_strict_hybrid"] == 0) & (df["truth_triple_strict"] == 1)
        ),
    }


def safe_eta(px: float, py: float, pz: float) -> float:
    p = math.sqrt(px * px + py * py + pz * pz)
    denom = p - pz
    if denom == 0.0:
        return float("inf") if pz >= 0.0 else float("-inf")
    return 0.5 * math.log((p + pz) / denom)


def build_gen_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pdg_ids = event["MC_GenPart_pdgId"]
    mother_idx = event["MC_GenPart_motherGenIdx"]
    mother_pdg = event["MC_GenPart_motherPdgId"]
    pt = event["MC_GenPart_pt"]
    eta = event["MC_GenPart_eta"]
    phi = event["MC_GenPart_phi"]
    mass = event["MC_GenPart_mass"]
    status = event["MC_GenPart_status"]
    handle_idx = event["MC_GenPart_handleIndex"]

    for idx, pdg_id in enumerate(pdg_ids):
        rows.append(
            {
                "gen_idx": idx,
                "handle_index": int(handle_idx[idx]) if idx < len(handle_idx) else -1,
                "pdg_id": int(pdg_id),
                "particle": pdg_label(pdg_id, PDG_LABELS),
                "status": int(status[idx]) if idx < len(status) else -1,
                "mother_idx": int(mother_idx[idx]) if idx < len(mother_idx) else -1,
                "mother_pdg_id": int(mother_pdg[idx]) if idx < len(mother_pdg) else 0,
                "mother_particle": pdg_label(mother_pdg[idx], PDG_LABELS) if idx < len(mother_pdg) else "unknown",
                "pt": float(pt[idx]) if idx < len(pt) else None,
                "eta": float(eta[idx]) if idx < len(eta) else None,
                "phi": float(phi[idx]) if idx < len(phi) else None,
                "mass": float(mass[idx]) if idx < len(mass) else None,
            }
        )
    return rows


def _event_arrays_for_detail(path: Path, config: StudyConfig, entry: int) -> dict[str, Any]:
    return read_ntuple_entry(path, config, DETAIL_DATA_BRANCHES, entry)


def candidate_detail_tables(candidate: dict[str, Any], config: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_file = Path(candidate["source_file"])
    entry = int(candidate["entry"])
    event = _event_arrays_for_detail(source_file, config, entry)
    gen_rows = build_gen_rows(event)
    gen_pdg = event["MC_GenPart_pdgId"]
    gen_mother_idx = event["MC_GenPart_motherGenIdx"]

    header = pd.DataFrame(
        [
            {
                "source_file": candidate["source_file"],
                "entry": candidate["entry"],
                "cand_idx": candidate["cand_idx"],
                "run": candidate["run"],
                "lumi": candidate["lumi"],
                "event": candidate["event"],
                "failure_mode": candidate["failure_mode"],
                "truth_triple_strict": candidate["truth_triple_strict"],
                "Pri_passAny": candidate["Pri_passAny"],
                "Phi_vertexCriteriaPass": candidate["Phi_vertexCriteriaPass"],
                "all6_same_recVtx": candidate["all6_same_recVtx"],
                "ranking_score": candidate["ranking_score"],
            }
        ]
    )

    mu_rows: list[dict[str, Any]] = []
    focus_gen_indices: set[int] = set()
    mu_indices = [candidate["mu1_idx"], candidate["mu2_idx"], candidate["mu3_idx"], candidate["mu4_idx"]]
    for slot, mu_idx in enumerate(mu_indices, start=1):
        match_idx = int(event["muGenMatchIdx"][mu_idx]) if 0 <= mu_idx < len(event["muGenMatchIdx"]) else -1
        match_source = int(event["muGenMatchSource"][mu_idx]) if 0 <= mu_idx < len(event["muGenMatchSource"]) else 0
        packed_method = int(event["muPackedMatchMethod"][mu_idx]) if 0 <= mu_idx < len(event["muPackedMatchMethod"]) else 0
        matched_gen = gen_rows[match_idx] if 0 <= match_idx < len(gen_rows) else None
        jpsi_ancestor_idx = first_ancestor_idx(gen_pdg, gen_mother_idx, match_idx, 443) if match_idx >= 0 else -1
        if match_idx >= 0:
            focus_gen_indices.add(match_idx)
        if jpsi_ancestor_idx >= 0:
            focus_gen_indices.add(jpsi_ancestor_idx)

        px = float(event["muPx"][mu_idx]) if 0 <= mu_idx < len(event["muPx"]) else None
        py = float(event["muPy"][mu_idx]) if 0 <= mu_idx < len(event["muPy"]) else None
        pz = float(event["muPz"][mu_idx]) if 0 <= mu_idx < len(event["muPz"]) else None
        mu_rows.append(
            {
                "slot": f"mu{slot}",
                "mu_idx": mu_idx,
                "pt": math.hypot(px, py) if px is not None and py is not None else None,
                "eta": safe_eta(px, py, pz) if px is not None and py is not None and pz is not None else None,
                "phi": math.atan2(py, px) if px is not None and py is not None else None,
                "vertexId": int(event["muVertexId"][mu_idx]) if 0 <= mu_idx < len(event["muVertexId"]) else -1,
                "fromPV": int(event["muFromPV"][mu_idx]) if 0 <= mu_idx < len(event["muFromPV"]) else -1,
                "pvAssocQuality": int(event["muPVAssocQuality"][mu_idx]) if 0 <= mu_idx < len(event["muPVAssocQuality"]) else -1,
                "packedMatchIdx": int(event["muPackedMatchIdx"][mu_idx]) if 0 <= mu_idx < len(event["muPackedMatchIdx"]) else -1,
                "packedMethod": label_value(MU_PACKED_METHOD_LABELS, packed_method),
                "packedVectorRelP": float(event["muPackedMatchVectorRelP"][mu_idx])
                if 0 <= mu_idx < len(event["muPackedMatchVectorRelP"])
                else None,
                "packedChi2": float(event["muPackedMatchChi2"][mu_idx]) if 0 <= mu_idx < len(event["muPackedMatchChi2"]) else None,
                "packedDzPV": float(event["muPackedMatchDzPV"][mu_idx]) if 0 <= mu_idx < len(event["muPackedMatchDzPV"]) else None,
                "packedDzAssocPV": float(event["muPackedMatchDzAssocPV"][mu_idx])
                if 0 <= mu_idx < len(event["muPackedMatchDzAssocPV"])
                else None,
                "genMatchIdx": match_idx,
                "genSource": label_value(MU_GEN_SOURCE_LABELS, match_source),
                "matchedParticle": matched_gen["particle"] if matched_gen else "unmatched",
                "matchedMother": matched_gen["mother_particle"] if matched_gen else "unmatched",
                "jpsiAncestorIdx": jpsi_ancestor_idx,
                "motherChain": mother_chain(gen_rows, match_idx),
            }
        )

    kaon_rows: list[dict[str, Any]] = []
    kaon_specs = [
        (
            "K1",
            int(candidate["cand_idx"]),
            int(candidate["k1_track_idx"]),
            "Phi_K_1_pt",
            "Phi_K_1_eta",
            "Phi_K_1_phi",
            "Phi_K_1_vertexId",
            "Phi_K_1_fromPV",
            "Phi_K_1_pvAssocQuality",
            "Phi_K_1_dzPV",
            "Phi_K_1_dxyPV",
            "Phi_K_1_dzAssocPV",
            "Phi_K_1_dxyAssocPV",
            "Phi_K_1_genMatchIdx",
            "Phi_K_1_genMatchSource",
        ),
        (
            "K2",
            int(candidate["cand_idx"]),
            int(candidate["k2_track_idx"]),
            "Phi_K_2_pt",
            "Phi_K_2_eta",
            "Phi_K_2_phi",
            "Phi_K_2_vertexId",
            "Phi_K_2_fromPV",
            "Phi_K_2_pvAssocQuality",
            "Phi_K_2_dzPV",
            "Phi_K_2_dxyPV",
            "Phi_K_2_dzAssocPV",
            "Phi_K_2_dxyAssocPV",
            "Phi_K_2_genMatchIdx",
            "Phi_K_2_genMatchSource",
        ),
    ]

    for role, cand_idx, track_idx, pt_key, eta_key, phi_key, vertex_key, from_pv_key, pvq_key, dz_key, dxy_key, dz_assoc_key, dxy_assoc_key, gen_key, gen_src_key in kaon_specs:
        match_idx = int(event[gen_key][cand_idx]) if cand_idx < len(event[gen_key]) else -1
        match_source = int(event[gen_src_key][cand_idx]) if cand_idx < len(event[gen_src_key]) else 0
        matched_gen = gen_rows[match_idx] if 0 <= match_idx < len(gen_rows) else None
        phi_ancestor_idx = first_ancestor_idx(gen_pdg, gen_mother_idx, match_idx, 333) if match_idx >= 0 else -1
        if match_idx >= 0:
            focus_gen_indices.add(match_idx)
        if phi_ancestor_idx >= 0:
            focus_gen_indices.add(phi_ancestor_idx)
        kaon_rows.append(
            {
                "role": role,
                "track_idx": track_idx,
                "pt": float(event[pt_key][cand_idx]) if cand_idx < len(event[pt_key]) else None,
                "eta": float(event[eta_key][cand_idx]) if cand_idx < len(event[eta_key]) else None,
                "phi": float(event[phi_key][cand_idx]) if cand_idx < len(event[phi_key]) else None,
                "vertexId": int(event[vertex_key][cand_idx]) if cand_idx < len(event[vertex_key]) else -1,
                "fromPV": int(event[from_pv_key][cand_idx]) if cand_idx < len(event[from_pv_key]) else -1,
                "pvAssocQuality": int(event[pvq_key][cand_idx]) if cand_idx < len(event[pvq_key]) else -1,
                "dzPV": float(event[dz_key][cand_idx]) if cand_idx < len(event[dz_key]) else None,
                "dxyPV": float(event[dxy_key][cand_idx]) if cand_idx < len(event[dxy_key]) else None,
                "dzAssocPV": float(event[dz_assoc_key][cand_idx]) if cand_idx < len(event[dz_assoc_key]) else None,
                "dxyAssocPV": float(event[dxy_assoc_key][cand_idx]) if cand_idx < len(event[dxy_assoc_key]) else None,
                "genMatchIdx": match_idx,
                "genSource": label_value(KAON_GEN_SOURCE_LABELS, match_source),
                "matchedParticle": matched_gen["particle"] if matched_gen else "unmatched",
                "matchedMother": matched_gen["mother_particle"] if matched_gen else "unmatched",
                "phiAncestorIdx": phi_ancestor_idx,
                "motherChain": mother_chain(gen_rows, match_idx),
            }
        )

    focused_gen_rows = []
    for gen_idx in sorted(focus_gen_indices):
        row = dict(gen_rows[gen_idx])
        row["motherChain"] = mother_chain(gen_rows, gen_idx)
        focused_gen_rows.append(row)

    return header, pd.DataFrame(mu_rows), pd.DataFrame(kaon_rows), pd.DataFrame(focused_gen_rows)
