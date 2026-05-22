from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from scipy.stats import beta

from .config import OfflineSelectionConfig
from .truth import first_ancestor_idx, to_int_idx


EFFICIENCY_STEPS = (
    "full_gen",
    "fiducial_acceptance",
    "hlt_muon_matched",
    "single_jpsi_reco",
    "double_jpsi_reco",
    "single_phi_reco",
    "triple_gen_matched_candidate",
    "jpsi_quality",
    "phi_quality",
    "all6_same_recVtx",
    "Pri_fitValid",
    "Pri_fitPass",
    "Pri_assocPVPass",
    "Pri_trackPVPass",
    "final_nominal",
)

CORRELATED_MAP_STEPS = (
    "hlt_muon_matched",
    "all6_same_recVtx",
    "Pri_fitValid",
    "Pri_fitPass",
    "final_nominal",
)

EFFICIENCY_BRANCHES = sorted(
    {
        "evtNum",
        "runNum",
        "lumiNum",
        "TrigNames",
        "TrigRes",
        "Jpsi_1_mass",
        "Jpsi_1_pt",
        "Jpsi_1_px",
        "Jpsi_1_py",
        "Jpsi_1_pz",
        "Jpsi_1_VtxProb",
        "Jpsi_1_mu_1_Idx",
        "Jpsi_1_mu_2_Idx",
        "Jpsi_2_mass",
        "Jpsi_2_pt",
        "Jpsi_2_px",
        "Jpsi_2_py",
        "Jpsi_2_pz",
        "Jpsi_2_VtxProb",
        "Jpsi_2_mu_1_Idx",
        "Jpsi_2_mu_2_Idx",
        "Phi_mass",
        "Phi_pt",
        "Phi_px",
        "Phi_py",
        "Phi_pz",
        "Phi_VtxProb",
        "Phi_K_1_Idx",
        "Phi_K_2_Idx",
        "Phi_K_1_pt",
        "Phi_K_1_eta",
        "Phi_K_1_vertexId",
        "Phi_K_1_genMatchIdx",
        "Phi_K_2_pt",
        "Phi_K_2_eta",
        "Phi_K_2_vertexId",
        "Phi_K_2_genMatchIdx",
        "muGenMatchIdx",
        "muVertexId",
        "muIsJpsiTrigMatch",
        "muIsJpsiFilterMatch",
        "muPx",
        "muPy",
        "muPz",
        "muCharge",
        "muIsPatSoftMuon",
        "Pri_fitValid",
        "Pri_fitPass",
        "Pri_assocPVPass",
        "Pri_trackPVPass",
        "Pri_passAny",
        "MC_GenPart_pdgId",
        "MC_GenPart_motherGenIdx",
        "MC_GenPart_pt",
        "MC_GenPart_eta",
        "MC_GenPart_phi",
        "MC_GenPart_mass",
    }
)


@dataclass(frozen=True)
class EfficiencyBinning:
    jpsi_pt_edges: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)
    phi_pt_edges: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 50.0)
    object_abs_y_edges: tuple[float, ...] = (0.0, 0.6, 1.2, 1.8, 2.4)
    triple_pt_edges: tuple[float, ...] = (0.0, 10.0, 20.0, 30.0, 50.0, 100.0, 200.0)
    triple_abs_y_edges: tuple[float, ...] = (0.0, 0.6, 1.2, 1.8, 2.4)
    triple_mass_edges: tuple[float, ...] = (0.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0)


@dataclass(frozen=True)
class EfficiencyRunConfig:
    analysis_mode: str = "JpsiJpsiPhi"
    tree_path: str = "mkcands/X_data"
    xrootd_host: str = "root://cceos.ihep.ac.cn//"
    sample_root: str = "/eos/ihep/cms/store/user/xcheng/MC_Production_v3/output"
    samples: tuple[str, ...] = ("JJP_DPS1", "JJP_DPS2_CS", "JJP_DPS2_G", "JJP_SPS_CS", "JJP_SPS_G")
    max_files: int | None = None
    min_plot_total: int = 1


@dataclass(frozen=True)
class GenParticle:
    idx: int
    pdg_id: int
    pt: float
    eta: float
    phi: float
    mass: float
    daughter_indices: tuple[int, ...]


@dataclass(frozen=True)
class GenSystem:
    jpsi_lead: GenParticle
    jpsi_sublead: GenParticle
    phi: GenParticle
    n_jpsi: int
    n_phi: int
    triple_pt: float
    triple_abs_y: float
    triple_mass: float


def xrootd_url(host: str, remote_path: str) -> str:
    return f"{host.rstrip('/')}//{remote_path}"


def _limit_efficiency_files(files: list[str], max_files: int | None) -> list[str]:
    if max_files is None:
        return files
    return files[:max_files]


def load_efficiency_file_manifest(
    path: str | Path,
    *,
    samples: tuple[str, ...] | None = None,
    max_files: int | None = None,
) -> dict[str, list[str]]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input file manifest is not valid JSON: {manifest_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Input file manifest must be a JSON object mapping sample names to file lists.")

    requested = tuple(samples or ())
    sample_names = requested if requested else tuple(str(sample) for sample in payload)
    files_by_sample: dict[str, list[str]] = {}
    missing_samples: list[str] = []
    for sample in sample_names:
        if not sample:
            raise ValueError("Input file manifest sample names must be non-empty strings.")
        if sample not in payload:
            missing_samples.append(sample)
            continue
        raw_files = payload[sample]
        if not isinstance(raw_files, list):
            raise ValueError(f"Input file manifest entry for sample {sample!r} must be a list of file strings.")
        files: list[str] = []
        for index, item in enumerate(raw_files):
            if not isinstance(item, str) or not item:
                raise ValueError(f"Input file manifest entry {sample}[{index}] must be a non-empty file string.")
            files.append(item)
        if not files:
            raise ValueError(f"Input file manifest entry for sample {sample!r} must contain at least one file.")
        files_by_sample[sample] = _limit_efficiency_files(files, max_files)

    if missing_samples:
        available = ", ".join(str(sample) for sample in payload) or "(none)"
        missing = ", ".join(missing_samples)
        raise ValueError(f"Input file manifest is missing requested sample(s): {missing}. Available samples: {available}")
    return files_by_sample


def xrdfs_ls(host: str, remote_path: str, dirs_only: bool = False) -> list[str]:
    cmd = ["xrdfs", host, "ls"]
    if dirs_only:
        cmd.append("-d")
    cmd.append(remote_path)
    completed = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def discover_xrootd_sample_files(
    *,
    host: str,
    sample_root: str,
    samples: tuple[str, ...],
    max_files: int | None = None,
) -> dict[str, list[str]]:
    discovered: dict[str, list[str]] = {}
    for sample in samples:
        sample_dir = f"{sample_root.rstrip('/')}/{sample}"
        job_dirs = sorted(xrdfs_ls(host, sample_dir), key=_natural_remote_key)
        files: list[str] = []
        for job_dir in job_dirs:
            entries = xrdfs_ls(host, job_dir)
            ntuples = [entry for entry in entries if entry.endswith("/output_ntuple.root")]
            files.extend(xrootd_url(host, path) for path in ntuples)
            if max_files is not None and len(files) >= max_files:
                files = _limit_efficiency_files(files, max_files)
                break
        discovered[sample] = files
    return discovered


def _natural_remote_key(path: str) -> tuple[Any, ...]:
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def _record_field(arrays: Any, name: str) -> ak.Array:
    if isinstance(arrays, dict):
        return arrays[name]
    return arrays[name]


def _record_fields(arrays: Any) -> list[str]:
    if isinstance(arrays, dict):
        return list(arrays)
    return list(arrays.fields)


def _pythonize_event(arrays: Any, entry: int) -> dict[str, Any]:
    return {name: ak.to_list(_record_field(arrays, name)[entry]) for name in _record_fields(arrays)}


def _event_value(event: dict[str, Any], field: str, idx: int, default: Any = None) -> Any:
    values = event.get(field)
    if values is None or idx < 0 or idx >= len(values):
        return default
    return values[idx]


def _p4_from_pt_eta_phi_m(pt: float, eta: float, phi: float, mass: float) -> tuple[float, float, float, float]:
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    energy = math.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px, py, pz, energy


def _p4_kinematics(px: float, py: float, pz: float, energy: float) -> tuple[float, float, float]:
    pt = math.hypot(px, py)
    mass2 = energy * energy - px * px - py * py - pz * pz
    mass = math.sqrt(max(mass2, 0.0))
    if energy + pz <= 0.0 or energy - pz <= 0.0:
        rapidity = math.nan
    else:
        rapidity = 0.5 * math.log((energy + pz) / (energy - pz))
    return pt, abs(rapidity), mass


def _daughter_indices(gen_mother_idx: list[int], mother_idx: int, abs_pdgs: set[int], gen_pdg: list[int]) -> tuple[int, ...]:
    return tuple(
        idx
        for idx, mother in enumerate(gen_mother_idx)
        if to_int_idx(mother, -1) == mother_idx and abs(int(gen_pdg[idx])) in abs_pdgs
    )


def find_jpsijpsiphi_gen_system(event: dict[str, Any]) -> GenSystem | None:
    gen_pdg = [int(value) for value in event.get("MC_GenPart_pdgId", [])]
    gen_mother_idx = [to_int_idx(value, -1) for value in event.get("MC_GenPart_motherGenIdx", [])]
    gen_pt = event.get("MC_GenPart_pt", [])
    gen_eta = event.get("MC_GenPart_eta", [])
    gen_phi = event.get("MC_GenPart_phi", [])
    gen_mass = event.get("MC_GenPart_mass", [])

    particles: list[GenParticle] = []
    for idx, pdg_id in enumerate(gen_pdg):
        abs_pdg = abs(int(pdg_id))
        if abs_pdg == 443:
            daughters = _daughter_indices(gen_mother_idx, idx, {13}, gen_pdg)
            if len(daughters) >= 2:
                particles.append(
                    GenParticle(idx, pdg_id, float(gen_pt[idx]), float(gen_eta[idx]), float(gen_phi[idx]), float(gen_mass[idx]), daughters[:2])
                )
        elif abs_pdg == 333:
            daughters = _daughter_indices(gen_mother_idx, idx, {321}, gen_pdg)
            if len(daughters) >= 2:
                particles.append(
                    GenParticle(idx, pdg_id, float(gen_pt[idx]), float(gen_eta[idx]), float(gen_phi[idx]), float(gen_mass[idx]), daughters[:2])
                )

    jpsis = sorted([particle for particle in particles if abs(particle.pdg_id) == 443], key=lambda item: item.pt, reverse=True)
    phis = sorted([particle for particle in particles if abs(particle.pdg_id) == 333], key=lambda item: item.pt, reverse=True)
    if len(jpsis) < 2 or not phis:
        return None

    selected = (jpsis[0], jpsis[1], phis[0])
    px = py = pz = energy = 0.0
    for particle in selected:
        p4 = _p4_from_pt_eta_phi_m(particle.pt, particle.eta, particle.phi, particle.mass)
        px += p4[0]
        py += p4[1]
        pz += p4[2]
        energy += p4[3]
    triple_pt, triple_abs_y, triple_mass = _p4_kinematics(px, py, pz, energy)
    return GenSystem(
        jpsi_lead=jpsis[0],
        jpsi_sublead=jpsis[1],
        phi=phis[0],
        n_jpsi=len(jpsis),
        n_phi=len(phis),
        triple_pt=triple_pt,
        triple_abs_y=triple_abs_y,
        triple_mass=triple_mass,
    )


def _daughter_fiducial(event: dict[str, Any], particle: GenParticle, cfg: OfflineSelectionConfig, kind: Literal["muon", "kaon"]) -> bool:
    pt_min = cfg.track_pt_min if kind == "kaon" else None
    eta_max = cfg.track_abs_eta_max if kind == "kaon" else cfg.mu_abs_eta_max
    for idx in particle.daughter_indices:
        pt = float(_event_value(event, "MC_GenPart_pt", idx, math.nan))
        eta = abs(float(_event_value(event, "MC_GenPart_eta", idx, math.nan)))
        if kind == "muon":
            barrel = eta < 1.2 and pt > cfg.mu_pt_barrel_min
            endcap = 1.2 <= eta < cfg.mu_abs_eta_max and pt > cfg.mu_pt_endcap_min
            if not (barrel or endcap):
                return False
        elif not (pt > float(pt_min) and eta < eta_max):
            return False
    return True


def gen_system_fiducial(event: dict[str, Any], system: GenSystem, cfg: OfflineSelectionConfig) -> bool:
    return (
        _daughter_fiducial(event, system.jpsi_lead, cfg, "muon")
        and _daughter_fiducial(event, system.jpsi_sublead, cfg, "muon")
        and _daughter_fiducial(event, system.phi, cfg, "kaon")
    )


def _muon_gen_ancestor(event: dict[str, Any], muon_idx: int) -> int:
    gen_idx = to_int_idx(_event_value(event, "muGenMatchIdx", muon_idx, -1), -1)
    return first_ancestor_idx(event["MC_GenPart_pdgId"], event["MC_GenPart_motherGenIdx"], gen_idx, 443)


def _kaon_gen_ancestor(event: dict[str, Any], field: str, cand_idx: int) -> int:
    gen_idx = to_int_idx(_event_value(event, field, cand_idx, -1), -1)
    return first_ancestor_idx(event["MC_GenPart_pdgId"], event["MC_GenPart_motherGenIdx"], gen_idx, 333)


def _candidate_leg_indices(event: dict[str, Any], cand_idx: int) -> dict[str, int]:
    mu_indices = {
        "Jpsi_1_mu_1": to_int_idx(_event_value(event, "Jpsi_1_mu_1_Idx", cand_idx, -1), -1),
        "Jpsi_1_mu_2": to_int_idx(_event_value(event, "Jpsi_1_mu_2_Idx", cand_idx, -1), -1),
        "Jpsi_2_mu_1": to_int_idx(_event_value(event, "Jpsi_2_mu_1_Idx", cand_idx, -1), -1),
        "Jpsi_2_mu_2": to_int_idx(_event_value(event, "Jpsi_2_mu_2_Idx", cand_idx, -1), -1),
    }
    jpsi1 = [_muon_gen_ancestor(event, mu_indices["Jpsi_1_mu_1"]), _muon_gen_ancestor(event, mu_indices["Jpsi_1_mu_2"])]
    jpsi2 = [_muon_gen_ancestor(event, mu_indices["Jpsi_2_mu_1"]), _muon_gen_ancestor(event, mu_indices["Jpsi_2_mu_2"])]
    phi = [
        _kaon_gen_ancestor(event, "Phi_K_1_genMatchIdx", cand_idx),
        _kaon_gen_ancestor(event, "Phi_K_2_genMatchIdx", cand_idx),
    ]
    return {
        "jpsi1": jpsi1[0] if jpsi1[0] >= 0 and jpsi1[0] == jpsi1[1] else -1,
        "jpsi2": jpsi2[0] if jpsi2[0] >= 0 and jpsi2[0] == jpsi2[1] else -1,
        "phi": phi[0] if phi[0] >= 0 and phi[0] == phi[1] else -1,
    }


def _candidate_matches_system(event: dict[str, Any], cand_idx: int, system: GenSystem) -> bool:
    legs = _candidate_leg_indices(event, cand_idx)
    jpsi_set = {legs["jpsi1"], legs["jpsi2"]}
    return (
        legs["phi"] == system.phi.idx
        and jpsi_set == {system.jpsi_lead.idx, system.jpsi_sublead.idx}
    )


def _candidate_hlt_muon_matched(event: dict[str, Any], cand_idx: int) -> bool:
    for slot1, slot2 in (("Jpsi_1_mu_1_Idx", "Jpsi_1_mu_2_Idx"), ("Jpsi_2_mu_1_Idx", "Jpsi_2_mu_2_Idx")):
        idx1 = to_int_idx(_event_value(event, slot1, cand_idx, -1), -1)
        idx2 = to_int_idx(_event_value(event, slot2, cand_idx, -1), -1)
        filter1 = bool(to_int_idx(_event_value(event, "muIsJpsiFilterMatch", idx1, 0), 0))
        filter2 = bool(to_int_idx(_event_value(event, "muIsJpsiFilterMatch", idx2, 0), 0))
        trig1 = bool(to_int_idx(_event_value(event, "muIsJpsiTrigMatch", idx1, 0), 0))
        trig2 = bool(to_int_idx(_event_value(event, "muIsJpsiTrigMatch", idx2, 0), 0))
        if (filter1 and filter2) or (trig1 and trig2):
            return True
    return False


def _candidate_all6_same_rec_vtx(event: dict[str, Any], cand_idx: int) -> bool:
    muon_indices = [
        to_int_idx(_event_value(event, "Jpsi_1_mu_1_Idx", cand_idx, -1), -1),
        to_int_idx(_event_value(event, "Jpsi_1_mu_2_Idx", cand_idx, -1), -1),
        to_int_idx(_event_value(event, "Jpsi_2_mu_1_Idx", cand_idx, -1), -1),
        to_int_idx(_event_value(event, "Jpsi_2_mu_2_Idx", cand_idx, -1), -1),
    ]
    vertex_ids = [to_int_idx(_event_value(event, "muVertexId", idx, -1), -1) for idx in muon_indices]
    vertex_ids.extend(
        [
            to_int_idx(_event_value(event, "Phi_K_1_vertexId", cand_idx, -1), -1),
            to_int_idx(_event_value(event, "Phi_K_2_vertexId", cand_idx, -1), -1),
        ]
    )
    return min(vertex_ids) >= 0 and len(set(vertex_ids)) == 1


def _scalar_rapidity(px: float, py: float, pz: float, mass: float) -> float:
    energy = math.sqrt(max(px * px + py * py + pz * pz + mass * mass, 0.0))
    if energy + pz <= 0.0 or energy - pz <= 0.0:
        return math.nan
    return 0.5 * math.log((energy + pz) / (energy - pz))


def _jpsi_quality(event: dict[str, Any], cand_idx: int, cfg: OfflineSelectionConfig) -> bool:
    for prefix in ("Jpsi_1", "Jpsi_2"):
        px = float(_event_value(event, f"{prefix}_px", cand_idx, math.nan))
        py = float(_event_value(event, f"{prefix}_py", cand_idx, math.nan))
        pz = float(_event_value(event, f"{prefix}_pz", cand_idx, math.nan))
        mass = float(_event_value(event, f"{prefix}_mass", cand_idx, math.nan))
        y = _scalar_rapidity(px, py, pz, mass)
        if not (
            cfg.jpsi_mass_window[0] <= mass <= cfg.jpsi_mass_window[1]
            and float(_event_value(event, f"{prefix}_pt", cand_idx, -1.0)) > cfg.jpsi_pt_min
            and abs(y) < cfg.jpsi_abs_y_max
            and float(_event_value(event, f"{prefix}_VtxProb", cand_idx, -1.0)) > cfg.jpsi_vtxprob_min
        ):
            return False
    return True


def _phi_quality(event: dict[str, Any], cand_idx: int, cfg: OfflineSelectionConfig) -> bool:
    return bool(
        cfg.phi_mass_window[0] <= float(_event_value(event, "Phi_mass", cand_idx, math.nan)) <= cfg.phi_mass_window[1]
        and float(_event_value(event, "Phi_pt", cand_idx, -1.0)) > cfg.phi_pt_min
        and float(_event_value(event, "Phi_VtxProb", cand_idx, -1.0)) > cfg.phi_vtxprob_min
        and float(_event_value(event, "Phi_K_1_pt", cand_idx, -1.0)) > cfg.track_pt_min
        and float(_event_value(event, "Phi_K_2_pt", cand_idx, -1.0)) > cfg.track_pt_min
        and abs(float(_event_value(event, "Phi_K_1_eta", cand_idx, math.nan))) < cfg.track_abs_eta_max
        and abs(float(_event_value(event, "Phi_K_2_eta", cand_idx, math.nan))) < cfg.track_abs_eta_max
    )


def _event_path_or(event: dict[str, Any], patterns: tuple[str, ...] = ("HLT_Dimuon0_Jpsi", "HLT_DoubleMu4_3_LowMass")) -> bool:
    names = event.get("TrigNames", [])
    results = event.get("TrigRes", [])
    for name, result in zip(names, results):
        if any(pattern in str(name) for pattern in patterns) and bool(result):
            return True
    return False


def build_event_efficiency_row(
    event: dict[str, Any],
    source_file: str,
    sample: str,
    entry: int,
    cfg: OfflineSelectionConfig,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    system = find_jpsijpsiphi_gen_system(event)
    if system is None:
        return None, None

    n_candidates = len(event.get("Jpsi_1_mass", []))
    raw = {step: False for step in EFFICIENCY_STEPS}
    raw["full_gen"] = True
    raw["fiducial_acceptance"] = gen_system_fiducial(event, system, cfg)
    raw["hlt_event_path_or"] = _event_path_or(event)

    matched_candidates = [cand_idx for cand_idx in range(n_candidates) if _candidate_matches_system(event, cand_idx, system)]
    jpsi_reco_indices: set[int] = set()
    phi_reco = False
    for cand_idx in range(n_candidates):
        legs = _candidate_leg_indices(event, cand_idx)
        if legs["jpsi1"] in {system.jpsi_lead.idx, system.jpsi_sublead.idx}:
            jpsi_reco_indices.add(legs["jpsi1"])
        if legs["jpsi2"] in {system.jpsi_lead.idx, system.jpsi_sublead.idx}:
            jpsi_reco_indices.add(legs["jpsi2"])
        phi_reco = phi_reco or legs["phi"] == system.phi.idx

    raw["single_jpsi_reco"] = bool(jpsi_reco_indices)
    raw["double_jpsi_reco"] = len(jpsi_reco_indices) >= 2
    raw["single_phi_reco"] = phi_reco
    raw["triple_gen_matched_candidate"] = bool(matched_candidates)
    raw["hlt_muon_matched"] = any(_candidate_hlt_muon_matched(event, cand_idx) for cand_idx in matched_candidates)
    raw["jpsi_quality"] = any(_jpsi_quality(event, cand_idx, cfg) for cand_idx in matched_candidates)
    raw["phi_quality"] = any(_phi_quality(event, cand_idx, cfg) for cand_idx in matched_candidates)
    raw["all6_same_recVtx"] = any(_candidate_all6_same_rec_vtx(event, cand_idx) for cand_idx in matched_candidates)
    for step in ("Pri_fitValid", "Pri_fitPass", "Pri_assocPVPass", "Pri_trackPVPass"):
        raw[step] = any(to_int_idx(_event_value(event, step, cand_idx, 0), 0) == 1 for cand_idx in matched_candidates)

    cumulative: dict[str, int] = {}
    keep = True
    for step in EFFICIENCY_STEPS[:-1]:
        keep = keep and bool(raw[step])
        cumulative[step] = int(keep)
    cumulative["final_nominal"] = int(
        cumulative["Pri_fitPass"] == 1
        and cumulative["Pri_assocPVPass"] == 1
        and cumulative["Pri_trackPVPass"] == 1
    )

    event_row: dict[str, Any] = {
        "sample": sample,
        "source_file": source_file,
        "entry": int(entry),
        "run": int(event["runNum"]),
        "lumi": int(event["lumiNum"]),
        "event": int(event["evtNum"]),
        "n_candidates": int(n_candidates),
        "n_gen_jpsi": int(system.n_jpsi),
        "n_gen_phi": int(system.n_phi),
        "n_triple_gen_matched_candidates": int(len(matched_candidates)),
        "hlt_event_path_or": int(raw["hlt_event_path_or"]),
        **cumulative,
    }
    gen_row = {
        "sample": sample,
        "source_file": source_file,
        "entry": int(entry),
        "run": int(event["runNum"]),
        "lumi": int(event["lumiNum"]),
        "event": int(event["evtNum"]),
        "jpsi_lead_gen_idx": system.jpsi_lead.idx,
        "jpsi_sublead_gen_idx": system.jpsi_sublead.idx,
        "phi_gen_idx": system.phi.idx,
        "jpsi_lead_pt": system.jpsi_lead.pt,
        "jpsi_lead_abs_y": abs(system.jpsi_lead.eta),
        "jpsi_sublead_pt": system.jpsi_sublead.pt,
        "jpsi_sublead_abs_y": abs(system.jpsi_sublead.eta),
        "phi_pt": system.phi.pt,
        "phi_abs_y": abs(system.phi.eta),
        "triple_pt": system.triple_pt,
        "triple_abs_y": system.triple_abs_y,
        "triple_mass": system.triple_mass,
    }
    return gen_row, event_row


def process_efficiency_file(path: str, sample: str, cfg: OfflineSelectionConfig, tree_path: str = "mkcands/X_data") -> dict[str, pd.DataFrame]:
    with uproot.open(path) as root_file:
        tree = root_file[tree_path]
        available = set(tree.keys())
        branches = [branch for branch in EFFICIENCY_BRANCHES if branch in available]
        arrays = tree.arrays(branches, library="ak")

    gen_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    n_entries = len(_record_field(arrays, "evtNum"))
    for entry in range(n_entries):
        event = _pythonize_event(arrays, entry)
        gen_row, event_row = build_event_efficiency_row(event, path, sample, entry, cfg)
        if gen_row is not None and event_row is not None:
            gen_rows.append(gen_row)
            event_rows.append(event_row)
    return {
        "gen_systems": pd.DataFrame(gen_rows),
        "event_step_flags": pd.DataFrame(event_rows),
    }


def run_efficiency_for_sample(
    files: list[str],
    sample: str,
    cfg: OfflineSelectionConfig | None = None,
    tree_path: str = "mkcands/X_data",
) -> dict[str, pd.DataFrame]:
    cfg = cfg or OfflineSelectionConfig()
    gen_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    for path in files:
        tables = process_efficiency_file(path, sample, cfg, tree_path=tree_path)
        if not tables["gen_systems"].empty:
            gen_parts.append(tables["gen_systems"])
        if not tables["event_step_flags"].empty:
            event_parts.append(tables["event_step_flags"])
    gen_df = pd.concat(gen_parts, ignore_index=True) if gen_parts else pd.DataFrame()
    event_df = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    counts_df = build_efficiency_counts(gen_df, event_df, EfficiencyBinning())
    return {
        "gen_systems": gen_df,
        "event_step_flags": event_df,
        "efficiency_counts": counts_df,
        "cutflow": build_cutflow(event_df),
    }


def clopper_pearson_interval(total: int, passed: int, confidence: float = 0.682689492) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    alpha = 1.0 - confidence
    low = 0.0 if passed <= 0 else float(beta.ppf(alpha / 2.0, passed, total - passed + 1))
    high = 1.0 if passed >= total else float(beta.ppf(1.0 - alpha / 2.0, passed + 1, total - passed))
    return low, high


def _efficiency_row(base: dict[str, Any], total: int, passed: int) -> dict[str, Any]:
    efficiency = float(passed / total) if total > 0 else math.nan
    low, high = clopper_pearson_interval(total, passed)
    err_low = efficiency - low if total > 0 else math.nan
    err_high = high - efficiency if total > 0 else math.nan
    return {
        **base,
        "total": int(total),
        "passed": int(passed),
        "efficiency": efficiency,
        "err_low": err_low,
        "err_high": err_high,
        "err_sym": max(err_low, err_high) if total > 0 else math.nan,
    }


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    if not np.isfinite(value):
        return -1
    idx = int(np.searchsorted(np.asarray(edges), value, side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return -1
    return idx


def _bin_label(edges: tuple[float, ...], idx: int) -> str:
    return f"{edges[idx]:g}-{edges[idx + 1]:g}"


def _merged_gen_events(gen_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample", "source_file", "entry", "run", "lumi", "event"]
    if gen_df.empty or event_df.empty:
        return pd.DataFrame()
    return gen_df.merge(event_df, on=keys, how="inner")


def build_cutflow(event_df: pd.DataFrame) -> pd.DataFrame:
    if event_df.empty:
        return pd.DataFrame(columns=["step", "total", "passed", "efficiency", "err_low", "err_high", "err_sym", "conditional_efficiency"])
    total = int(event_df["full_gen"].sum())
    rows: list[dict[str, Any]] = []
    previous = total
    for step in EFFICIENCY_STEPS:
        passed = int(event_df[step].sum()) if step in event_df.columns else 0
        row = _efficiency_row({"step": step}, total, passed)
        row["conditional_total"] = int(previous)
        row["conditional_passed"] = int(passed)
        row["conditional_efficiency"] = float(passed / previous) if previous > 0 else math.nan
        rows.append(row)
        previous = passed
    return pd.DataFrame(rows)


def build_efficiency_counts(gen_df: pd.DataFrame, event_df: pd.DataFrame, binning: EfficiencyBinning) -> pd.DataFrame:
    merged = _merged_gen_events(gen_df, event_df)
    if merged.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    rows.extend(_inclusive_counts(merged))
    rows.extend(_object_2d_counts(merged, binning))
    rows.extend(_correlated_3d_counts(merged, binning))
    rows.extend(_triple_sidecheck_counts(merged, binning))
    return pd.DataFrame(rows)


def _inclusive_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for step in EFFICIENCY_STEPS:
        rows.append(_efficiency_row({"map_type": "inclusive", "step": step}, int(frame["full_gen"].sum()), int(frame[step].sum())))
    return rows


def _object_2d_counts(frame: pd.DataFrame, binning: EfficiencyBinning) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = (
        ("jpsi_lead", "jpsi_lead_pt", "jpsi_lead_abs_y", binning.jpsi_pt_edges, binning.object_abs_y_edges),
        ("jpsi_sublead", "jpsi_sublead_pt", "jpsi_sublead_abs_y", binning.jpsi_pt_edges, binning.object_abs_y_edges),
        ("phi", "phi_pt", "phi_abs_y", binning.phi_pt_edges, binning.object_abs_y_edges),
    )
    for obj, pt_col, y_col, pt_edges, y_edges in specs:
        for ix in range(len(pt_edges) - 1):
            for iy in range(len(y_edges) - 1):
                subset = frame[
                    (frame[pt_col] >= pt_edges[ix])
                    & (frame[pt_col] < pt_edges[ix + 1])
                    & (frame[y_col] >= y_edges[iy])
                    & (frame[y_col] < y_edges[iy + 1])
                ]
                total = int(subset["full_gen"].sum())
                for step in EFFICIENCY_STEPS:
                    rows.append(
                        _efficiency_row(
                            {
                                "map_type": "object_2d",
                                "object": obj,
                                "step": step,
                                "x_axis": "pt",
                                "y_axis": "abs_y",
                                "x_bin": ix,
                                "y_bin": iy,
                                "x_min": pt_edges[ix],
                                "x_max": pt_edges[ix + 1],
                                "y_min": y_edges[iy],
                                "y_max": y_edges[iy + 1],
                                "x_label": _bin_label(pt_edges, ix),
                                "y_label": _bin_label(y_edges, iy),
                            },
                            total,
                            int(subset[step].sum()) if total else 0,
                        )
                    )
    return rows


def _correlated_3d_counts(frame: pd.DataFrame, binning: EfficiencyBinning) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    jpsi_edges = binning.jpsi_pt_edges
    phi_edges = binning.phi_pt_edges
    for ix in range(len(jpsi_edges) - 1):
        for iy in range(len(jpsi_edges) - 1):
            for iz in range(len(phi_edges) - 1):
                subset = frame[
                    (frame["jpsi_lead_pt"] >= jpsi_edges[ix])
                    & (frame["jpsi_lead_pt"] < jpsi_edges[ix + 1])
                    & (frame["jpsi_sublead_pt"] >= jpsi_edges[iy])
                    & (frame["jpsi_sublead_pt"] < jpsi_edges[iy + 1])
                    & (frame["phi_pt"] >= phi_edges[iz])
                    & (frame["phi_pt"] < phi_edges[iz + 1])
                ]
                total = int(subset["full_gen"].sum())
                for step in CORRELATED_MAP_STEPS:
                    rows.append(
                        _efficiency_row(
                            {
                                "map_type": "correlated_3d",
                                "step": step,
                                "x_axis": "jpsi_lead_pt",
                                "y_axis": "jpsi_sublead_pt",
                                "z_axis": "phi_pt",
                                "x_bin": ix,
                                "y_bin": iy,
                                "z_bin": iz,
                                "x_min": jpsi_edges[ix],
                                "x_max": jpsi_edges[ix + 1],
                                "y_min": jpsi_edges[iy],
                                "y_max": jpsi_edges[iy + 1],
                                "z_min": phi_edges[iz],
                                "z_max": phi_edges[iz + 1],
                                "x_label": _bin_label(jpsi_edges, ix),
                                "y_label": _bin_label(jpsi_edges, iy),
                                "z_label": _bin_label(phi_edges, iz),
                            },
                            total,
                            int(subset[step].sum()) if total else 0,
                        )
                    )
    return rows


def _triple_sidecheck_counts(frame: pd.DataFrame, binning: EfficiencyBinning) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = (
        ("triple_pt", binning.triple_pt_edges),
        ("triple_abs_y", binning.triple_abs_y_edges),
        ("triple_mass", binning.triple_mass_edges),
    )
    for axis, edges in specs:
        for idx in range(len(edges) - 1):
            subset = frame[(frame[axis] >= edges[idx]) & (frame[axis] < edges[idx + 1])]
            total = int(subset["full_gen"].sum())
            for step in EFFICIENCY_STEPS:
                rows.append(
                    _efficiency_row(
                        {
                            "map_type": "triple_1d",
                            "step": step,
                            "x_axis": axis,
                            "x_bin": idx,
                            "x_min": edges[idx],
                            "x_max": edges[idx + 1],
                            "x_label": _bin_label(edges, idx),
                        },
                        total,
                        int(subset[step].sum()) if total else 0,
                    )
                )
    return rows


def build_subprocess_envelope(sample_count_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for sample, frame in sample_count_tables.items():
        if frame.empty:
            continue
        tagged = frame.copy()
        tagged["sample"] = sample
        frames.append(tagged)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    group_cols = [
        column
        for column in combined.columns
        if column
        not in {
            "sample",
            "total",
            "passed",
            "efficiency",
            "err_low",
            "err_high",
            "err_sym",
        }
        and not column.startswith("err_")
    ]
    rows = []
    for keys, group in combined.groupby(group_cols, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, key_values))
        values = group["efficiency"].dropna()
        median = float(values.median()) if not values.empty else math.nan
        row.update(
            {
                "n_samples": int(group["sample"].nunique()),
                "eff_min": float(values.min()) if not values.empty else math.nan,
                "eff_max": float(values.max()) if not values.empty else math.nan,
                "eff_median": median,
                "max_abs_deviation": float(np.max(np.abs(values.to_numpy() - median))) if not values.empty else math.nan,
                "total_sum": int(group["total"].sum()),
                "passed_sum": int(group["passed"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
