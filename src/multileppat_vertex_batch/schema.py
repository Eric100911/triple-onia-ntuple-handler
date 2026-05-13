from __future__ import annotations

from dataclasses import dataclass

EVENT_KEYS = ["source_file", "run", "lumi", "event"]

CACHE_METADATA_FILENAME = "cache_meta.json"
TRUTH_CACHE_VERSION = 2
MASS_SELECTION_CACHE_VERSION = 3
FIT_COMPARE_CACHE_VERSIONS = {
    "roofit": 7,
    "iminuit": 5,
}
FIT_SIGNIFICANCE_VERSION = 2
UPS_SIGNAL_MODEL_VERSION = 5
PHI_BACKGROUND_MODEL_VERSION = 2

PDG_LABELS = {
    443: "J/psi",
    -443: "J/psi",
    553: "Upsilon",
    -553: "Upsilon",
    333: "phi",
    -333: "phi",
    13: "mu-",
    -13: "mu+",
    321: "K+",
    -321: "K-",
}

MU_PACKED_METHOD_LABELS = {
    0: "unmatched",
    1: "sourceCandidatePtr",
    2: "vector",
    3: "chi2",
    4: "dzAssoc",
    5: "dzPv",
}

MU_GEN_SOURCE_LABELS = {
    0: "unmatched",
    1: "patRef",
    2: "chi2Fallback",
}

KAON_GEN_SOURCE_LABELS = {
    0: "unmatched",
    1: "phiMotherChi2",
    2: "chi2Fallback",
}


@dataclass(frozen=True)
class AxisSpec:
    branch: str
    role: str
    object_name: str


@dataclass(frozen=True)
class TruthLegSpec:
    object_name: str
    role: str
    daughter_keys: tuple[str, ...]
    ancestor_abs_pdgs: tuple[int, ...]
    requires_distinct_mother: bool = True


@dataclass(frozen=True)
class AnalysisModeSpec:
    analysis_mode: str
    fit_branches: tuple[str, str, str]
    ranking_pt_branches: tuple[str, str, str]
    ranking_tiebreak_branches: tuple[str, ...]
    muon_slots: tuple[tuple[str, str], ...]
    same_vertex_muon_slots: tuple[str, ...]
    same_vertex_track_branches: tuple[str, ...]
    jpsi_objects: tuple[str, ...]
    ups_objects: tuple[str, ...]
    phi_objects: tuple[str, ...]
    rapidity_objects: tuple[str, ...]
    truth_leg_specs: tuple[TruthLegSpec, ...]
    axes: tuple[AxisSpec, AxisSpec, AxisSpec]
    selectors: tuple[str, ...] = ("all6_same_recVtx", "Pri_fitValid")
    selector_name: str = "all6_same_recVtx"

CORE_DATA_BRANCHES = [
    "evtNum",
    "runNum",
    "lumiNum",
    "Jpsi_1_mu_1_Idx",
    "Jpsi_1_mu_2_Idx",
    "Jpsi_2_mu_1_Idx",
    "Jpsi_2_mu_2_Idx",
    "Phi_K_1_Idx",
    "Phi_K_2_Idx",
    "muGenMatchIdx",
    "muGenMatchSource",
    "muVertexId",
    "muPackedMatchMethod",
    "Phi_K_1_vertexId",
    "Phi_K_2_vertexId",
    "Phi_K_1_genMatchIdx",
    "Phi_K_2_genMatchIdx",
    "Phi_K_1_genMatchSource",
    "Phi_K_2_genMatchSource",
    "Pri_fitValid",
    "Pri_fitPass",
    "Pri_assocPVPass",
    "Pri_trackPVPass",
    "Pri_passAny",
    "Pri_VtxProb",
    "Pri_maxAbsDzPV",
    "Pri_maxAbsDxyPV",
    "Phi_fitPass",
    "Phi_commonAssocPVPass",
    "Phi_commonAssocPVIdx",
    "Phi_trackPVPass",
    "Phi_vertexCriteriaPass",
    "Phi_VtxProb",
    "Phi_maxAbsDzPV",
    "Phi_maxAbsDxyPV",
    "DiOnia_fitValid",
    "DiOnia_fitPass",
    "DiOnia_commonRecVtxPass",
    "DiOnia_passAny",
    "DiOnia_VtxProb",
    "Jpsi_1_VtxProb",
    "Jpsi_2_VtxProb",
    "Ups_VtxProb",
    "MC_GenPart_pdgId",
    "MC_GenPart_motherGenIdx",
    "MC_GenPart_motherPdgId",
]

TRUTH_MATCH_BRANCHES = CORE_DATA_BRANCHES

def _reso_family(prefix: str, include_mass_diff: bool = True) -> list[str]:
    fields = [f"{prefix}_mass", f"{prefix}_massErr"]
    if include_mass_diff:
        fields.append(f"{prefix}_massDiff")
    fields.extend(
        [
            f"{prefix}_ctau",
            f"{prefix}_ctauErr",
            f"{prefix}_Chi2",
            f"{prefix}_ndof",
            f"{prefix}_VtxProb",
            f"{prefix}_px",
            f"{prefix}_py",
            f"{prefix}_pz",
            f"{prefix}_phi",
            f"{prefix}_eta",
            f"{prefix}_pt",
            f"{prefix}_pxErr",
            f"{prefix}_pyErr",
            f"{prefix}_pzErr",
            f"{prefix}_ptErr",
        ]
    )
    return fields


MASS_SELECTION_CORE_BRANCHES = [
    "evtNum",
    "runNum",
    "lumiNum",
    "Jpsi_1_mu_1_Idx", "Jpsi_1_mu_2_Idx",
    "Jpsi_2_mu_1_Idx", "Jpsi_2_mu_2_Idx",
    "Ups_mu_1_Idx", "Ups_mu_2_Idx",
    "Phi_K_1_Idx", "Phi_K_2_Idx",
]

FIT_INPUT_CANDIDATE_BRANCHES = [
    *_reso_family("Jpsi_1"),
    *_reso_family("Jpsi_2"),
    *_reso_family("Ups"),
    *_reso_family("Phi"),
    *_reso_family("Pri", include_mass_diff=False),
    "Phi_fitPass",
    "Phi_commonAssocPVPass",
    "Phi_commonAssocPVIdx",
    "Phi_trackPVPass",
    "Phi_vertexCriteriaPass",
    "Phi_maxAbsDzPV",
    "Phi_maxAbsDxyPV",
    "Pri_fitValid",
    "Pri_fitPass",
    "Pri_assocPVPass",
    "Pri_assocPVIdx",
    "Pri_trackPVPass",
    "Pri_passAny",
    "Pri_maxAbsDzPV",
    "Pri_maxAbsDxyPV",
    "DiOnia_fitValid",
    "DiOnia_fitPass",
    "DiOnia_commonRecVtxPass",
    "DiOnia_commonRecVtxIdx",
    "DiOnia_passAny",
    "DiOnia_Chi2",
    "DiOnia_ndof",
    "DiOnia_VtxProb",
    "Phi_K_1_px",
    "Phi_K_1_py",
    "Phi_K_1_pz",
    "Phi_K_1_phi",
    "Phi_K_1_eta",
    "Phi_K_1_pt",
    "Phi_K_1_fromPV",
    "Phi_K_1_pvAssocQuality",
    "Phi_K_1_hasAssocPV",
    "Phi_K_1_passDzPV",
    "Phi_K_1_passDxyPV",
    "Phi_K_1_passTrackPV",
    "Phi_K_1_vertexId",
    "Phi_K_1_dzPV",
    "Phi_K_1_dxyPV",
    "Phi_K_1_dzAssocPV",
    "Phi_K_1_dxyAssocPV",
    "Phi_K_1_genMatchIdx",
    "Phi_K_1_genMatchSource",
    "Phi_K_1_genMatchChi2",
    "Phi_K_2_px",
    "Phi_K_2_py",
    "Phi_K_2_pz",
    "Phi_K_2_phi",
    "Phi_K_2_eta",
    "Phi_K_2_pt",
    "Phi_K_2_fromPV",
    "Phi_K_2_pvAssocQuality",
    "Phi_K_2_hasAssocPV",
    "Phi_K_2_passDzPV",
    "Phi_K_2_passDxyPV",
    "Phi_K_2_passTrackPV",
    "Phi_K_2_vertexId",
    "Phi_K_2_dzPV",
    "Phi_K_2_dxyPV",
    "Phi_K_2_dzAssocPV",
    "Phi_K_2_dxyAssocPV",
    "Phi_K_2_genMatchIdx",
    "Phi_K_2_genMatchSource",
    "Phi_K_2_genMatchChi2",
]

FIT_INPUT_MUON_VECTOR_BRANCH_OUTPUTS = {
    "muPx": "px",
    "muPy": "py",
    "muPz": "pz",
    "muCharge": "charge",
    "muIsGoodSoftMuonNewIlseMod": "isGoodSoftMuonNewIlseMod",
    "muIsGoodSoftMuonNewIlse": "isGoodSoftMuonNewIlse",
    "muIsGoodLooseMuonNew": "isGoodLooseMuonNew",
    "muIsGoodLooseMuon": "isGoodLooseMuon",
    "muIsGoodTightMuon": "isGoodTightMuon",
    "muIsGlobalMuon": "isGlobalMuon",
    "muIsPatLooseMuon": "isPatLooseMuon",
    "muIsPatTightMuon": "isPatTightMuon",
    "muIsPatSoftMuon": "isPatSoftMuon",
    "muIsPatMediumMuon": "isPatMediumMuon",
    "muFromPV": "fromPV",
    "muPVAssocQuality": "pvAssocQuality",
    "muPxErr": "pxErr",
    "muPyErr": "pyErr",
    "muPzErr": "pzErr",
    "muPtErr": "ptErr",
    "muVertexId": "vertexId",
    "muDzAssocPV": "dzAssocPV",
    "muDxyAssocPV": "dxyAssocPV",
    "muFromPVAssocPV": "fromPVAssocPV",
    "muPackedMatchIdx": "packedMatchIdx",
    "muPackedMatchMethod": "packedMatchMethod",
    "muPackedMatchVectorRelP": "packedMatchVectorRelP",
    "muPackedMatchChi2": "packedMatchChi2",
    "muPackedMatchDzPV": "packedMatchDzPV",
    "muPackedMatchDzAssocPV": "packedMatchDzAssocPV",
    "muGenMatchIdx": "genMatchIdx",
    "muGenMatchSource": "genMatchSource",
    "muGenMatchChi2": "genMatchChi2",
    "muMVAMuonID": "MVAMuonID",
    "musegmentCompatibility": "segmentCompatibility",
}

FIT_INPUT_MUON_VECTOR_BRANCHES = list(FIT_INPUT_MUON_VECTOR_BRANCH_OUTPUTS)

FIT_INPUT_MUON_SLOTS = (
    ("Jpsi_1_mu_1", "Jpsi_1_mu_1_Idx"),
    ("Jpsi_1_mu_2", "Jpsi_1_mu_2_Idx"),
    ("Jpsi_2_mu_1", "Jpsi_2_mu_1_Idx"),
    ("Jpsi_2_mu_2", "Jpsi_2_mu_2_Idx"),
    ("Ups_mu_1", "Ups_mu_1_Idx"),
    ("Ups_mu_2", "Ups_mu_2_Idx"),
)

MASS_STUDY_BRANCHES = sorted(set(MASS_SELECTION_CORE_BRANCHES + FIT_INPUT_CANDIDATE_BRANCHES + FIT_INPUT_MUON_VECTOR_BRANCHES))

RECO_SELECTION_BRANCHES = MASS_STUDY_BRANCHES

DETAIL_ONLY_BRANCHES = [
    "muPx",
    "muPy",
    "muPz",
    "muFromPV",
    "muPVAssocQuality",
    "muPackedMatchIdx",
    "muPackedMatchVectorRelP",
    "muPackedMatchChi2",
    "muPackedMatchDzPV",
    "muPackedMatchDzAssocPV",
    "Phi_K_1_pt",
    "Phi_K_2_pt",
    "Phi_K_1_eta",
    "Phi_K_2_eta",
    "Phi_K_1_phi",
    "Phi_K_2_phi",
    "Phi_K_1_fromPV",
    "Phi_K_2_fromPV",
    "Phi_K_1_pvAssocQuality",
    "Phi_K_2_pvAssocQuality",
    "Phi_K_1_dzPV",
    "Phi_K_2_dzPV",
    "Phi_K_1_dxyPV",
    "Phi_K_2_dxyPV",
    "Phi_K_1_dzAssocPV",
    "Phi_K_2_dzAssocPV",
    "Phi_K_1_dxyAssocPV",
    "Phi_K_2_dxyAssocPV",
    "MC_GenPart_status",
    "MC_GenPart_handleIndex",
    "MC_GenPart_pt",
    "MC_GenPart_eta",
    "MC_GenPart_phi",
    "MC_GenPart_mass",
]

DETAIL_DATA_BRANCHES = sorted(set(CORE_DATA_BRANCHES + DETAIL_ONLY_BRANCHES))
DETAIL_ENTRY_BRANCHES = DETAIL_DATA_BRANCHES

MASS_BRANCHES_BY_MODE = {
    "JpsiJpsiPhi": ["Jpsi_1_mass", "Jpsi_2_mass", "Phi_mass", "Pri_mass"],
    "JpsiUpsPhi": ["Jpsi_1_mass", "Ups_mass", "Phi_mass", "Pri_mass"],
    "JpsiJpsiUps": ["Jpsi_1_mass", "Jpsi_2_mass", "Ups_mass", "Pri_mass"],
}

FIT_BRANCHES_BY_MODE = {
    "JpsiJpsiPhi": ["Jpsi_1_mass", "Jpsi_2_mass", "Phi_mass"],
    "JpsiUpsPhi": ["Jpsi_1_mass", "Ups_mass", "Phi_mass"],
    "JpsiJpsiUps": ["Jpsi_1_mass", "Jpsi_2_mass", "Ups_mass"],
}

UPSILON_TRUTH_ABS_PDGS = (553, 100553, 200553)


ANALYSIS_MODE_SPECS = {
    "JpsiJpsiPhi": AnalysisModeSpec(
        analysis_mode="JpsiJpsiPhi",
        fit_branches=("Jpsi_1_mass", "Jpsi_2_mass", "Phi_mass"),
        ranking_pt_branches=("Jpsi_1_pt", "Jpsi_2_pt", "Phi_pt"),
        ranking_tiebreak_branches=("Pri_VtxProb", "Phi_VtxProb", "cand_idx"),
        muon_slots=(
            ("Jpsi_1_mu_1", "Jpsi_1_mu_1_Idx"),
            ("Jpsi_1_mu_2", "Jpsi_1_mu_2_Idx"),
            ("Jpsi_2_mu_1", "Jpsi_2_mu_1_Idx"),
            ("Jpsi_2_mu_2", "Jpsi_2_mu_2_Idx"),
        ),
        same_vertex_muon_slots=("Jpsi_1_mu_1", "Jpsi_1_mu_2", "Jpsi_2_mu_1", "Jpsi_2_mu_2"),
        same_vertex_track_branches=("Phi_K_1_vertexId", "Phi_K_2_vertexId"),
        jpsi_objects=("Jpsi_1", "Jpsi_2"),
        ups_objects=(),
        phi_objects=("Phi",),
        rapidity_objects=("Jpsi_1", "Jpsi_2"),
        truth_leg_specs=(
            TruthLegSpec("Jpsi_1", "jpsi", ("Jpsi_1_mu_1", "Jpsi_1_mu_2"), (443,)),
            TruthLegSpec("Jpsi_2", "jpsi", ("Jpsi_2_mu_1", "Jpsi_2_mu_2"), (443,)),
            TruthLegSpec("Phi", "phi", ("Phi_K_1", "Phi_K_2"), (333,)),
        ),
        axes=(
            AxisSpec("Jpsi_1_mass", "jpsi", "Jpsi_1"),
            AxisSpec("Jpsi_2_mass", "jpsi", "Jpsi_2"),
            AxisSpec("Phi_mass", "phi", "Phi"),
        ),
    ),
    "JpsiUpsPhi": AnalysisModeSpec(
        analysis_mode="JpsiUpsPhi",
        fit_branches=("Jpsi_1_mass", "Ups_mass", "Phi_mass"),
        ranking_pt_branches=("Jpsi_1_pt", "Ups_pt", "Phi_pt"),
        ranking_tiebreak_branches=("Pri_VtxProb", "Phi_VtxProb", "Ups_VtxProb", "cand_idx"),
        muon_slots=(
            ("Jpsi_1_mu_1", "Jpsi_1_mu_1_Idx"),
            ("Jpsi_1_mu_2", "Jpsi_1_mu_2_Idx"),
            ("Ups_mu_1", "Ups_mu_1_Idx"),
            ("Ups_mu_2", "Ups_mu_2_Idx"),
        ),
        same_vertex_muon_slots=("Jpsi_1_mu_1", "Jpsi_1_mu_2", "Ups_mu_1", "Ups_mu_2"),
        same_vertex_track_branches=("Phi_K_1_vertexId", "Phi_K_2_vertexId"),
        jpsi_objects=("Jpsi_1",),
        ups_objects=("Ups",),
        phi_objects=("Phi",),
        rapidity_objects=("Jpsi_1", "Ups"),
        truth_leg_specs=(
            TruthLegSpec("Jpsi_1", "jpsi", ("Jpsi_1_mu_1", "Jpsi_1_mu_2"), (443,)),
            TruthLegSpec("Ups", "ups", ("Ups_mu_1", "Ups_mu_2"), UPSILON_TRUTH_ABS_PDGS),
            TruthLegSpec("Phi", "phi", ("Phi_K_1", "Phi_K_2"), (333,)),
        ),
        axes=(
            AxisSpec("Jpsi_1_mass", "jpsi", "Jpsi_1"),
            AxisSpec("Ups_mass", "ups", "Ups"),
            AxisSpec("Phi_mass", "phi", "Phi"),
        ),
    ),
    "JpsiJpsiUps": AnalysisModeSpec(
        analysis_mode="JpsiJpsiUps",
        fit_branches=("Jpsi_1_mass", "Jpsi_2_mass", "Ups_mass"),
        ranking_pt_branches=("Jpsi_1_pt", "Jpsi_2_pt", "Ups_pt"),
        ranking_tiebreak_branches=("Pri_VtxProb", "Ups_VtxProb", "DiOnia_VtxProb", "cand_idx"),
        muon_slots=(
            ("Jpsi_1_mu_1", "Jpsi_1_mu_1_Idx"),
            ("Jpsi_1_mu_2", "Jpsi_1_mu_2_Idx"),
            ("Jpsi_2_mu_1", "Jpsi_2_mu_1_Idx"),
            ("Jpsi_2_mu_2", "Jpsi_2_mu_2_Idx"),
            ("Ups_mu_1", "Ups_mu_1_Idx"),
            ("Ups_mu_2", "Ups_mu_2_Idx"),
        ),
        same_vertex_muon_slots=("Jpsi_1_mu_1", "Jpsi_1_mu_2", "Jpsi_2_mu_1", "Jpsi_2_mu_2", "Ups_mu_1", "Ups_mu_2"),
        same_vertex_track_branches=(),
        jpsi_objects=("Jpsi_1", "Jpsi_2"),
        ups_objects=("Ups",),
        phi_objects=(),
        rapidity_objects=("Jpsi_1", "Jpsi_2", "Ups"),
        truth_leg_specs=(
            TruthLegSpec("Jpsi_1", "jpsi", ("Jpsi_1_mu_1", "Jpsi_1_mu_2"), (443,)),
            TruthLegSpec("Jpsi_2", "jpsi", ("Jpsi_2_mu_1", "Jpsi_2_mu_2"), (443,)),
            TruthLegSpec("Ups", "ups", ("Ups_mu_1", "Ups_mu_2"), UPSILON_TRUTH_ABS_PDGS),
        ),
        axes=(
            AxisSpec("Jpsi_1_mass", "jpsi", "Jpsi_1"),
            AxisSpec("Jpsi_2_mass", "jpsi", "Jpsi_2"),
            AxisSpec("Ups_mass", "ups", "Ups"),
        ),
    ),
}


def get_analysis_mode_spec(analysis_mode: str) -> AnalysisModeSpec:
    try:
        return ANALYSIS_MODE_SPECS[analysis_mode]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported analysis mode '{analysis_mode}'. Expected one of {sorted(ANALYSIS_MODE_SPECS)}."
        ) from exc

DEFAULT_SELECTOR_NAMES = ("all6_same_recVtx", "Pri_fitValid")

CONFIG_BRANCHES = [
    "AnalysisMode",
    "DoMonteCarloTree",
    "RequireAcceptedCandidatesForMonteCarloTree",
    "DoJPsiMassConstraint",
    "Debug_Output",
    "DebugMask",
    "MuonSelection",
    "TrackSelection",
    "PVNdofMin",
    "PVMaxAbsZ",
    "PVMaxRho",
    "JpsiMassMin",
    "JpsiMassMax",
    "UpsMassMin",
    "UpsMassMax",
    "PhiMassMin",
    "PhiMassMax",
    "TrackPtMin",
    "TrackDRMax",
    "LegacyOniaDecayVtxProbCut",
    "JpsiDecayVtxProbCut",
    "UpsDecayVtxProbCut",
    "PhiDecayVtxProbCut",
    "DiOniaVtxProbCut",
    "PriVtxProbCut",
    "DoJpsiDecayVtxFit",
    "DoUpsDecayVtxFit",
    "DoPhiDecayVtxFit",
    "DoDiOniaVtxFit",
    "DoPriVtxFit",
    "PriRequireCommonAssocPV",
    "PriRequireTrackPVCompatibility",
    "PriTrackDzPVMax",
    "PriTrackDxyPVMax",
    "CheckFinalMass",
    "PVSelectionMode",
    "MinTrackFromPV",
    "MinMuonCount",
    "MuTrkMatchMethod",
    "MuTrkMatchDebug",
    "MuonPackedMatchVectorRelPMax",
    "MuonPackedMatchChi2Max",
    "MuonPackedMatchDzPvChi2Max",
    "MuonPackedMatchDzAssocChi2Max",
    "RecoGenMuonMatchChi2Max",
    "RecoGenKaonMatchChi2Max",
]

CLASSIFIER_SPECS = [
    ("Pri_fitValid", "Pri_fitValid"),
    ("Pri_fitPass", "Pri_fitPass"),
    ("Pri_assocPVPass", "Pri_assocPVPass"),
    ("Pri_trackPVPass", "Pri_trackPVPass"),
    ("Pri_passAny", "Pri_passAny"),
    ("DiOnia_fitValid", "DiOnia_fitValid"),
    ("DiOnia_fitPass", "DiOnia_fitPass"),
    ("DiOnia_commonRecVtxPass", "DiOnia_commonRecVtxPass"),
    ("Phi_fitPass", "Phi_fitPass"),
    ("Phi_commonAssocPVPass", "Phi_commonAssocPVPass"),
    ("Phi_trackPVPass", "Phi_trackPVPass"),
    ("Phi_vertexCriteriaPass", "Phi_vertexCriteriaPass"),
    ("muons_same_recVtx", "muons_same_recVtx"),
    ("all6_same_recVtx", "all6_same_recVtx"),
    ("phi_same_recVtx_as_all_muons", "phi_same_recVtx_as_all_muons"),
    ("clf_phi_and_pri_fit", "Phi_vertexCriteriaPass & Pri_fitPass"),
    ("clf_strict_hybrid", "Phi_vertexCriteriaPass & Pri_fitPass & all6_same_recVtx"),
]

CACHE_FILENAMES = {
    "candidate_df": "candidate_rows.parquet",
    "event_df": "event_rows.parquet",
    "config_df": "config_rows.parquet",
}

MASS_SELECTION_FILENAMES = {
    "candidate_pool_df": "candidate_pool_df.parquet",
    "selected_candidate_df": "selected_candidate_df.parquet",
    "audit_df": "audit_df.parquet",
    "window_audit_df": "window_audit_df.parquet",
    "selection_summary_df": "selection_summary_df.parquet",
    "selected_for_selector_df": "selected_for_selector_df.parquet",
    "fit_df": "fit_df.parquet",
}
