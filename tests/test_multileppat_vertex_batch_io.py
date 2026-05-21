from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import awkward as ak
import pandas as pd
import uproot

from multileppat_vertex_batch.cache import (
    load_mass_selection_bundle_if_compatible,
    stage_cache_matches,
    write_mass_selection_bundle,
)
from multileppat_vertex_batch.config import OfflineSelectionConfig
from multileppat_vertex_batch.efficiency import (
    EfficiencyBinning,
    build_cutflow,
    build_efficiency_counts,
    build_event_efficiency_row,
    clopper_pearson_interval,
    find_jpsijpsiphi_gen_system,
)
from multileppat_vertex_batch.fit_iminuit import minuit_parameter_table
from multileppat_vertex_batch.fit_roofit import (
    build_ups_peak_significance_table,
    one_sided_profile_significance,
    roofit_parameter_table,
    ups_signal_fractions,
    ups_signal_yields,
)
from multileppat_vertex_batch.io import read_json, resolve_input_files, stable_data_hash, write_root_trees
from multileppat_vertex_batch.selection import select_best_candidates
from multileppat_vertex_batch.schema import MASS_SELECTION_CACHE_VERSION, get_analysis_mode_spec
from multileppat_vertex_batch.truth import build_file_records, first_ancestor_idx


class ResolveInputFilesTest(unittest.TestCase):
    def test_resolve_input_files_deduplicates_and_natural_sorts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_io_", dir="/tmp/chiw") as tmp_dir:
            tmp_path = Path(tmp_dir)
            for name in ("sample_10.root", "sample_2.root", "sample_1.root"):
                (tmp_path / name).write_text("", encoding="utf-8")

            resolved = resolve_input_files(
                [
                    str(tmp_path / "sample_10.root"),
                    str(tmp_path / "sample_*.root"),
                    str(tmp_path / "sample_2.root"),
                ]
            )

            self.assertEqual(
                [path.name for path in resolved],
                ["sample_1.root", "sample_2.root", "sample_10.root"],
            )

    def test_resolve_input_files_raises_on_missing_literal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_io_", dir="/tmp/chiw") as tmp_dir:
            missing = Path(tmp_dir) / "missing.root"
            with self.assertRaises(FileNotFoundError):
                resolve_input_files([str(missing)])


class WriteRootTreesTest(unittest.TestCase):
    def test_write_root_trees_persists_scalar_and_string_columns(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_root_", dir="/tmp/chiw") as tmp_dir:
            output_path = Path(tmp_dir) / "fit_input_candidates.root"
            frame = pd.DataFrame(
                {
                    "cand_idx": [4, 7],
                    "Jpsi_1_mass": [3.09, 3.10],
                    "selector": ["all6_same_recVtx", "Pri_fitValid"],
                    "is_best": [True, False],
                }
            )

            write_root_trees(output_path, {"fit_input_all6_same_recVtx": frame})

            with uproot.open(output_path) as root_file:
                self.assertIn("fit_input_all6_same_recVtx;1", root_file.keys())
                tree = root_file["fit_input_all6_same_recVtx"]
                arrays = tree.arrays(library="np")

            self.assertEqual(arrays["cand_idx"].tolist(), [4, 7])
            self.assertAlmostEqual(arrays["Jpsi_1_mass"][0], 3.09, places=6)
            self.assertEqual(arrays["selector"].tolist(), ["all6_same_recVtx", "Pri_fitValid"])
            self.assertEqual(arrays["is_best"].tolist(), [1, 0])

    def test_write_root_trees_preserves_explicit_muon_slot_names(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_root_", dir="/tmp/chiw") as tmp_dir:
            output_path = Path(tmp_dir) / "fit_input_candidates.root"
            frame = pd.DataFrame(
                {
                    "Jpsi_1_mu_1_isPatSoftMuon": [1],
                    "Jpsi_1_mu_2_isPatMediumMuon": [0],
                    "Jpsi_2_mu_1_fromPV": [2],
                    "Jpsi_2_mu_2_packedMatchMethod": [4],
                    "Jpsi_1_ctau": [0.012],
                }
            )

            write_root_trees(output_path, {"fit_input_all6_same_recVtx": frame})

            with uproot.open(output_path) as root_file:
                tree = root_file["fit_input_all6_same_recVtx"]
                arrays = tree.arrays(library="np")

            self.assertIn("Jpsi_1_mu_1_isPatSoftMuon", arrays)
            self.assertIn("Jpsi_2_mu_2_packedMatchMethod", arrays)
            self.assertNotIn("mu1_soft", arrays)
            self.assertAlmostEqual(arrays["Jpsi_1_ctau"][0], 0.012, places=6)

    def test_write_root_trees_preserves_explicit_upsilon_slot_names(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_root_", dir="/tmp/chiw") as tmp_dir:
            output_path = Path(tmp_dir) / "fit_input_candidates.root"
            frame = pd.DataFrame(
                {
                    "Ups_mass": [9.46],
                    "Ups_mu_1_isPatSoftMuon": [1],
                    "Ups_mu_2_packedMatchMethod": [3],
                }
            )

            write_root_trees(output_path, {"fit_input_all6_same_recVtx": frame})

            with uproot.open(output_path) as root_file:
                arrays = root_file["fit_input_all6_same_recVtx"].arrays(library="np")

            self.assertIn("Ups_mu_1_isPatSoftMuon", arrays)
            self.assertIn("Ups_mu_2_packedMatchMethod", arrays)
            self.assertAlmostEqual(arrays["Ups_mass"][0], 9.46, places=6)

    def test_write_root_trees_skips_empty_frames_without_failing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_root_", dir="/tmp/chiw") as tmp_dir:
            output_path = Path(tmp_dir) / "empty_fit_input.root"
            empty_frame = pd.DataFrame(columns=["cand_idx", "selector"])

            write_root_trees(output_path, {"fit_input_empty": empty_frame})

            with uproot.open(output_path) as root_file:
                self.assertEqual(root_file.keys(), [])


class CacheHelpersTest(unittest.TestCase):
    def test_stable_data_hash_is_order_insensitive_for_dict_keys(self) -> None:
        self.assertEqual(
            stable_data_hash({"b": 2, "a": 1}),
            stable_data_hash({"a": 1, "b": 2}),
        )

    def test_mass_selection_bundle_roundtrips_when_cache_matches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_cache_", dir="/tmp/chiw") as tmp_dir:
            output_dir = Path(tmp_dir)
            payload = {"input_files": [{"path": "/tmp/sample.root", "size": 10, "mtime_ns": 20}], "selector_name": "all6_same_recVtx"}
            tables = {
                "candidate_pool_df": pd.DataFrame({"selector": ["all6_same_recVtx"], "cand_idx": [3]}),
                "selected_candidate_df": pd.DataFrame({"selector": ["all6_same_recVtx"], "cand_idx": [3]}),
                "audit_df": pd.DataFrame({"selector": ["all"], "stage": ["initial_candidates"], "count": [1]}),
                "window_audit_df": pd.DataFrame({"selector": ["all"], "stage": ["initial_candidates"], "count": [1]}),
                "selection_summary_df": pd.DataFrame({"selector": ["all6_same_recVtx"], "n_total_candidates": [1]}),
                "selected_for_selector_df": pd.DataFrame({"selector": ["all6_same_recVtx"], "cand_idx": [3]}),
                "fit_df": pd.DataFrame({"Jpsi_1_mass": [3.09], "Jpsi_2_mass": [3.10], "Phi_mass": [1.019]}),
                "fit_input_dfs_by_selector": {
                    "all6_same_recVtx": pd.DataFrame({"Jpsi_1_ctau": [0.01], "Jpsi_1_mu_1_isPatSoftMuon": [1]})
                },
            }

            write_mass_selection_bundle(
                output_dir,
                tables,
                cache_version=MASS_SELECTION_CACHE_VERSION,
                cache_payload=payload,
            )

            self.assertTrue(stage_cache_matches(output_dir, "mass_selection", MASS_SELECTION_CACHE_VERSION, payload))
            cache_meta = read_json(output_dir / "cache_meta.json")
            self.assertEqual(cache_meta["stage"], "mass_selection")

            loaded = load_mass_selection_bundle_if_compatible(
                output_dir,
                MASS_SELECTION_CACHE_VERSION,
                payload,
            )
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["selected_candidate_df"]["cand_idx"].tolist(), [3])
            self.assertEqual(loaded["fit_input_dfs_by_selector"], {})

    def test_mass_selection_bundle_is_rejected_when_cache_payload_changes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="multileppat_cache_", dir="/tmp/chiw") as tmp_dir:
            output_dir = Path(tmp_dir)
            payload = {"selector_name": "all6_same_recVtx"}
            tables = {
                "candidate_pool_df": pd.DataFrame(columns=["selector"]),
                "selected_candidate_df": pd.DataFrame(columns=["selector"]),
                "audit_df": pd.DataFrame(columns=["selector", "stage", "count"]),
                "window_audit_df": pd.DataFrame(columns=["selector", "stage", "count"]),
                "selection_summary_df": pd.DataFrame(columns=["selector"]),
                "selected_for_selector_df": pd.DataFrame(columns=["selector"]),
                "fit_df": pd.DataFrame(columns=["Jpsi_1_mass", "Jpsi_2_mass", "Phi_mass"]),
            }
            write_mass_selection_bundle(
                output_dir,
                tables,
                cache_version=MASS_SELECTION_CACHE_VERSION,
                cache_payload=payload,
            )

            loaded = load_mass_selection_bundle_if_compatible(
                output_dir,
                MASS_SELECTION_CACHE_VERSION,
                {"selector_name": "Pri_fitValid"},
            )
            self.assertIsNone(loaded)


class SignificanceHelpersTest(unittest.TestCase):
    def test_one_sided_profile_significance_is_zero_for_nonpositive_signal(self) -> None:
        result = one_sided_profile_significance(full_nll=100.0, null_nll=104.5, signal_yield=0.0)
        self.assertEqual(result["N_sss_significance_sigma"], 0.0)
        self.assertGreaterEqual(result["q0"], 0.0)

    def test_one_sided_profile_significance_matches_simple_profile_case(self) -> None:
        result = one_sided_profile_significance(full_nll=100.0, null_nll=104.5, signal_yield=12.0)
        self.assertAlmostEqual(result["delta_nll"], 4.5, places=6)
        self.assertAlmostEqual(result["q0"], 9.0, places=6)
        self.assertAlmostEqual(result["N_sss_significance_sigma"], 3.0, places=6)

    def test_ups_signal_fractions_and_yields_match_parameterization(self) -> None:
        fractions = ups_signal_fractions(0.30, 0.25)
        assert fractions is not None
        self.assertAlmostEqual(fractions["ups_1s_fraction"], 0.70, places=6)
        self.assertAlmostEqual(fractions["ups_2s_fraction"], 0.225, places=6)
        self.assertAlmostEqual(fractions["ups_3s_fraction"], 0.075, places=6)
        self.assertAlmostEqual(fractions["ups_excited_fraction"], 0.30, places=6)

        yields = ups_signal_yields(40.0, 0.30, 0.25)
        assert yields is not None
        self.assertAlmostEqual(yields["ups_2s_signal_yield"], 9.0, places=6)
        self.assertAlmostEqual(yields["ups_3s_signal_yield"], 3.0, places=6)
        self.assertAlmostEqual(yields["ups_excited_signal_yield"], 12.0, places=6)

    def test_build_ups_peak_significance_table_reports_component_metrics(self) -> None:
        table = build_ups_peak_significance_table(
            full_nll=100.0,
            full_fit_valid=True,
            total_signal_yield=20.0,
            frac_excited=0.30,
            frac_3s_in_excited=0.25,
            null_results={
                "Ups_2S": {"null_hypothesis": "no 2S model", "nll": 102.0, "fit_valid": True},
                "Ups_3S": {"null_hypothesis": "no 3S model", "nll": 101.125, "fit_valid": True},
                "Ups_excited": {"null_hypothesis": "no 2S or 3S model", "nll": 104.5, "fit_valid": True},
            },
        )
        self.assertEqual(table["component"].tolist(), ["Ups_2S", "Ups_3S", "Ups_excited"])
        self.assertAlmostEqual(table.loc[table["component"] == "Ups_2S", "signal_yield"].iloc[0], 4.5, places=6)
        self.assertAlmostEqual(table.loc[table["component"] == "Ups_3S", "signal_yield"].iloc[0], 1.5, places=6)
        self.assertAlmostEqual(table.loc[table["component"] == "Ups_excited", "significance_sigma"].iloc[0], 3.0, places=6)


class AnalysisModeSpecTest(unittest.TestCase):
    def test_jpsiupsphi_uses_ups_branch_as_second_fit_axis(self) -> None:
        spec = get_analysis_mode_spec("JpsiUpsPhi")
        self.assertEqual(spec.fit_branches, ("Jpsi_1_mass", "Ups_mass", "Phi_mass"))
        self.assertNotIn(("Jpsi_2_mu_1", "Jpsi_2_mu_1_Idx"), spec.muon_slots)
        self.assertIn(("Ups_mu_1", "Ups_mu_1_Idx"), spec.muon_slots)
        self.assertEqual(spec.ranking_tiebreak_branches, ("Pri_VtxProb", "Phi_VtxProb", "Ups_VtxProb", "cand_idx"))

    def test_jpsijpsiups_all6_same_vertex_uses_six_muons(self) -> None:
        spec = get_analysis_mode_spec("JpsiJpsiUps")
        self.assertEqual(len(spec.same_vertex_muon_slots), 6)
        self.assertEqual(spec.same_vertex_track_branches, ())
        self.assertEqual(spec.ranking_tiebreak_branches, ("Pri_VtxProb", "Ups_VtxProb", "DiOnia_VtxProb", "cand_idx"))


class SelectionHelpersTest(unittest.TestCase):
    def test_select_best_candidates_uses_mode_specific_tiebreakers(self) -> None:
        spec = get_analysis_mode_spec("JpsiJpsiUps")
        candidate_pool_df = pd.DataFrame(
            {
                "selector": ["all6_same_recVtx", "all6_same_recVtx"],
                "source_file": ["sample.root", "sample.root"],
                "run": [1, 1],
                "lumi": [1, 1],
                "event": [1, 1],
                "cand_idx": [1, 0],
                "triple_pt2_sum": [100.0, 100.0],
                "Pri_VtxProb": [0.20, 0.20],
                "Ups_VtxProb": [0.40, 0.80],
                "DiOnia_VtxProb": [0.70, 0.20],
            }
        )

        selected = select_best_candidates(candidate_pool_df, spec)

        self.assertEqual(selected["cand_idx"].tolist(), [0])
        self.assertEqual(selected["n_passing_candidates"].tolist(), [2])


class TruthHelpersTest(unittest.TestCase):
    def test_first_ancestor_idx_accepts_excited_upsilon_states(self) -> None:
        result = first_ancestor_idx([13, 100553], [1, -1], 0, (553, 100553, 200553))
        self.assertEqual(result, 1)

    def test_build_file_records_accepts_upsilon_2s_in_jpsiupsphi(self) -> None:
        arrays = {
            "evtNum": ak.Array([11]),
            "runNum": ak.Array([7]),
            "lumiNum": ak.Array([3]),
            "Jpsi_1_mu_1_Idx": ak.Array([[0]]),
            "Jpsi_1_mu_2_Idx": ak.Array([[1]]),
            "Ups_mu_1_Idx": ak.Array([[2]]),
            "Ups_mu_2_Idx": ak.Array([[3]]),
            "Phi_K_1_Idx": ak.Array([[0]]),
            "Phi_K_2_Idx": ak.Array([[1]]),
            "muGenMatchIdx": ak.Array([[0, 1, 2, 3]]),
            "muGenMatchSource": ak.Array([[1, 1, 1, 1]]),
            "muVertexId": ak.Array([[5, 5, 5, 5]]),
            "muPackedMatchMethod": ak.Array([[1, 1, 1, 1]]),
            "Phi_K_1_vertexId": ak.Array([[5]]),
            "Phi_K_2_vertexId": ak.Array([[5]]),
            "Phi_K_1_genMatchIdx": ak.Array([[4]]),
            "Phi_K_2_genMatchIdx": ak.Array([[5]]),
            "Phi_K_1_genMatchSource": ak.Array([[1]]),
            "Phi_K_2_genMatchSource": ak.Array([[1]]),
            "MC_GenPart_pdgId": ak.Array([[13, -13, 13, -13, 321, -321, 443, 100553, 333]]),
            "MC_GenPart_motherGenIdx": ak.Array([[6, 6, 7, 7, 8, 8, -1, -1, -1]]),
            "Pri_fitValid": ak.Array([[1]]),
            "Pri_fitPass": ak.Array([[1]]),
            "Pri_assocPVPass": ak.Array([[1]]),
            "Pri_trackPVPass": ak.Array([[1]]),
            "Pri_passAny": ak.Array([[1]]),
            "Pri_VtxProb": ak.Array([[0.9]]),
            "Pri_maxAbsDzPV": ak.Array([[0.0]]),
            "Pri_maxAbsDxyPV": ak.Array([[0.0]]),
            "Phi_fitPass": ak.Array([[1]]),
            "Phi_commonAssocPVPass": ak.Array([[1]]),
            "Phi_commonAssocPVIdx": ak.Array([[0]]),
            "Phi_trackPVPass": ak.Array([[1]]),
            "Phi_vertexCriteriaPass": ak.Array([[1]]),
            "Phi_VtxProb": ak.Array([[0.8]]),
            "Phi_maxAbsDzPV": ak.Array([[0.0]]),
            "Phi_maxAbsDxyPV": ak.Array([[0.0]]),
            "DiOnia_fitValid": ak.Array([[1]]),
            "DiOnia_fitPass": ak.Array([[1]]),
            "DiOnia_commonRecVtxPass": ak.Array([[1]]),
            "DiOnia_passAny": ak.Array([[1]]),
            "DiOnia_VtxProb": ak.Array([[0.7]]),
            "Jpsi_1_VtxProb": ak.Array([[0.6]]),
            "Ups_VtxProb": ak.Array([[0.5]]),
        }

        candidate_rows, event_rows = build_file_records(arrays, "sample.root", "JpsiUpsPhi")

        self.assertEqual(len(candidate_rows), 1)
        self.assertEqual(candidate_rows[0]["truth_triple_strict"], 1)
        self.assertEqual(candidate_rows[0]["ups_pair_consistent"], 1)
        self.assertEqual(candidate_rows[0]["phi_pair_consistent"], 1)
        self.assertEqual(candidate_rows[0]["all6_same_recVtx"], 1)
        self.assertEqual(candidate_rows[0]["failure_mode"], "truth_positive")
        self.assertEqual(event_rows[0]["has_truth_triple_strict"], 1)

    def test_build_file_records_accepts_upsilon_3s_in_jpsijpsiups(self) -> None:
        arrays = {
            "evtNum": ak.Array([21]),
            "runNum": ak.Array([8]),
            "lumiNum": ak.Array([4]),
            "Jpsi_1_mu_1_Idx": ak.Array([[0]]),
            "Jpsi_1_mu_2_Idx": ak.Array([[1]]),
            "Jpsi_2_mu_1_Idx": ak.Array([[2]]),
            "Jpsi_2_mu_2_Idx": ak.Array([[3]]),
            "Ups_mu_1_Idx": ak.Array([[4]]),
            "Ups_mu_2_Idx": ak.Array([[5]]),
            "muGenMatchIdx": ak.Array([[0, 1, 2, 3, 4, 5]]),
            "muGenMatchSource": ak.Array([[1, 1, 1, 1, 1, 1]]),
            "muVertexId": ak.Array([[9, 9, 9, 9, 9, 9]]),
            "muPackedMatchMethod": ak.Array([[1, 1, 1, 1, 1, 1]]),
            "MC_GenPart_pdgId": ak.Array([[13, -13, 13, -13, 13, -13, 443, 443, 200553]]),
            "MC_GenPart_motherGenIdx": ak.Array([[6, 6, 7, 7, 8, 8, -1, -1, -1]]),
            "Pri_fitValid": ak.Array([[1]]),
            "Pri_fitPass": ak.Array([[1]]),
            "Pri_assocPVPass": ak.Array([[1]]),
            "Pri_trackPVPass": ak.Array([[1]]),
            "Pri_passAny": ak.Array([[1]]),
            "Pri_VtxProb": ak.Array([[0.7]]),
            "Pri_maxAbsDzPV": ak.Array([[0.0]]),
            "Pri_maxAbsDxyPV": ak.Array([[0.0]]),
            "DiOnia_fitValid": ak.Array([[1]]),
            "DiOnia_fitPass": ak.Array([[1]]),
            "DiOnia_commonRecVtxPass": ak.Array([[1]]),
            "DiOnia_passAny": ak.Array([[1]]),
            "DiOnia_VtxProb": ak.Array([[0.6]]),
            "Jpsi_1_VtxProb": ak.Array([[0.5]]),
            "Jpsi_2_VtxProb": ak.Array([[0.4]]),
            "Ups_VtxProb": ak.Array([[0.3]]),
        }

        candidate_rows, event_rows = build_file_records(arrays, "sample.root", "JpsiJpsiUps")

        self.assertEqual(candidate_rows[0]["truth_triple_strict"], 1)
        self.assertEqual(candidate_rows[0]["ups_pair_consistent"], 1)
        self.assertEqual(candidate_rows[0]["all_k_gen_ok"], 1)
        self.assertEqual(candidate_rows[0]["all6_same_recVtx"], 1)
        self.assertEqual(candidate_rows[0]["mu5_idx"], 4)
        self.assertEqual(candidate_rows[0]["mu6_idx"], 5)
        self.assertEqual(event_rows[0]["has_truth_triple_strict"], 1)


class EfficiencyHelpersTest(unittest.TestCase):
    def _jpsijpsiphi_event(self) -> dict:
        return {
            "evtNum": 101,
            "runNum": 1,
            "lumiNum": 2,
            "TrigNames": ["HLT_DoubleMu4_3_LowMass_v1"],
            "TrigRes": [1],
            "MC_GenPart_pdgId": [443, 443, 333, 13, -13, 13, -13, 321, -321],
            "MC_GenPart_motherGenIdx": [-1, -1, -1, 0, 0, 1, 1, 2, 2],
            "MC_GenPart_pt": [18.0, 11.0, 7.0, 5.0, 5.2, 4.5, 4.8, 2.4, 2.5],
            "MC_GenPart_eta": [0.3, -0.4, 0.2, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
            "MC_GenPart_phi": [0.0, 1.0, 2.0, 0.1, -0.1, 1.1, 0.9, 2.1, 1.9],
            "MC_GenPart_mass": [3.096, 3.096, 1.019, 0.105, 0.105, 0.105, 0.105, 0.494, 0.494],
            "Jpsi_1_mass": [3.09],
            "Jpsi_1_pt": [18.0],
            "Jpsi_1_px": [18.0],
            "Jpsi_1_py": [0.0],
            "Jpsi_1_pz": [0.0],
            "Jpsi_1_VtxProb": [0.2],
            "Jpsi_1_mu_1_Idx": [0],
            "Jpsi_1_mu_2_Idx": [1],
            "Jpsi_2_mass": [3.10],
            "Jpsi_2_pt": [11.0],
            "Jpsi_2_px": [11.0],
            "Jpsi_2_py": [0.0],
            "Jpsi_2_pz": [0.0],
            "Jpsi_2_VtxProb": [0.3],
            "Jpsi_2_mu_1_Idx": [2],
            "Jpsi_2_mu_2_Idx": [3],
            "Phi_mass": [1.019],
            "Phi_pt": [7.0],
            "Phi_px": [7.0],
            "Phi_py": [0.0],
            "Phi_pz": [0.0],
            "Phi_VtxProb": [0.4],
            "Phi_K_1_Idx": [0],
            "Phi_K_2_Idx": [1],
            "Phi_K_1_pt": [2.4],
            "Phi_K_1_eta": [0.3],
            "Phi_K_1_vertexId": [9],
            "Phi_K_1_genMatchIdx": [7],
            "Phi_K_2_pt": [2.5],
            "Phi_K_2_eta": [-0.3],
            "Phi_K_2_vertexId": [9],
            "Phi_K_2_genMatchIdx": [8],
            "muGenMatchIdx": [3, 4, 5, 6],
            "muVertexId": [9, 9, 9, 9],
            "muIsJpsiTrigMatch": [1, 1, 0, 0],
            "muIsJpsiFilterMatch": [1, 1, 0, 0],
            "Pri_fitValid": [1],
            "Pri_fitPass": [1],
            "Pri_assocPVPass": [1],
            "Pri_trackPVPass": [1],
            "Pri_passAny": [1],
        }

    def test_find_jpsijpsiphi_gen_system_orders_jpsis_by_pt(self) -> None:
        system = find_jpsijpsiphi_gen_system(self._jpsijpsiphi_event())

        assert system is not None
        self.assertEqual(system.jpsi_lead.idx, 0)
        self.assertEqual(system.jpsi_sublead.idx, 1)
        self.assertEqual(system.phi.idx, 2)
        self.assertGreater(system.triple_mass, 0.0)

    def test_build_event_efficiency_row_sets_cumulative_flags(self) -> None:
        gen_row, event_row = build_event_efficiency_row(
            self._jpsijpsiphi_event(),
            "sample.root",
            "JJP_TEST",
            0,
            OfflineSelectionConfig(),
        )

        assert gen_row is not None
        assert event_row is not None
        self.assertEqual(event_row["full_gen"], 1)
        self.assertEqual(event_row["fiducial_acceptance"], 1)
        self.assertEqual(event_row["hlt_muon_matched"], 1)
        self.assertEqual(event_row["all6_same_recVtx"], 1)
        self.assertEqual(event_row["final_nominal"], 1)
        self.assertEqual(gen_row["jpsi_lead_gen_idx"], 0)

    def test_clopper_pearson_interval_handles_edges(self) -> None:
        low, high = clopper_pearson_interval(10, 0)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        low, high = clopper_pearson_interval(10, 10)
        self.assertLess(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_build_efficiency_counts_includes_uncertainties_and_correlated_maps(self) -> None:
        gen_row, event_row = build_event_efficiency_row(
            self._jpsijpsiphi_event(),
            "sample.root",
            "JJP_TEST",
            0,
            OfflineSelectionConfig(),
        )
        gen_df = pd.DataFrame([gen_row])
        event_df = pd.DataFrame([event_row])

        counts = build_efficiency_counts(gen_df, event_df, EfficiencyBinning())
        cutflow = build_cutflow(event_df)

        self.assertIn("err_low", counts.columns)
        self.assertIn("err_high", counts.columns)
        self.assertTrue((counts["map_type"] == "correlated_3d").any())
        self.assertAlmostEqual(cutflow.loc[cutflow["step"] == "final_nominal", "efficiency"].iloc[0], 1.0)


class ParameterTableHelpersTest(unittest.TestCase):
    def test_roofit_parameter_table_records_values_errors_and_bounds(self) -> None:
        class DummyVar:
            def __init__(self, value, error, constant=False, bounds=None):
                self._value = value
                self._error = error
                self._constant = constant
                self._bounds = bounds

            def getVal(self):
                return self._value

            def getError(self):
                return self._error

            def isConstant(self):
                return self._constant

            def hasMin(self):
                return self._bounds is not None

            def hasMax(self):
                return self._bounds is not None

            def getMin(self):
                return self._bounds[0]

            def getMax(self):
                return self._bounds[1]

        yield_vars = {"N_sss": DummyVar(12.0, 2.5, False, (0.0, 100.0))}
        axis_states = {
            "Jpsi_1_mass": {
                "role": "jpsi",
                "mean": DummyVar(3.097, 0.004, False, (3.0, 3.2)),
                "sigma_cb": DummyVar(0.03, 0.002),
                "alpha": DummyVar(1.5, 0.1, True),
                "n": DummyVar(4.0, 0.3),
                "sigma_g": DummyVar(0.05, 0.004),
                "frac": DummyVar(0.7, 0.05),
                "slope": DummyVar(-1.0, 0.2),
            },
            "Ups_mass": {
                "role": "ups",
                "mean": DummyVar(9.46, 0.01, False, (9.2, 9.7)),
                "sigma": DummyVar(0.12, 0.01),
                "frac_excited": DummyVar(0.3, 0.05),
                "frac_3s_in_excited": DummyVar(0.25, 0.03),
                "coeff_vars": [DummyVar(0.1, 0.02), DummyVar(-0.2, 0.03)],
            },
            "Phi_mass": {
                "role": "phi",
                "mean": DummyVar(1.019, 0.001, False, (1.005, 1.035)),
                "sigma": DummyVar(0.004, 0.0004),
                "coeff_vars": [DummyVar(0.1, 0.02), DummyVar(-0.2, 0.03)],
            },
        }

        table = roofit_parameter_table(yield_vars, axis_states)

        self.assertIn("parameter", table.columns)
        self.assertIn("error", table.columns)
        self.assertIn("bounds", table.columns)
        self.assertTrue((table["parameter"] == "N_sss").any())
        self.assertTrue((table["parameter"] == "Ups_mass.frac_excited").any())
        self.assertTrue((table["parameter"] == "Phi_mass.c2").any())

    def test_minuit_parameter_table_records_fixed_flags(self) -> None:
        class FakeMinuit:
            parameters = ("N_sss", "phi_c1", "j1_mean")
            limits = {
                "N_sss": (0.0, 50.0),
                "phi_c1": (-10.0, 10.0),
                "j1_mean": (3.0, 3.2),
            }
            values = {"N_sss": 11.0, "phi_c1": 0.4, "j1_mean": 3.097}
            errors = {"N_sss": 2.0, "phi_c1": 0.1, "j1_mean": 0.004}
            fixed = {"N_sss": False, "phi_c1": False, "j1_mean": True}

        table = minuit_parameter_table(FakeMinuit())

        self.assertEqual(table.loc[table["parameter"] == "j1_mean", "constant"].iloc[0], 1)
        self.assertEqual(table.loc[table["parameter"] == "phi_c1", "group"].iloc[0], "phi")


if __name__ == "__main__":
    unittest.main()
