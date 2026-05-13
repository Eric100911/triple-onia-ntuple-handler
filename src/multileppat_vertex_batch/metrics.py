from __future__ import annotations

import math

import pandas as pd

from .schema import CLASSIFIER_SPECS, EVENT_KEYS


def compute_confusion(pred: pd.Series, truth: pd.Series) -> dict[str, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)

    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    tn = int((~pred & ~truth).sum())
    fn = int((~pred & truth).sum())

    def ratio(num: int, den: int) -> float:
        return float(num) / den if den else float("nan")

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    fpr = ratio(fp, fp + tn)
    fnr = ratio(fn, fn + tp)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    bal_acc = ((recall + specificity) / 2.0) if math.isfinite(recall) and math.isfinite(specificity) else float("nan")

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "FPR": fpr,
        "FNR": fnr,
        "F1": f1,
        "balanced_accuracy": bal_acc,
    }


def summarize_classifier(df: pd.DataFrame, pred_col: str, label: str | None = None, truth_col: str = "truth_triple_strict") -> dict:
    row = compute_confusion(df[pred_col], df[truth_col])
    row.update(
        {
            "classifier": label or pred_col,
            "pred_positive": int(df[pred_col].sum()),
            "truth_positive": int(df[truth_col].sum()),
            "n_rows": int(len(df)),
        }
    )
    return row


def summarize_event_level(df: pd.DataFrame, pred_col: str, label: str | None = None, truth_col: str = "truth_triple_strict") -> dict:
    event_df = (
        df.groupby(EVENT_KEYS, as_index=False)
        .agg(pred_any=(pred_col, "max"), truth_any=(truth_col, "max"), n_candidates=("cand_idx", "count"))
    )
    row = compute_confusion(event_df["pred_any"], event_df["truth_any"])
    row.update(
        {
            "classifier": label or pred_col,
            "pred_positive_events": int(event_df["pred_any"].sum()),
            "truth_positive_events": int(event_df["truth_any"].sum()),
            "n_events": int(len(event_df)),
        }
    )
    return row


def summarize_best_candidate(
    df: pd.DataFrame,
    pred_col: str,
    label: str | None = None,
    truth_col: str = "truth_triple_strict",
    score_col: str = "ranking_score",
) -> dict:
    passed = df[df[pred_col] == 1].copy()
    truth_events = (
        df.groupby(EVENT_KEYS, as_index=False)[truth_col]
        .max()
        .rename(columns={truth_col: "event_truth_positive"})
    )

    if passed.empty:
        truth_total = int(truth_events["event_truth_positive"].sum())
        return {
            "classifier": label or pred_col,
            "events_with_pass": 0,
            "best_candidate_truth_positive": 0,
            "best_candidate_purity": float("nan"),
            "truth_positive_events": truth_total,
            "truth_events_recovered_by_best": 0,
            "best_candidate_recall_over_truth_events": 0.0 if truth_total else float("nan"),
        }

    passed = passed.sort_values(
        EVENT_KEYS + [score_col, "cand_idx"],
        ascending=[True, True, True, True, False, True],
    )
    best = passed.groupby(EVENT_KEYS, as_index=False).first()
    truth_total = int(truth_events["event_truth_positive"].sum())
    selected_true = int(best[truth_col].sum())
    events_with_pass = int(len(best))
    return {
        "classifier": label or pred_col,
        "events_with_pass": events_with_pass,
        "best_candidate_truth_positive": selected_true,
        "best_candidate_purity": selected_true / events_with_pass if events_with_pass else float("nan"),
        "truth_positive_events": truth_total,
        "truth_events_recovered_by_best": selected_true,
        "best_candidate_recall_over_truth_events": selected_true / truth_total if truth_total else float("nan"),
    }


def build_candidate_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_classifier(df, column, label) for column, label in CLASSIFIER_SPECS]
    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "precision", "recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_event_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_event_level(df, column, label) for column, label in CLASSIFIER_SPECS]
    return pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "precision", "recall"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_best_candidate_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_best_candidate(df, column, label) for column, label in CLASSIFIER_SPECS]
    return pd.DataFrame(rows).sort_values(
        ["best_candidate_purity", "best_candidate_recall_over_truth_events"],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_phi_vtxprob_scan(
    frame: pd.DataFrame,
    threshold_grid: tuple[float, ...] | list[float],
    baseline_col: str = "Phi_vertexCriteriaPass",
) -> pd.DataFrame:
    baseline = frame.loc[frame[baseline_col] == 1].copy()
    baseline_signal_total = int((baseline["truth_triple_strict"] == 1).sum())
    baseline_background_total = int((baseline["truth_triple_strict"] == 0).sum())
    baseline_pileup_total = int(baseline["pileup_like_proxy"].sum())

    rows: list[dict] = []
    for threshold in threshold_grid:
        selected = baseline.loc[baseline["Phi_VtxProb"] >= threshold].copy()
        signal_kept = int((selected["truth_triple_strict"] == 1).sum())
        background_kept = int((selected["truth_triple_strict"] == 0).sum())
        pileup_kept = int(selected["pileup_like_proxy"].sum())
        rows.append(
            {
                "Phi_VtxProb_cut": threshold,
                "kept_candidates": int(len(selected)),
                "truth_positive_kept": signal_kept,
                "truth_negative_kept": background_kept,
                "pileup_like_kept": pileup_kept,
                "truth_positive_eff_vs_baseline": signal_kept / baseline_signal_total if baseline_signal_total else float("nan"),
                "truth_negative_eff_vs_baseline": background_kept / baseline_background_total if baseline_background_total else float("nan"),
                "pileup_like_eff_vs_baseline": pileup_kept / baseline_pileup_total if baseline_pileup_total else float("nan"),
                "pileup_like_rejection_vs_baseline": 1.0 - (pileup_kept / baseline_pileup_total)
                if baseline_pileup_total
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)
