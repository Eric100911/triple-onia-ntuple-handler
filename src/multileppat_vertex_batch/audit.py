from __future__ import annotations

import pandas as pd


def summarize_mass_window_audit(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame(columns=["selector", "stage", "count"])
    return (
        audit_df.groupby(["selector", "stage"], as_index=False)["count"]
        .sum()
        .sort_values(["selector", "stage"])
        .reset_index(drop=True)
    )


def selector_stage_count(audit_df: pd.DataFrame, selector: str, stage: str) -> int:
    if audit_df.empty:
        return 0
    series = audit_df.loc[(audit_df["selector"] == selector) & (audit_df["stage"] == stage), "count"]
    return int(series.sum()) if not series.empty else 0
