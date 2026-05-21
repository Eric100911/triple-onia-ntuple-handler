from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    def display(obj):
        print(obj)

import matplotlib.pyplot as plt

from .config import CmsPlotStyleConfig
from .fit_iminuit import projection_curves_iminuit
from .fit_roofit import projection_specs


FIT_PROJECTION_LABELS = {
    "Jpsi_1_mass": r"$m_{\mu\mu}$ [GeV]",
    "Jpsi_2_mass": r"$m_{\mu\mu}$ [GeV]",
    "Ups_mass": r"$m_{\mu\mu}$ [GeV]",
    "Phi_mass": r"$m_{KK}$ [GeV]",
}

FIT_PROJECTION_TITLES = {
    "roofit": {
        "Jpsi_1_mass": r"3D-fit projection on $J/\psi_1$",
        "Jpsi_2_mass": r"3D-fit projection on $J/\psi_2$",
        "Ups_mass": r"3D-fit projection on $\Upsilon$",
        "Phi_mass": r"3D-fit projection on $\phi$",
    },
    "iminuit": {
        "Jpsi_1_mass": r"iminuit projection on $J/\psi_1$",
        "Jpsi_2_mass": r"iminuit projection on $J/\psi_2$",
        "Ups_mass": r"iminuit projection on $\Upsilon$",
        "Phi_mass": r"iminuit projection on $\phi$",
    },
}


def _require_mplhep():
    try:
        import mplhep as hep
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mplhep is required to render CMS-style fit projection plots.") from exc
    return hep


def default_fit_plot_specs(
    fit_branches: tuple[str, ...] | list[str],
    backend: str,
    bins: int = 20,
) -> list[dict[str, object]]:
    if backend not in FIT_PROJECTION_TITLES:
        raise KeyError(f"Unknown fit projection backend: {backend}")
    specs = []
    for branch in fit_branches:
        if branch not in FIT_PROJECTION_LABELS:
            continue
        specs.append(
            {
                "branch": branch,
                "xlabel": FIT_PROJECTION_LABELS[branch],
                "title": FIT_PROJECTION_TITLES[backend][branch],
                "bins": int(bins),
            }
        )
    return specs


def _cms_caption(plot_style_cfg: CmsPlotStyleConfig) -> str:
    if plot_style_cfg.caption is not None:
        return plot_style_cfg.caption
    return "Preliminary" if plot_style_cfg.is_data else "Simulation Preliminary"


def apply_cms_label(ax, plot_style_cfg: CmsPlotStyleConfig) -> None:
    hep = _require_mplhep()
    label_kwargs = {
        "ax": ax,
        "data": bool(plot_style_cfg.is_data),
        "label": _cms_caption(plot_style_cfg),
        "com": float(plot_style_cfg.energy_tev),
    }
    if plot_style_cfg.lumi_fb is not None:
        label_kwargs["lumi"] = float(plot_style_cfg.lumi_fb)
    if plot_style_cfg.era and plot_style_cfg.era.isdigit():
        label_kwargs["year"] = int(plot_style_cfg.era)

    hep.cms.label(**label_kwargs)

    if plot_style_cfg.era and not plot_style_cfg.era.isdigit():
        ax.text(0.03, 0.88, plot_style_cfg.era, transform=ax.transAxes, ha="left", va="top")


def evaluate_roofit_pdf_counts(root_module, var, pdf, yield_value: float, n_bins: int, x_range: tuple[float, float]):
    x_min, x_max = x_range
    x = np.linspace(x_min, x_max, 800)
    norm_set = root_module.RooArgSet(var)
    y = np.empty_like(x)
    for idx, xv in enumerate(x):
        var.setVal(float(xv))
        y[idx] = pdf.getVal(norm_set)
    bin_width = (x_max - x_min) / n_bins
    return x, float(yield_value) * y * bin_width


def save_fit_projection_plot(
    output_path: Path,
    values: np.ndarray,
    x_range: tuple[float, float],
    plot_spec: dict[str, object],
    x: np.ndarray,
    signal_curve: np.ndarray,
    background_curve: np.ndarray,
    plot_style_cfg: CmsPlotStyleConfig,
) -> Path:
    hep = _require_mplhep()
    hep.style.use("CMS")
    counts, edges = np.histogram(values, bins=int(plot_spec["bins"]), range=x_range)
    centers = 0.5 * (edges[:-1] + edges[1:])
    errors = np.sqrt(np.maximum(counts, 1.0))

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.errorbar(centers, counts, yerr=errors, fmt="o", color="black", ms=4.5, label="Data")
    ax.plot(x, signal_curve + background_curve, color="#d62728", lw=2.0, label="Total fit")
    ax.plot(x, signal_curve, color="#1f77b4", lw=2.0, ls="--", label="Signal projection")
    ax.plot(x, background_curve, color="#2ca02c", lw=2.0, ls=":", label="Background projection")
    ax.set_xlim(*x_range)
    ax.set_xlabel(str(plot_spec["xlabel"]))
    ax.set_ylabel("Candidates / bin")
    ax.set_title(str(plot_spec["title"]))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    apply_cms_label(ax, plot_style_cfg)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_roofit_projection_plots(
    output_dir: Path,
    fit_payload: dict[str, object],
    plot_style_cfg: CmsPlotStyleConfig,
    plot_specs: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    fit_df = fit_payload["fit_df"]
    if fit_df.empty:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = plot_specs or default_fit_plot_specs(tuple(fit_payload["fit_branches"]), backend="roofit")
    projection_payloads = projection_specs(fit_payload, plot_specs)
    written: dict[str, Path] = {}

    for branch, spec in projection_payloads.items():
        x_range = fit_payload["active_windows"][branch]
        values = fit_df[branch].to_numpy(dtype=float)
        root_module = fit_payload["root_module"]
        x, signal_curve = evaluate_roofit_pdf_counts(
            root_module,
            spec["var"],
            spec["signal_pdf"],
            spec["signal"],
            int(spec["bins"]),
            x_range,
        )
        _, background_curve = evaluate_roofit_pdf_counts(
            root_module,
            spec["var"],
            spec["background_pdf"],
            spec["background"],
            int(spec["bins"]),
            x_range,
        )
        path = output_dir / f"projection_{branch}.png"
        written[branch] = save_fit_projection_plot(
            path,
            values,
            x_range,
            spec,
            x,
            signal_curve,
            background_curve,
            plot_style_cfg,
        )
    return written


def write_iminuit_projection_plots(
    output_dir: Path,
    fit_payload: dict[str, object],
    plot_style_cfg: CmsPlotStyleConfig,
    plot_specs: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    fit_df = fit_payload["fit_df"]
    if fit_df.empty:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = plot_specs or default_fit_plot_specs(tuple(fit_payload["fit_branches"]), backend="iminuit")
    plot_spec_map = {str(spec["branch"]): spec for spec in plot_specs}
    written: dict[str, Path] = {}

    for branch in fit_payload["fit_branches"]:
        spec = plot_spec_map[branch]
        x_range = fit_payload["active_windows"][branch]
        values = fit_df[branch].to_numpy(dtype=float)
        x = np.linspace(*x_range, 800)
        bin_width = (x_range[1] - x_range[0]) / int(spec["bins"])
        signal_curve, background_curve = projection_curves_iminuit(
            fit_payload["minuit"],
            branch,
            x,
            fit_payload["active_windows"],
            fit_payload["analysis_mode"],
            fit_payload.get("ups_background_order", 4),
        )
        path = output_dir / f"projection_{branch}.png"
        written[branch] = save_fit_projection_plot(
            path,
            values,
            x_range,
            spec,
            x,
            signal_curve * bin_width,
            background_curve * bin_width,
            plot_style_cfg,
        )
    return written


def display_frame(obj, sort_by: str | list[str] | None = None, columns: list[str] | None = None, head: int | None = None):
    frame = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    if columns is not None:
        frame = frame[columns]
    if sort_by is not None and not frame.empty:
        frame = frame.sort_values(sort_by).reset_index(drop=True)
    if head is not None:
        frame = frame.head(head)
    display(frame)
    return frame


def plot_metric_bars(frame: pd.DataFrame, classifier_col: str = "classifier", metric_cols: list[str] | None = None, title: str = "Classifier metrics"):
    metric_cols = metric_cols or ["precision", "recall", "specificity"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = list(range(len(frame)))
    width = 0.8 / len(metric_cols)
    for offset, metric in enumerate(metric_cols):
        values = frame[metric].tolist()
        positions = [xi + (offset - (len(metric_cols) - 1) / 2.0) * width for xi in x]
        ax.bar(positions, values, width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(frame[classifier_col].tolist(), rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_truth_heatmap(df: pd.DataFrame, pred_col: str, truth_col: str = "truth_triple_strict", title: str | None = None):
    table = pd.crosstab(df[pred_col], df[truth_col]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    im = ax.imshow(table.values, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["truth=0", "truth=1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"{pred_col}=0", f"{pred_col}=1"])
    ax.set_title(title or f"{pred_col} vs truth")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(table.values[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    return table


def plot_hist_by_truth(df: pd.DataFrame, column: str, clip: tuple[float, float] | None = None, bins: int = 30, xscale: str = "linear", yscale: str = "linear"):
    if xscale not in ["linear", "log"]:
        raise ValueError(f"xscale must be 'linear' or 'log', got '{xscale}'")
    if yscale not in ["linear", "log"]:
        raise ValueError(f"yscale must be 'linear' or 'log', got '{yscale}'")

    data = df[[column, "truth_triple_strict"]].copy()
    data = data[pd.notna(data[column])]
    if clip is not None:
        data = data[(data[column] >= clip[0]) & (data[column] <= clip[1])]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for truth_value, label, color in [(0, "truth-negative", "#d95f02"), (1, "truth-positive", "#1b9e77")]:
        subset = data.loc[data["truth_triple_strict"] == truth_value, column]
        if not subset.empty:
            ax.hist(subset, bins=bins, histtype="step", linewidth=2, label=label, color=color)
    ax.set_title(f"{column} by truth label")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(column)
    ax.set_ylabel("Candidates")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_phi_vtxprob_scan(scan_df: pd.DataFrame, chosen_cut: float):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    thresholds = scan_df["Phi_VtxProb_cut"].replace(0.0, 1e-6)

    axes[0].plot(thresholds, scan_df["truth_positive_eff_vs_baseline"], marker="o", label="truth-positive efficiency")
    axes[0].plot(thresholds, scan_df["truth_negative_eff_vs_baseline"], marker="o", label="truth-negative efficiency")
    axes[0].plot(thresholds, scan_df["pileup_like_eff_vs_baseline"], marker="o", label="pileup-like efficiency")
    axes[0].axvline(chosen_cut, color="black", linestyle="--", linewidth=1.5, label=f"chosen cut = {chosen_cut:.0e}")
    axes[0].set_xscale("log")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("Phi_VtxProb threshold")
    axes[0].set_ylabel("Efficiency vs Phi_vertexCriteriaPass baseline")
    axes[0].set_title("Retention scan inside the baseline")
    axes[0].legend()

    axes[1].plot(thresholds, scan_df["pileup_like_rejection_vs_baseline"], marker="o", color="#d95f02")
    axes[1].axvline(chosen_cut, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xlabel("Phi_VtxProb threshold")
    axes[1].set_ylabel("Pileup-like rejection vs baseline")
    axes[1].set_title("How much extra background is removed?")
    plt.tight_layout()
    plt.show()


def _efficiency_matrix(frame: pd.DataFrame, value_col: str) -> tuple[np.ndarray, list[str], list[str]]:
    if frame.empty:
        return np.empty((0, 0)), [], []
    x_bins = sorted(int(value) for value in frame["x_bin"].dropna().unique())
    y_bins = sorted(int(value) for value in frame["y_bin"].dropna().unique())
    matrix = np.full((len(y_bins), len(x_bins)), np.nan)
    for _, row in frame.iterrows():
        if pd.isna(row.get("x_bin")) or pd.isna(row.get("y_bin")):
            continue
        ix = x_bins.index(int(row["x_bin"]))
        iy = y_bins.index(int(row["y_bin"]))
        if int(row.get("total", 0)) > 0:
            matrix[iy, ix] = float(row[value_col])
    x_labels = [
        str(frame.loc[frame["x_bin"] == idx, "x_label"].dropna().iloc[0])
        if not frame.loc[frame["x_bin"] == idx, "x_label"].dropna().empty
        else str(idx)
        for idx in x_bins
    ]
    y_labels = [
        str(frame.loc[frame["y_bin"] == idx, "y_label"].dropna().iloc[0])
        if not frame.loc[frame["y_bin"] == idx, "y_label"].dropna().empty
        else str(idx)
        for idx in y_bins
    ]
    return matrix, x_labels, y_labels


def _annotate_efficiency_bins(ax, frame: pd.DataFrame, max_cells: int = 64) -> None:
    if frame.empty or frame.shape[0] > max_cells:
        return
    x_bins = sorted(int(value) for value in frame["x_bin"].dropna().unique())
    y_bins = sorted(int(value) for value in frame["y_bin"].dropna().unique())
    for _, row in frame.iterrows():
        total = int(row.get("total", 0))
        if total <= 0 or pd.isna(row.get("x_bin")) or pd.isna(row.get("y_bin")):
            continue
        ix = x_bins.index(int(row["x_bin"]))
        iy = y_bins.index(int(row["y_bin"]))
        text = f"{float(row['efficiency']):.2f}\n+/-{float(row['err_sym']):.2f}\n{int(row['passed'])}/{total}"
        ax.text(ix, iy, text, ha="center", va="center", fontsize=7, color="black")


def save_efficiency_heatmap_pair(
    output_path: Path,
    frame: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    plot_style_cfg: CmsPlotStyleConfig,
    min_total: int = 1,
) -> Path:
    hep = _require_mplhep()
    hep.style.use("CMS")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame.loc[frame["total"] < int(min_total), ["efficiency", "err_sym"]] = np.nan
    eff, x_labels, y_labels = _efficiency_matrix(frame, "efficiency")
    err, _, _ = _efficiency_matrix(frame, "err_sym")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    for ax, matrix, panel_title, zlabel in (
        (axes[0], eff, "Efficiency", "Efficiency"),
        (axes[1], err, "Uncertainty", "Sym. CP uncertainty"),
    ):
        im = ax.imshow(matrix, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        fig.colorbar(im, ax=ax, label=zlabel)
    _annotate_efficiency_bins(axes[0], frame)
    fig.suptitle(title)
    apply_cms_label(axes[0], plot_style_cfg)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def write_efficiency_plots(
    output_dir: Path,
    counts_df: pd.DataFrame,
    plot_style_cfg: CmsPlotStyleConfig,
    min_total: int = 1,
) -> dict[str, Path]:
    if counts_df.empty:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    object_df = counts_df.loc[counts_df["map_type"] == "object_2d"].copy()
    for (obj, step), frame in object_df.groupby(["object", "step"], dropna=False):
        path = output_dir / f"object2d_{obj}_{step}.png"
        written[f"object2d.{obj}.{step}"] = save_efficiency_heatmap_pair(
            path,
            frame,
            title=f"{obj} {step}",
            xlabel=r"$p_{\mathrm{T}}$ [GeV]",
            ylabel=r"$|y|$",
            plot_style_cfg=plot_style_cfg,
            min_total=min_total,
        )

    corr_df = counts_df.loc[counts_df["map_type"] == "correlated_3d"].copy()
    for (step, z_bin), frame in corr_df.groupby(["step", "z_bin"], dropna=False):
        z_label = str(frame["z_label"].dropna().iloc[0]) if not frame["z_label"].dropna().empty else str(z_bin)
        path = output_dir / f"corr3d_{step}_phiPt_{z_bin}.png"
        written[f"corr3d.{step}.{z_bin}"] = save_efficiency_heatmap_pair(
            path,
            frame,
            title=rf"{step}, $p_{{T}}(\phi)$ = {z_label} GeV",
            xlabel=r"$p_{\mathrm{T}}(J/\psi_{\mathrm{lead}})$ [GeV]",
            ylabel=r"$p_{\mathrm{T}}(J/\psi_{\mathrm{sublead}})$ [GeV]",
            plot_style_cfg=plot_style_cfg,
            min_total=min_total,
        )
    return written
