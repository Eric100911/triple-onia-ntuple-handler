from __future__ import annotations

from array import array
from dataclasses import dataclass, fields
import math
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .schema import AnalysisModeSpec, get_analysis_mode_spec


UPS_1S_MASS = 9.46030
UPS_2S_MASS = 10.02326
UPS_3S_MASS = 10.3552
UPS_2S_DELTA = UPS_2S_MASS - UPS_1S_MASS
UPS_3S_DELTA = UPS_3S_MASS - UPS_1S_MASS

ROOFIT_STATE: dict[str, object] = {}
UPS_PEAK_COMPONENTS = (
    ("Ups_2S", "ups_2s_fraction", "ups_2s_signal_yield"),
    ("Ups_3S", "ups_3s_fraction", "ups_3s_signal_yield"),
    ("Ups_excited", "ups_excited_fraction", "ups_excited_signal_yield"),
)


@dataclass(frozen=True)
class FitParamSpec:
    value: float
    bounds: tuple[float, float] | None
    constant: bool = False


@dataclass(frozen=True)
class JpsiPdfConfig:
    mean: FitParamSpec
    sigma_cb: FitParamSpec
    alpha: FitParamSpec
    n: FitParamSpec
    sigma_g: FitParamSpec
    frac: FitParamSpec
    slope: FitParamSpec


DEFAULT_JPSI_PDF_CONFIG = JpsiPdfConfig(
    mean=FitParamSpec(3.097, (3.00, 3.20), False),
    sigma_cb=FitParamSpec(0.035, (0.003, 0.20), False),
    alpha=FitParamSpec(1.5, (0.2, 8.0), False),
    n=FitParamSpec(4.0, (1.01, 80.0), False),
    sigma_g=FitParamSpec(0.060, (0.003, 0.30), False),
    frac=FitParamSpec(0.75, (0.0, 1.0), False),
    slope=FitParamSpec(-1.0, (-20.0, -1e-4), False),
)

JPSI_PDF_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "float_all": {},
    "small_sample": {
        "alpha": {"constant": True},
        "n": {"constant": True},
        "frac": {"constant": True},
    },
    "tails_locked": {
        "alpha": {"constant": True},
        "n": {"constant": True},
    },
}


def _config_to_map(config: JpsiPdfConfig) -> dict[str, FitParamSpec]:
    return {field.name: getattr(config, field.name) for field in fields(JpsiPdfConfig)}


def _map_to_config(spec_map: Mapping[str, FitParamSpec]) -> JpsiPdfConfig:
    return JpsiPdfConfig(**{field.name: spec_map[field.name] for field in fields(JpsiPdfConfig)})


def _merge_param_spec(spec: FitParamSpec, override: Mapping[str, Any]) -> FitParamSpec:
    allowed_keys = {"value", "bounds", "constant"}
    unknown_keys = set(override) - allowed_keys
    if unknown_keys:
        raise KeyError(f"Unknown FitParamSpec override keys: {sorted(unknown_keys)}")
    value = float(override["value"]) if "value" in override else float(spec.value)
    bounds_raw = override["bounds"] if "bounds" in override else spec.bounds
    bounds = None if bounds_raw is None else (float(bounds_raw[0]), float(bounds_raw[1]))
    constant = bool(override["constant"]) if "constant" in override else bool(spec.constant)
    if bounds is not None and not (bounds[0] <= value <= bounds[1]):
        raise ValueError(f"Parameter value {value} lies outside bounds {bounds}.")
    return FitParamSpec(value=value, bounds=bounds, constant=constant)


def resolve_jpsi_pdf_config(
    preset: str = "float_all",
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> JpsiPdfConfig:
    if preset not in JPSI_PDF_PRESETS:
        raise KeyError(f"Unknown J/psi PDF preset '{preset}'. Expected one of {sorted(JPSI_PDF_PRESETS)}.")
    spec_map = _config_to_map(DEFAULT_JPSI_PDF_CONFIG)
    for layer in (JPSI_PDF_PRESETS[preset], overrides or {}):
        for name, override in layer.items():
            if name not in spec_map:
                raise KeyError(f"Unknown J/psi PDF parameter '{name}'.")
            spec_map[name] = _merge_param_spec(spec_map[name], override)
    return _map_to_config(spec_map)


def locked_param_names(config: JpsiPdfConfig) -> list[str]:
    return [name for name, spec in _config_to_map(config).items() if spec.constant]


def jpsi_pdf_config_table(config: JpsiPdfConfig, prefix: str = "jpsi") -> pd.DataFrame:
    rows = []
    for name, spec in _config_to_map(config).items():
        rows.append({"parameter": f"{prefix}.{name}", "value": spec.value, "bounds": spec.bounds, "constant": int(spec.constant)})
    return pd.DataFrame(rows)


def export_jpsi_pdf_values(state: Mapping[str, Any]) -> dict[str, float]:
    exported: dict[str, float] = {}
    for key in ("mean", "sigma_cb", "alpha", "n", "sigma_g", "frac", "slope"):
        obj = state[key]
        exported[key] = float(obj.getVal()) if hasattr(obj, "getVal") else float(obj)
    return exported


def _make_roorealvar(root_mod, name: str, title: str, spec: FitParamSpec):
    if spec.bounds is None:
        var = root_mod.RooRealVar(name, title, spec.value)
    else:
        var = root_mod.RooRealVar(name, title, spec.value, spec.bounds[0], spec.bounds[1])
    var.setConstant(bool(spec.constant))
    return var


def build_fit_mask(selected_candidate_df: pd.DataFrame, fit_branches: list[str], windows: dict[str, tuple[float, float]]) -> pd.Series:
    fit_mask = pd.Series(True, index=selected_candidate_df.index, dtype=bool)
    for branch in fit_branches:
        low, high = windows[branch]
        fit_mask &= np.isfinite(selected_candidate_df[branch]) & (selected_candidate_df[branch] >= low) & (selected_candidate_df[branch] <= high)
    return fit_mask


def build_fit_frame(selected_candidate_df: pd.DataFrame, fit_branches: list[str], windows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    if selected_candidate_df.empty:
        return selected_candidate_df.loc[:, fit_branches].copy() if fit_branches else pd.DataFrame()
    return selected_candidate_df.loc[build_fit_mask(selected_candidate_df, fit_branches, windows), fit_branches].copy()


def reset_roofit_state():
    ROOFIT_STATE.clear()


def make_roodataset(root_mod, frame, variables):
    tree = root_mod.TTree("fit_tree", "fit_tree")
    buffers = {}
    ordered_names = list(variables.keys())
    for name in ordered_names:
        buffers[name] = array("d", [0.0])
        tree.Branch(name, buffers[name], f"{name}/D")
    for row in frame[ordered_names].itertuples(index=False, name=None):
        for name, value in zip(ordered_names, row):
            buffers[name][0] = float(value)
        tree.Fill()
    argset = root_mod.RooArgSet()
    for name in ordered_names:
        argset.add(variables[name])
    dataset = root_mod.RooDataSet("data", "data", argset, root_mod.RooFit.Import(tree))
    ROOFIT_STATE["dataset"] = {"tree": tree, "buffers": buffers, "argset": argset, "ordered_names": ordered_names, "dataset": dataset}
    return dataset


def build_jpsi_model(root_mod, var, tag, pdf_config: JpsiPdfConfig | None = None):
    pdf_config = pdf_config or DEFAULT_JPSI_PDF_CONFIG
    mean = _make_roorealvar(root_mod, f"mean_{tag}", f"mean_{tag}", pdf_config.mean)
    sigma_cb = _make_roorealvar(root_mod, f"sigma_cb_{tag}", f"sigma_cb_{tag}", pdf_config.sigma_cb)
    alpha = _make_roorealvar(root_mod, f"alpha_{tag}", f"alpha_{tag}", pdf_config.alpha)
    n = _make_roorealvar(root_mod, f"n_{tag}", f"n_{tag}", pdf_config.n)
    sigma_g = _make_roorealvar(root_mod, f"sigma_g_{tag}", f"sigma_g_{tag}", pdf_config.sigma_g)
    frac = _make_roorealvar(root_mod, f"frac_{tag}", f"frac_{tag}", pdf_config.frac)
    slope = _make_roorealvar(root_mod, f"slope_{tag}", f"slope_{tag}", pdf_config.slope)
    cb = root_mod.RooCBShape(f"cb_{tag}", f"cb_{tag}", var, mean, sigma_cb, alpha, n)
    gauss = root_mod.RooGaussian(f"gauss_{tag}", f"gauss_{tag}", var, mean, sigma_g)
    signal_pdfs = root_mod.RooArgList()
    signal_pdfs.add(cb)
    signal_pdfs.add(gauss)
    signal_fracs = root_mod.RooArgList()
    signal_fracs.add(frac)
    signal = root_mod.RooAddPdf(f"sig_{tag}", f"sig_{tag}", signal_pdfs, signal_fracs)
    background = root_mod.RooExponential(f"bkg_{tag}", f"bkg_{tag}", var, slope)
    state = {
        "role": "jpsi",
        "var": var,
        "mean": mean,
        "sigma_cb": sigma_cb,
        "alpha": alpha,
        "n": n,
        "sigma_g": sigma_g,
        "frac": frac,
        "cb": cb,
        "gauss": gauss,
        "signal": signal,
        "slope": slope,
        "background": background,
        "pdf_config": pdf_config,
        "locked_params": locked_param_names(pdf_config),
    }
    ROOFIT_STATE[f"jpsi_{tag}"] = state
    return state


def _normalized_axis_formula(root_mod, var, tag: str, x_range: tuple[float, float]):
    x_min, x_max = map(float, x_range)
    return root_mod.RooFormulaVar(
        f"t_{tag}",
        f"(2.0*(@0-{x_min:.16g})/{(x_max - x_min):.16g} - 1.0)",
        root_mod.RooArgList(var),
    )


def build_phi_model(root_mod, var, tag, x_range: tuple[float, float], background_kind="polynomial"):
    mean = root_mod.RooRealVar(f"mean_{tag}", f"mean_{tag}", 1.019, 1.005, 1.035)
    sigma = root_mod.RooRealVar(f"sigma_{tag}", f"sigma_{tag}", 0.004, 0.0005, 0.03)
    signal = root_mod.RooGaussian(f"sig_{tag}", f"sig_{tag}", var, mean, sigma)
    t_var = _normalized_axis_formula(root_mod, var, tag, x_range)
    coeff_vars = [
        root_mod.RooRealVar(f"phi_c{idx}_{tag}", f"phi_c{idx}_{tag}", 0.0, -2.0 * (10 ** idx), 2.0 * (10 ** idx))
        for idx in range(1, 5)
    ]
    if background_kind == "chebychev":
        coeffs = root_mod.RooArgList()
        for coeff in coeff_vars:
            coeffs.add(coeff)
        background = root_mod.RooChebychev(f"bkg_{tag}", f"bkg_{tag}", t_var, coeffs)
    elif background_kind == "polynomial":
        background = root_mod.RooGenericPdf(
            f"bkg_{tag}",
            "1.0 + @1*@0 + @2*@0*@0 + @3*@0*@0*@0 + @4*@0*@0*@0*@0",
            root_mod.RooArgList(t_var, *coeff_vars),
        )
    else:
        raise ValueError(f"Unknown phi background kind: {background_kind}")
    state = {
        "role": "phi",
        "var": var,
        "t_var": t_var,
        "mean": mean,
        "sigma": sigma,
        "signal": signal,
        "background": background,
        "coeff_vars": coeff_vars,
        "background_kind": background_kind,
        "background_degree": 4,
    }
    ROOFIT_STATE[f"phi_{tag}"] = state
    return state


def build_ups_model(root_mod, var, tag: str, x_range: tuple[float, float], background_order: int = 4):
    if background_order not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported Upsilon background order {background_order}; expected an integer in [1, 4].")
    mean_1s = root_mod.RooRealVar(f"mean_{tag}", f"mean_{tag}", UPS_1S_MASS, max(x_range[0], UPS_1S_MASS - 0.2), min(x_range[1], UPS_1S_MASS + 0.2))
    sigma = root_mod.RooRealVar(f"sigma_{tag}", f"sigma_{tag}", 0.12, 0.02, 0.60)
    mean_2s = root_mod.RooFormulaVar(f"mean2_{tag}", f"@0+{UPS_2S_DELTA:.8f}", root_mod.RooArgList(mean_1s))
    mean_3s = root_mod.RooFormulaVar(f"mean3_{tag}", f"@0+{UPS_3S_DELTA:.8f}", root_mod.RooArgList(mean_1s))
    gauss_1s = root_mod.RooGaussian(f"g1_{tag}", f"g1_{tag}", var, mean_1s, sigma)
    gauss_2s = root_mod.RooGaussian(f"g2_{tag}", f"g2_{tag}", var, mean_2s, sigma)
    gauss_3s = root_mod.RooGaussian(f"g3_{tag}", f"g3_{tag}", var, mean_3s, sigma)
    frac_excited = root_mod.RooRealVar(f"ups_frac_excited_{tag}", f"ups_frac_excited_{tag}", 0.30, 0.0, 1.0)
    frac_3s_in_excited = root_mod.RooRealVar(f"ups_frac_3s_in_excited_{tag}", f"ups_frac_3s_in_excited_{tag}", 0.33, 0.0, 1.0)
    frac_1s = root_mod.RooFormulaVar(f"ups_frac1_{tag}", "1.0-@0", root_mod.RooArgList(frac_excited))
    frac_2s = root_mod.RooFormulaVar(
        f"ups_frac2_{tag}",
        "@0*(1.0-@1)",
        root_mod.RooArgList(frac_excited, frac_3s_in_excited),
    )
    frac_3s = root_mod.RooFormulaVar(
        f"ups_frac3_{tag}",
        "@0*@1",
        root_mod.RooArgList(frac_excited, frac_3s_in_excited),
    )
    excited_pdfs = root_mod.RooArgList()
    excited_pdfs.add(gauss_3s)
    excited_pdfs.add(gauss_2s)
    excited_fracs = root_mod.RooArgList()
    excited_fracs.add(frac_3s_in_excited)
    excited_signal = root_mod.RooAddPdf(f"sig_excited_{tag}", f"sig_excited_{tag}", excited_pdfs, excited_fracs, False)
    signal_pdfs = root_mod.RooArgList()
    signal_pdfs.add(excited_signal)
    signal_pdfs.add(gauss_1s)
    signal_fracs = root_mod.RooArgList()
    signal_fracs.add(frac_excited)
    signal = root_mod.RooAddPdf(f"sig_{tag}", f"sig_{tag}", signal_pdfs, signal_fracs, False)
    t_var = _normalized_axis_formula(root_mod, var, tag, x_range)
    coeff_vars = [
        root_mod.RooRealVar(f"ups_c{idx}_{tag}", f"ups_c{idx}_{tag}", 0.0, -2.0 * (10 ** idx), 2.0 * (10 ** idx))
        for idx in range(1, background_order + 1)
    ]
    formula_terms = ["1.0"] + [f"@{idx}*pow(@0,{idx})" for idx in range(1, background_order + 1)]
    background = root_mod.RooGenericPdf(
        f"bkg_{tag}",
        " + ".join(formula_terms),
        root_mod.RooArgList(t_var, *coeff_vars),
    )
    return {
        "role": "ups",
        "var": var,
        "t_var": t_var,
        "mean": mean_1s,
        "mean_2s": mean_2s,
        "mean_3s": mean_3s,
        "sigma": sigma,
        "gauss_1s": gauss_1s,
        "gauss_2s": gauss_2s,
        "gauss_3s": gauss_3s,
        "excited_signal": excited_signal,
        "frac_excited": frac_excited,
        "frac_3s_in_excited": frac_3s_in_excited,
        "frac_1s": frac_1s,
        "frac_2s": frac_2s,
        "frac_3s": frac_3s,
        "signal": signal,
        "background": background,
        "coeff_vars": coeff_vars,
        "background_order": background_order,
        "locked_params": [],
    }


def make_yield(root_mod, name, initial, upper):
    return root_mod.RooRealVar(name, name, initial, 0.0, upper)


def ups_signal_fractions(frac_excited: float, frac_3s_in_excited: float) -> dict[str, float] | None:
    frac_excited = float(frac_excited)
    frac_3s_in_excited = float(frac_3s_in_excited)
    if not (0.0 <= frac_excited <= 1.0):
        return None
    if not (0.0 <= frac_3s_in_excited <= 1.0):
        return None
    frac_3s = frac_excited * frac_3s_in_excited
    frac_2s = frac_excited * (1.0 - frac_3s_in_excited)
    frac_1s = 1.0 - frac_excited
    return {
        "ups_1s_fraction": frac_1s,
        "ups_2s_fraction": frac_2s,
        "ups_3s_fraction": frac_3s,
        "ups_excited_fraction": frac_excited,
    }


def ups_signal_yields(total_signal_yield: float, frac_excited: float, frac_3s_in_excited: float) -> dict[str, float] | None:
    fractions = ups_signal_fractions(frac_excited, frac_3s_in_excited)
    if fractions is None:
        return None
    total_signal_yield = float(total_signal_yield)
    return {
        "ups_1s_signal_yield": total_signal_yield * fractions["ups_1s_fraction"],
        "ups_2s_signal_yield": total_signal_yield * fractions["ups_2s_fraction"],
        "ups_3s_signal_yield": total_signal_yield * fractions["ups_3s_fraction"],
        "ups_excited_signal_yield": total_signal_yield * fractions["ups_excited_fraction"],
    }


def profile_component_significance(
    *,
    full_nll: float,
    null_nll: float,
    signal_yield: float,
    full_fit_valid: bool,
    null_fit_valid: bool,
) -> dict[str, float]:
    if not full_fit_valid or not null_fit_valid:
        return {
            "delta_nll": np.nan,
            "q0": np.nan,
            "p0": np.nan,
            "significance_sigma": np.nan,
        }
    delta_nll = float(null_nll) - float(full_nll)
    q0 = max(0.0, 2.0 * delta_nll)
    z = math.sqrt(q0) if float(signal_yield) > 0.0 else 0.0
    p0 = 0.5 * math.erfc(z / math.sqrt(2.0))
    return {"delta_nll": delta_nll, "q0": q0, "p0": p0, "significance_sigma": z}


def one_sided_profile_significance(full_nll: float, null_nll: float, signal_yield: float) -> dict[str, float]:
    generic = profile_component_significance(
        full_nll=full_nll,
        null_nll=null_nll,
        signal_yield=signal_yield,
        full_fit_valid=True,
        null_fit_valid=True,
    )
    return {
        "delta_nll": generic["delta_nll"],
        "q0": generic["q0"],
        "p0": generic["p0"],
        "N_sss_significance_sigma": generic["significance_sigma"],
    }


def roofit_fit_is_valid(result: Any) -> bool:
    return bool(result.status() == 0 and result.covQual() >= 2)


def build_ups_peak_significance_table(
    *,
    full_nll: float,
    full_fit_valid: bool,
    total_signal_yield: float,
    frac_excited: float,
    frac_3s_in_excited: float,
    null_results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    fractions = ups_signal_fractions(frac_excited, frac_3s_in_excited)
    yields = ups_signal_yields(total_signal_yield, frac_excited, frac_3s_in_excited)
    rows: list[dict[str, Any]] = []
    for component, fraction_key, yield_key in UPS_PEAK_COMPONENTS:
        null_info = null_results.get(component, {})
        signal_fraction = np.nan if fractions is None else float(fractions[fraction_key])
        signal_yield = np.nan if yields is None else float(yields[yield_key])
        metrics = profile_component_significance(
            full_nll=full_nll,
            null_nll=float(null_info.get("nll", np.nan)),
            signal_yield=signal_yield,
            full_fit_valid=full_fit_valid,
            null_fit_valid=bool(null_info.get("fit_valid", False)),
        )
        rows.append(
            {
                "component": component,
                "signal_fraction": signal_fraction,
                "signal_yield": signal_yield,
                "null_hypothesis": null_info.get("null_hypothesis", f"no {component} model"),
                "full_fit_valid": int(bool(full_fit_valid)),
                "null_fit_valid": int(bool(null_info.get("fit_valid", False))),
                "delta_nll": metrics["delta_nll"],
                "q0": metrics["q0"],
                "p0": metrics["p0"],
                "significance_sigma": metrics["significance_sigma"],
            }
        )
    return pd.DataFrame(rows)


def roofit_parameter_table(yield_vars: Mapping[str, Any], axis_states: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_param(name: str, obj: Any, group: str) -> None:
        if not hasattr(obj, "getVal"):
            return
        bounds = None
        if hasattr(obj, "hasMin") and hasattr(obj, "hasMax") and obj.hasMin() and obj.hasMax():
            bounds = (float(obj.getMin()), float(obj.getMax()))
        rows.append(
            {
                "group": group,
                "parameter": name,
                "value": float(obj.getVal()),
                "error": float(obj.getError()) if hasattr(obj, "getError") else np.nan,
                "constant": int(bool(obj.isConstant())) if hasattr(obj, "isConstant") else np.nan,
                "bounds": bounds,
            }
        )

    for name, obj in yield_vars.items():
        add_param(name, obj, "yield")
    for branch, state in axis_states.items():
        group = branch.replace("_mass", "")
        if state["role"] == "jpsi":
            for key in ("mean", "sigma_cb", "alpha", "n", "sigma_g", "frac", "slope"):
                add_param(f"{branch}.{key}", state[key], group)
        elif state["role"] == "phi":
            add_param(f"{branch}.mean", state["mean"], group)
            add_param(f"{branch}.sigma", state["sigma"], group)
            for idx, coeff in enumerate(state["coeff_vars"], start=1):
                add_param(f"{branch}.c{idx}", coeff, group)
        elif state["role"] == "ups":
            for key in ("mean", "sigma", "frac_excited", "frac_3s_in_excited"):
                add_param(f"{branch}.{key}", state[key], group)
            for idx, coeff in enumerate(state["coeff_vars"], start=1):
                add_param(f"{branch}.c{idx}", coeff, group)
    return pd.DataFrame(rows)


def sanity_check_pdf(root_mod, pdf, var, points):
    norm = root_mod.RooArgSet(var)
    values = []
    for x in points:
        var.setVal(float(x))
        values.append(float(pdf.getVal(norm)))
    return values


def _axis_title(branch: str) -> str:
    if branch == "Phi_mass":
        return r"\phi"
    if branch == "Ups_mass":
        return r"\Upsilon"
    if branch == "Jpsi_1_mass":
        return r"J/\psi_1"
    if branch == "Jpsi_2_mass":
        return r"J/\psi_2"
    return branch


def _build_axis_state(root_module, axis, windows, jpsi1_pdf_config, jpsi2_pdf_config, phi_background_kind, ups_background_order):
    branch = axis.branch
    if axis.role == "jpsi":
        cfg = jpsi1_pdf_config if axis.object_name == "Jpsi_1" else (jpsi2_pdf_config or jpsi1_pdf_config)
        var = root_module.RooRealVar(branch, branch, windows[branch][0], windows[branch][1])
        return var, build_jpsi_model(root_module, var, axis.object_name.lower(), pdf_config=cfg)
    if axis.role == "phi":
        var = root_module.RooRealVar(branch, branch, windows[branch][0], windows[branch][1])
        return var, build_phi_model(root_module, var, axis.object_name.lower(), windows[branch], background_kind=phi_background_kind)
    if axis.role == "ups":
        var = root_module.RooRealVar(branch, branch, windows[branch][0], windows[branch][1])
        return var, build_ups_model(root_module, var, axis.object_name.lower(), windows[branch], background_order=ups_background_order)
    raise KeyError(f"Unsupported axis role '{axis.role}'.")


def run_roofit_3d_jpsijpsiphi(
    root_module,
    fit_df: pd.DataFrame,
    windows: dict[str, tuple[float, float]],
    analysis_mode: str,
    phi_background_kind: str = "polynomial",
    ups_background_order: int = 4,
    jpsi1_pdf_config: JpsiPdfConfig | None = None,
    jpsi2_pdf_config: JpsiPdfConfig | None = None,
) -> dict[str, object]:
    reset_roofit_state()
    if fit_df.empty:
        raise RuntimeError("No candidates survive the active mass windows; cannot build the RooFit sample.")
    mode_spec = get_analysis_mode_spec(analysis_mode)
    fit_vars: dict[str, Any] = {}
    axis_states: dict[str, dict[str, Any]] = {}
    for axis in mode_spec.axes:
        var, state = _build_axis_state(
            root_module,
            axis,
            windows,
            jpsi1_pdf_config or DEFAULT_JPSI_PDF_CONFIG,
            jpsi2_pdf_config or jpsi1_pdf_config or DEFAULT_JPSI_PDF_CONFIG,
            phi_background_kind,
            ups_background_order,
        )
        fit_vars[axis.branch] = var
        axis_states[axis.branch] = state
    ROOFIT_STATE["fit_vars"] = fit_vars
    dataset = make_roodataset(root_module, fit_df, fit_vars)
    if dataset.numEntries() != len(fit_df):
        raise RuntimeError(f"RooDataSet entry mismatch: dataset={dataset.numEntries()} vs fit_df={len(fit_df)}")

    n_total = max(1, len(fit_df))
    component_patterns = {
        "pdf_sss": "sss",
        "pdf_ssb": "ssb",
        "pdf_sbs": "sbs",
        "pdf_bss": "bss",
        "pdf_sbb": "sbb",
        "pdf_bsb": "bsb",
        "pdf_bbs": "bbs",
        "pdf_bbb": "bbb",
    }
    prod_pdfs = {}
    for name, pattern in component_patterns.items():
        arglist = root_module.RooArgList()
        for axis_index, axis in enumerate(mode_spec.axes):
            state = axis_states[axis.branch]
            arglist.add(state["signal"] if pattern[axis_index] == "s" else state["background"])
        prod_pdfs[name] = {"pattern": pattern, "arglist": arglist, "pdf": root_module.RooProdPdf(name, name, arglist)}

    yield_names = ["N_sss", "N_ssb", "N_sbs", "N_bss", "N_sbb", "N_bsb", "N_bbs", "N_bbb"]
    yield_initials = [0.08, 0.05, 0.05, 0.05, 0.12, 0.12, 0.12, 0.41]
    yield_vars = {name: make_yield(root_module, name, frac * n_total, 2.0 * n_total) for name, frac in zip(yield_names, yield_initials)}

    pdf_list = root_module.RooArgList()
    yield_list = root_module.RooArgList()
    for key in ("pdf_sss", "pdf_ssb", "pdf_sbs", "pdf_bss", "pdf_sbb", "pdf_bsb", "pdf_bbs", "pdf_bbb"):
        pdf_list.add(prod_pdfs[key]["pdf"])
    for name in yield_names:
        yield_list.add(yield_vars[name])
    model3d = root_module.RooAddPdf("model3d", "model3d", pdf_list, yield_list)

    fit_result = model3d.fitTo(
        dataset,
        root_module.RooFit.Save(True),
        root_module.RooFit.Extended(True),
        root_module.RooFit.NumCPU(4),
        root_module.RooFit.Strategy(1),
        root_module.RooFit.Minimizer("Minuit2", "migrad"),
        root_module.RooFit.PrintLevel(-1),
    )

    def capture_snapshot(*items: Any) -> dict[int, dict[str, object]]:
        nominal_snapshot: dict[int, dict[str, object]] = {}

        def visit_snapshot(item) -> None:
            if isinstance(item, Mapping):
                for value in item.values():
                    visit_snapshot(value)
                return
            if isinstance(item, (list, tuple)):
                for value in item:
                    visit_snapshot(value)
                return
            if hasattr(item, "getVal") and hasattr(item, "setVal"):
                nominal_snapshot[id(item)] = {
                    "obj": item,
                    "value": float(item.getVal()),
                    "constant": bool(item.isConstant()) if hasattr(item, "isConstant") else False,
                }

        for item in items:
            visit_snapshot(item)
        return nominal_snapshot

    def restore_snapshot(snapshot: Mapping[int, Mapping[str, object]]) -> None:
        for state in snapshot.values():
            obj = state["obj"]
            obj.setVal(float(state["value"]))
            if hasattr(obj, "setConstant"):
                obj.setConstant(bool(state["constant"]))

    def run_constrained_fit(constraints: list[tuple[Any, float]]) -> Any:
        snapshot = capture_snapshot(yield_vars, axis_states)
        for obj, value in constraints:
            obj.setVal(float(value))
            obj.setConstant(True)
        fit_result_local = model3d.fitTo(
            dataset,
            root_module.RooFit.Save(True),
            root_module.RooFit.Extended(True),
            root_module.RooFit.NumCPU(4),
            root_module.RooFit.Strategy(1),
            root_module.RooFit.Minimizer("Minuit2", "migrad"),
            root_module.RooFit.PrintLevel(-1),
        )
        restore_snapshot(snapshot)
        return fit_result_local
    null_fit_result = run_constrained_fit([(yield_vars["N_sss"], 0.0)])

    significance = one_sided_profile_significance(
        full_nll=fit_result.minNll(),
        null_nll=null_fit_result.minNll(),
        signal_yield=yield_vars["N_sss"].getVal(),
    )
    full_fit_valid = roofit_fit_is_valid(fit_result)

    ups_peak_significance_table = pd.DataFrame(
        columns=[
            "component",
            "signal_fraction",
            "signal_yield",
            "null_hypothesis",
            "full_fit_valid",
            "null_fit_valid",
            "delta_nll",
            "q0",
            "p0",
            "significance_sigma",
        ]
    )
    ups_null_results: dict[str, dict[str, Any]] = {}
    if "Ups_mass" in axis_states:
        ups_state = axis_states["Ups_mass"]
        ups_null_results = {
            "Ups_2S": {
                "null_hypothesis": "no 2S model",
                "fit_result": run_constrained_fit([(ups_state["frac_3s_in_excited"], 1.0)]),
            },
            "Ups_3S": {
                "null_hypothesis": "no 3S model",
                "fit_result": run_constrained_fit([(ups_state["frac_3s_in_excited"], 0.0)]),
            },
            "Ups_excited": {
                "null_hypothesis": "no 2S or 3S model",
                "fit_result": run_constrained_fit(
                    [
                        (ups_state["frac_excited"], 0.0),
                        (ups_state["frac_3s_in_excited"], 0.0),
                    ]
                ),
            },
        }
        for component, null_info in ups_null_results.items():
            null_result = null_info["fit_result"]
            null_info["nll"] = float(null_result.minNll())
            null_info["fit_valid"] = roofit_fit_is_valid(null_result)
        ups_peak_significance_table = build_ups_peak_significance_table(
            full_nll=float(fit_result.minNll()),
            full_fit_valid=full_fit_valid,
            total_signal_yield=float(yield_vars["N_sss"].getVal()),
            frac_excited=float(ups_state["frac_excited"].getVal()),
            frac_3s_in_excited=float(ups_state["frac_3s_in_excited"].getVal()),
            null_results=ups_null_results,
        )

    ROOFIT_STATE.update(
        {
            "prod_pdfs": prod_pdfs,
            "yield_vars": yield_vars,
            "yield_names": yield_names,
            "model3d": model3d,
            "fit_result": fit_result,
            "null_fit_result": null_fit_result,
            "ups_null_results": ups_null_results,
            "axis_states": axis_states,
            "analysis_mode": analysis_mode,
        }
    )

    jpsi1_locked = axis_states.get("Jpsi_1_mass", {}).get("locked_params", [])
    jpsi2_locked = axis_states.get("Jpsi_2_mass", {}).get("locked_params", [])
    fit_summary_rows = [
        {"quantity": "analysis_mode", "value": analysis_mode},
        {"quantity": "status", "value": fit_result.status()},
        {"quantity": "covQual", "value": fit_result.covQual()},
        {"quantity": "EDM", "value": fit_result.edm()},
        {"quantity": "minNll", "value": fit_result.minNll()},
        {"quantity": "N_fit_events", "value": len(fit_df)},
        {"quantity": "jpsi1_locked_params", "value": ",".join(jpsi1_locked) or "(none)"},
        {"quantity": "jpsi2_locked_params", "value": ",".join(jpsi2_locked) or "(n/a)"},
        {"quantity": "N_sss (3D signal event yield)", "value": yield_vars["N_sss"].getVal()},
        {"quantity": "N_sss_err", "value": yield_vars["N_sss"].getError()},
        {"quantity": "null_status", "value": null_fit_result.status()},
        {"quantity": "null_covQual", "value": null_fit_result.covQual()},
        {"quantity": "null_minNll", "value": null_fit_result.minNll()},
        {"quantity": "delta_nll", "value": significance["delta_nll"]},
        {"quantity": "q0", "value": significance["q0"]},
        {"quantity": "p0", "value": significance["p0"]},
        {"quantity": "N_sss_significance_sigma", "value": significance["N_sss_significance_sigma"]},
    ]
    if "Phi_mass" in axis_states:
        fit_summary_rows.append({"quantity": "phi_background_kind", "value": axis_states["Phi_mass"]["background_kind"]})
        fit_summary_rows.append({"quantity": "phi_background_degree", "value": axis_states["Phi_mass"]["background_degree"]})
    if "Ups_mass" in axis_states:
        fit_summary_rows.append({"quantity": "ups_background_order", "value": axis_states["Ups_mass"]["background_order"]})
        for row in ups_peak_significance_table.itertuples(index=False):
            fit_summary_rows.extend(
                [
                    {"quantity": f"{row.component}_signal_fraction", "value": row.signal_fraction},
                    {"quantity": f"{row.component}_signal_yield", "value": row.signal_yield},
                    {"quantity": f"{row.component}_null_fit_valid", "value": row.null_fit_valid},
                    {"quantity": f"{row.component}_delta_nll", "value": row.delta_nll},
                    {"quantity": f"{row.component}_q0", "value": row.q0},
                    {"quantity": f"{row.component}_p0", "value": row.p0},
                    {"quantity": f"{row.component}_significance_sigma", "value": row.significance_sigma},
                ]
            )
    fit_summary = pd.DataFrame(fit_summary_rows)
    yield_table = pd.DataFrame([{"yield": name, "value": yield_vars[name].getVal(), "error": yield_vars[name].getError()} for name in yield_names])
    parameter_table = roofit_parameter_table(yield_vars, axis_states)

    payload = {
        "root_module": root_module,
        "dataset": dataset,
        "fit_result": fit_result,
        "fit_summary": fit_summary,
        "parameter_table": parameter_table,
        "yield_table": yield_table,
        "yield_names": yield_names,
        "yield_vars": yield_vars,
        "fit_vars": fit_vars,
        "axis_states": axis_states,
        "null_fit_result": null_fit_result,
        "ups_peak_significance_table": ups_peak_significance_table,
    }
    if "Phi_mass" in axis_states:
        phi = fit_vars["Phi_mass"]
        phi_state = axis_states["Phi_mass"]
        payload["phi_debug_values"] = sanity_check_pdf(
            root_module,
            phi_state["background"],
            phi,
            [
                windows["Phi_mass"][0] + 0.1 * (windows["Phi_mass"][1] - windows["Phi_mass"][0]),
                windows["Phi_mass"][0] + 0.3 * (windows["Phi_mass"][1] - windows["Phi_mass"][0]),
                windows["Phi_mass"][0] + 0.5 * (windows["Phi_mass"][1] - windows["Phi_mass"][0]),
                windows["Phi_mass"][0] + 0.7 * (windows["Phi_mass"][1] - windows["Phi_mass"][0]),
                windows["Phi_mass"][0] + 0.9 * (windows["Phi_mass"][1] - windows["Phi_mass"][0]),
            ],
        )
    return payload


def projection_specs(fit_payload: dict[str, object], plot_specs: list[dict[str, object]] | None = None) -> dict[str, dict[str, object]]:
    yield_vars = fit_payload["yield_vars"]
    fit_vars = fit_payload["fit_vars"]
    axis_states = fit_payload["axis_states"]
    fit_branches = tuple(fit_payload["fit_branches"])
    required = set(fit_branches)
    if plot_specs is None:
        plot_spec_map = {}
        for branch in fit_branches:
            xlabel = r"$m_{KK}$ [GeV]" if branch == "Phi_mass" else r"$m_{\mu\mu}$ [GeV]"
            plot_spec_map[branch] = {"xlabel": xlabel, "bins": 60, "title": rf"3D-fit projection on ${_axis_title(branch)}$"}
    else:
        plot_spec_map = {str(spec["branch"]): spec for spec in plot_specs}
        missing = sorted(required - set(plot_spec_map))
        if missing:
            raise KeyError(f"Missing fit plot specs for: {missing}")

    payloads: dict[str, dict[str, object]] = {}
    for axis_index, branch in enumerate(fit_branches):
        signal_total = 0.0
        background_total = 0.0
        for name, yield_var in yield_vars.items():
            pattern = name.removeprefix("N_")
            if pattern[axis_index] == "s":
                signal_total += yield_var.getVal()
            else:
                background_total += yield_var.getVal()
        state = axis_states[branch]
        payloads[branch] = {
            "signal": signal_total,
            "background": background_total,
            "signal_pdf": state["signal"],
            "background_pdf": state["background"],
            "var": fit_vars[branch],
            "xlabel": str(plot_spec_map[branch]["xlabel"]),
            "title": str(plot_spec_map[branch].get("title", rf"3D-fit projection on ${_axis_title(branch)}$")),
            "bins": int(plot_spec_map[branch]["bins"]),
        }
    return payloads
