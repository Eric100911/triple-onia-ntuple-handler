from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import crystalball, norm

from .fit_roofit import (
    DEFAULT_JPSI_PDF_CONFIG,
    FitParamSpec,
    JpsiPdfConfig,
    UPS_1S_MASS,
    UPS_2S_DELTA,
    UPS_3S_DELTA,
    build_ups_peak_significance_table,
    build_fit_frame,
    locked_param_names,
    one_sided_profile_significance,
    ups_signal_fractions,
)
from .schema import get_analysis_mode_spec


def truncated_gauss_pdf(x, mean, sigma, xmin, xmax):
    z_hi = (xmax - mean) / sigma
    z_lo = (xmin - mean) / sigma
    norm_int = norm.cdf(z_hi) - norm.cdf(z_lo)
    return norm.pdf(x, loc=mean, scale=sigma) / np.clip(norm_int, 1e-12, None)


def truncated_cb_pdf(x, mean, sigma, alpha, n, xmin, xmax):
    beta = abs(alpha)
    shape_n = max(n, 1.01)
    z_hi = (xmax - mean) / sigma
    z_lo = (xmin - mean) / sigma
    norm_int = crystalball.cdf(z_hi, beta, shape_n) - crystalball.cdf(z_lo, beta, shape_n)
    return crystalball.pdf((x - mean) / sigma, beta, shape_n) / np.clip(sigma * norm_int, 1e-12, None)


def jpsi_signal_pdf(x, mean, sigma_cb, alpha, n, sigma_g, frac, xmin, xmax):
    return frac * truncated_cb_pdf(x, mean, sigma_cb, alpha, n, xmin, xmax) + (1.0 - frac) * truncated_gauss_pdf(x, mean, sigma_g, xmin, xmax)


def expo_pdf(x, slope, xmin, xmax):
    if abs(slope) < 1e-12:
        return np.full_like(x, 1.0 / (xmax - xmin), dtype=float)
    norm_factor = (np.exp(slope * xmax) - np.exp(slope * xmin)) / slope
    return np.exp(slope * x) / np.clip(norm_factor, 1e-12, None)


def phi_signal_pdf(x, mean, sigma, xmin, xmax):
    return truncated_gauss_pdf(x, mean, sigma, xmin, xmax)


def normalized_poly_pdf(x, coeffs, xmin, xmax):
    t = 2.0 * (x - xmin) / (xmax - xmin) - 1.0
    raw = np.ones_like(t, dtype=float)
    for order, coeff in enumerate(coeffs, start=1):
        raw = raw + coeff * t**order
    if np.any(raw <= 0.0):
        return None
    integral = np.trapezoid(raw, x)
    if integral <= 0.0:
        return None
    return raw / integral


def phi_poly4_pdf(x, c1, c2, c3, c4, xmin, xmax):
    return normalized_poly_pdf(x, (c1, c2, c3, c4), xmin, xmax)


def ups_signal_pdf(x, mean, sigma, frac_excited, frac_3s_in_excited, xmin, xmax):
    fractions = ups_signal_fractions(frac_excited, frac_3s_in_excited)
    if fractions is None:
        return None
    return (
        fractions["ups_1s_fraction"] * truncated_gauss_pdf(x, mean, sigma, xmin, xmax)
        + fractions["ups_2s_fraction"] * truncated_gauss_pdf(x, mean + UPS_2S_DELTA, sigma, xmin, xmax)
        + fractions["ups_3s_fraction"] * truncated_gauss_pdf(x, mean + UPS_3S_DELTA, sigma, xmin, xmax)
    )


def apply_minuit_param_spec(minuit_obj, name: str, spec: FitParamSpec) -> None:
    minuit_obj.values[name] = spec.value
    if spec.bounds is not None:
        minuit_obj.limits[name] = spec.bounds
    minuit_obj.fixed[name] = bool(spec.constant)


def projection_curves_iminuit(
    minuit_obj,
    branch: str,
    x,
    windows: dict[str, tuple[float, float]],
    analysis_mode: str,
    ups_background_order: int = 4,
):
    fit_branches = get_analysis_mode_spec(analysis_mode).fit_branches
    axis_index = fit_branches.index(branch)
    p = minuit_obj.values.to_dict()

    if branch == "Jpsi_1_mass":
        signal_pdf = jpsi_signal_pdf(x, p["j1_mean"], p["j1_sigma_cb"], p["j1_alpha"], p["j1_n"], p["j1_sigma_g"], p["j1_frac"], *windows[branch])
        background_pdf = expo_pdf(x, p["j1_slope"], *windows[branch])
    elif branch == "Jpsi_2_mass":
        signal_pdf = jpsi_signal_pdf(x, p["j2_mean"], p["j2_sigma_cb"], p["j2_alpha"], p["j2_n"], p["j2_sigma_g"], p["j2_frac"], *windows[branch])
        background_pdf = expo_pdf(x, p["j2_slope"], *windows[branch])
    elif branch == "Phi_mass":
        signal_pdf = phi_signal_pdf(x, p["phi_mean"], p["phi_sigma"], *windows[branch])
        background_pdf = normalized_poly_pdf(x, (p["phi_c1"], p["phi_c2"], p["phi_c3"], p["phi_c4"]), *windows[branch])
    elif branch == "Ups_mass":
        signal_pdf = ups_signal_pdf(x, p["ups_mean"], p["ups_sigma"], p["ups_frac_excited"], p["ups_frac_3s_in_excited"], *windows[branch])
        coeffs = tuple(p[f"ups_c{idx}"] for idx in range(1, ups_background_order + 1))
        background_pdf = normalized_poly_pdf(x, coeffs, *windows[branch])
    else:
        raise KeyError(branch)
    if background_pdf is None:
        background_pdf = np.zeros_like(x, dtype=float)

    signal = 0.0
    background = 0.0
    for yield_name in ("N_sss", "N_ssb", "N_sbs", "N_bss", "N_sbb", "N_bsb", "N_bbs", "N_bbb"):
        pattern = yield_name.removeprefix("N_")
        if pattern[axis_index] == "s":
            signal += p[yield_name]
        else:
            background += p[yield_name]
    return signal * signal_pdf, background * background_pdf


def minuit_parameter_table(minuit_obj, used_parameters: set[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name in minuit_obj.parameters:
        if used_parameters is not None and name not in used_parameters:
            continue
        limits = minuit_obj.limits[name]
        bounds = None
        if limits is not None:
            low = None if limits[0] is None else float(limits[0])
            high = None if limits[1] is None else float(limits[1])
            bounds = (low, high)
        if name.startswith("N_"):
            group = "yield"
        elif name.startswith("phi_"):
            group = "phi"
        elif name.startswith("ups_"):
            group = "ups"
        elif name.startswith("j1_"):
            group = "jpsi1"
        elif name.startswith("j2_"):
            group = "jpsi2"
        else:
            group = "other"
        rows.append(
            {
                "group": group,
                "parameter": name,
                "value": float(minuit_obj.values[name]),
                "error": float(minuit_obj.errors[name]),
                "constant": int(bool(minuit_obj.fixed[name])),
                "bounds": bounds,
            }
        )
    return pd.DataFrame(rows)


def run_iminuit_3d_jpsijpsiphi(
    Minuit,
    fit_df: pd.DataFrame,
    windows: dict[str, tuple[float, float]],
    analysis_mode: str,
    ups_background_order: int = 4,
    jpsi1_pdf_config: JpsiPdfConfig | None = None,
    jpsi2_pdf_config: JpsiPdfConfig | None = None,
) -> dict[str, object]:
    if fit_df.empty:
        raise RuntimeError("No candidates survive the active mass windows; cannot build the iminuit fit sample.")

    mode_spec = get_analysis_mode_spec(analysis_mode)
    fit_branches = mode_spec.fit_branches
    jpsi1_pdf_config = jpsi1_pdf_config or DEFAULT_JPSI_PDF_CONFIG
    jpsi2_pdf_config = jpsi2_pdf_config or jpsi1_pdf_config
    data = {branch: fit_df[branch].to_numpy(dtype=float) for branch in fit_branches}

    def axis_signal_background(branch: str, params: dict[str, float]):
        if branch == "Jpsi_1_mass":
            return (
                jpsi_signal_pdf(data[branch], params["j1_mean"], params["j1_sigma_cb"], params["j1_alpha"], params["j1_n"], params["j1_sigma_g"], params["j1_frac"], *windows[branch]),
                expo_pdf(data[branch], params["j1_slope"], *windows[branch]),
            )
        if branch == "Jpsi_2_mass":
            return (
                jpsi_signal_pdf(data[branch], params["j2_mean"], params["j2_sigma_cb"], params["j2_alpha"], params["j2_n"], params["j2_sigma_g"], params["j2_frac"], *windows[branch]),
                expo_pdf(data[branch], params["j2_slope"], *windows[branch]),
            )
        if branch == "Phi_mass":
            return (
                phi_signal_pdf(data[branch], params["phi_mean"], params["phi_sigma"], *windows[branch]),
                normalized_poly_pdf(data[branch], (params["phi_c1"], params["phi_c2"], params["phi_c3"], params["phi_c4"]), *windows[branch]),
            )
        if branch == "Ups_mass":
            coeffs = tuple(params[f"ups_c{idx}"] for idx in range(1, ups_background_order + 1))
            return (
                ups_signal_pdf(
                    data[branch],
                    params["ups_mean"],
                    params["ups_sigma"],
                    params["ups_frac_excited"],
                    params["ups_frac_3s_in_excited"],
                    *windows[branch],
                ),
                normalized_poly_pdf(data[branch], coeffs, *windows[branch]),
            )
        raise KeyError(branch)

    def total_intensity(
        N_sss, N_ssb, N_sbs, N_bss, N_sbb, N_bsb, N_bbs, N_bbb,
        j1_mean, j1_sigma_cb, j1_alpha, j1_n, j1_sigma_g, j1_frac, j1_slope,
        j2_mean, j2_sigma_cb, j2_alpha, j2_n, j2_sigma_g, j2_frac, j2_slope,
        phi_mean, phi_sigma, phi_c1, phi_c2, phi_c3, phi_c4,
        ups_mean, ups_sigma, ups_frac_excited, ups_frac_3s_in_excited, ups_c1, ups_c2, ups_c3, ups_c4,
    ):
        params = {
            "N_sss": N_sss, "N_ssb": N_ssb, "N_sbs": N_sbs, "N_bss": N_bss, "N_sbb": N_sbb, "N_bsb": N_bsb, "N_bbs": N_bbs, "N_bbb": N_bbb,
            "j1_mean": j1_mean, "j1_sigma_cb": j1_sigma_cb, "j1_alpha": j1_alpha, "j1_n": j1_n, "j1_sigma_g": j1_sigma_g, "j1_frac": j1_frac, "j1_slope": j1_slope,
            "j2_mean": j2_mean, "j2_sigma_cb": j2_sigma_cb, "j2_alpha": j2_alpha, "j2_n": j2_n, "j2_sigma_g": j2_sigma_g, "j2_frac": j2_frac, "j2_slope": j2_slope,
            "phi_mean": phi_mean, "phi_sigma": phi_sigma, "phi_c1": phi_c1, "phi_c2": phi_c2, "phi_c3": phi_c3, "phi_c4": phi_c4,
            "ups_mean": ups_mean, "ups_sigma": ups_sigma,
            "ups_frac_excited": ups_frac_excited,
            "ups_frac_3s_in_excited": ups_frac_3s_in_excited,
            "ups_c1": ups_c1, "ups_c2": ups_c2, "ups_c3": ups_c3, "ups_c4": ups_c4,
        }
        axis_terms = []
        for branch in fit_branches:
            signal, background = axis_signal_background(branch, params)
            if signal is None or background is None:
                return None
            axis_terms.append((signal, background))
        intensity = 0.0
        for yield_name in ("N_sss", "N_ssb", "N_sbs", "N_bss", "N_sbb", "N_bsb", "N_bbs", "N_bbb"):
            pattern = yield_name.removeprefix("N_")
            component = float(params[yield_name])
            for axis_index, (signal, background) in enumerate(axis_terms):
                component = component * (signal if pattern[axis_index] == "s" else background)
            intensity = intensity + component
        return intensity

    def extended_nll(
        N_sss, N_ssb, N_sbs, N_bss, N_sbb, N_bsb, N_bbs, N_bbb,
        j1_mean, j1_sigma_cb, j1_alpha, j1_n, j1_sigma_g, j1_frac, j1_slope,
        j2_mean, j2_sigma_cb, j2_alpha, j2_n, j2_sigma_g, j2_frac, j2_slope,
        phi_mean, phi_sigma, phi_c1, phi_c2, phi_c3, phi_c4,
        ups_mean, ups_sigma, ups_frac_excited, ups_frac_3s_in_excited, ups_c1, ups_c2, ups_c3, ups_c4,
    ):
        total = total_intensity(
            N_sss, N_ssb, N_sbs, N_bss, N_sbb, N_bsb, N_bbs, N_bbb,
            j1_mean, j1_sigma_cb, j1_alpha, j1_n, j1_sigma_g, j1_frac, j1_slope,
            j2_mean, j2_sigma_cb, j2_alpha, j2_n, j2_sigma_g, j2_frac, j2_slope,
            phi_mean, phi_sigma, phi_c1, phi_c2, phi_c3, phi_c4,
            ups_mean, ups_sigma, ups_frac_excited, ups_frac_3s_in_excited, ups_c1, ups_c2, ups_c3, ups_c4,
        )
        if total is None or np.any(~np.isfinite(total)) or np.any(total <= 0.0):
            return 1e30
        n_exp = N_sss + N_ssb + N_sbs + N_bss + N_sbb + N_bsb + N_bbs + N_bbb
        return n_exp - np.sum(np.log(total))

    n_total = len(fit_df)

    def make_minuit(
        start_values: dict[str, float] | None = None,
        fix_n_sss_zero: bool = False,
        fixed_ups_params: dict[str, float] | None = None,
    ):
        start = {
            "N_sss": 0.08 * n_total,
            "N_ssb": 0.05 * n_total,
            "N_sbs": 0.05 * n_total,
            "N_bss": 0.05 * n_total,
            "N_sbb": 0.12 * n_total,
            "N_bsb": 0.12 * n_total,
            "N_bbs": 0.12 * n_total,
            "N_bbb": 0.41 * n_total,
            "j1_mean": jpsi1_pdf_config.mean.value,
            "j1_sigma_cb": jpsi1_pdf_config.sigma_cb.value,
            "j1_alpha": jpsi1_pdf_config.alpha.value,
            "j1_n": jpsi1_pdf_config.n.value,
            "j1_sigma_g": jpsi1_pdf_config.sigma_g.value,
            "j1_frac": jpsi1_pdf_config.frac.value,
            "j1_slope": jpsi1_pdf_config.slope.value,
            "j2_mean": jpsi2_pdf_config.mean.value,
            "j2_sigma_cb": jpsi2_pdf_config.sigma_cb.value,
            "j2_alpha": jpsi2_pdf_config.alpha.value,
            "j2_n": jpsi2_pdf_config.n.value,
            "j2_sigma_g": jpsi2_pdf_config.sigma_g.value,
            "j2_frac": jpsi2_pdf_config.frac.value,
            "j2_slope": jpsi2_pdf_config.slope.value,
            "phi_mean": 1.019,
            "phi_sigma": 0.004,
            "phi_c1": 0.0,
            "phi_c2": 0.0,
            "phi_c3": 0.0,
            "phi_c4": 0.0,
            "ups_mean": UPS_1S_MASS,
            "ups_sigma": 0.12,
            "ups_frac_excited": 0.30,
            "ups_frac_3s_in_excited": 1.0 / 3.0,
            "ups_c1": 0.0,
            "ups_c2": 0.0,
            "ups_c3": 0.0,
            "ups_c4": 0.0,
        }
        if start_values:
            start.update({key: float(value) for key, value in start_values.items()})
        if fixed_ups_params:
            start.update({key: float(value) for key, value in fixed_ups_params.items()})
        if fix_n_sss_zero:
            start["N_sss"] = 0.0
        minuit_obj = Minuit(extended_nll, **start)
        minuit_obj.errordef = Minuit.LIKELIHOOD
        for name in ("N_sss", "N_ssb", "N_sbs", "N_bss", "N_sbb", "N_bsb", "N_bbs", "N_bbb"):
            minuit_obj.limits[name] = (0.0, 2.0 * n_total)

        apply_minuit_param_spec(minuit_obj, "j1_mean", FitParamSpec(jpsi1_pdf_config.mean.value, windows["Jpsi_1_mass"], jpsi1_pdf_config.mean.constant))
        apply_minuit_param_spec(minuit_obj, "j1_sigma_cb", jpsi1_pdf_config.sigma_cb)
        apply_minuit_param_spec(minuit_obj, "j1_alpha", jpsi1_pdf_config.alpha)
        apply_minuit_param_spec(minuit_obj, "j1_n", jpsi1_pdf_config.n)
        apply_minuit_param_spec(minuit_obj, "j1_sigma_g", jpsi1_pdf_config.sigma_g)
        apply_minuit_param_spec(minuit_obj, "j1_frac", jpsi1_pdf_config.frac)
        apply_minuit_param_spec(minuit_obj, "j1_slope", jpsi1_pdf_config.slope)

        if "Jpsi_2_mass" in fit_branches:
            apply_minuit_param_spec(minuit_obj, "j2_mean", FitParamSpec(jpsi2_pdf_config.mean.value, windows["Jpsi_2_mass"], jpsi2_pdf_config.mean.constant))
            apply_minuit_param_spec(minuit_obj, "j2_sigma_cb", jpsi2_pdf_config.sigma_cb)
            apply_minuit_param_spec(minuit_obj, "j2_alpha", jpsi2_pdf_config.alpha)
            apply_minuit_param_spec(minuit_obj, "j2_n", jpsi2_pdf_config.n)
            apply_minuit_param_spec(minuit_obj, "j2_sigma_g", jpsi2_pdf_config.sigma_g)
            apply_minuit_param_spec(minuit_obj, "j2_frac", jpsi2_pdf_config.frac)
            apply_minuit_param_spec(minuit_obj, "j2_slope", jpsi2_pdf_config.slope)
        else:
            for name in ("j2_mean", "j2_sigma_cb", "j2_alpha", "j2_n", "j2_sigma_g", "j2_frac", "j2_slope"):
                minuit_obj.fixed[name] = True

        if "Phi_mass" in fit_branches:
            minuit_obj.limits["phi_mean"] = windows["Phi_mass"]
            minuit_obj.limits["phi_sigma"] = (0.0005, 0.03)
            for name in ("phi_c1", "phi_c2", "phi_c3", "phi_c4"):
                minuit_obj.limits[name] = (-2.0, 2.0)
        else:
            for name in ("phi_mean", "phi_sigma", "phi_c1", "phi_c2", "phi_c3", "phi_c4"):
                minuit_obj.fixed[name] = True

        if "Ups_mass" in fit_branches:
            minuit_obj.limits["ups_mean"] = (max(windows["Ups_mass"][0], UPS_1S_MASS - 0.2), min(windows["Ups_mass"][1], UPS_1S_MASS + 0.2))
            minuit_obj.limits["ups_sigma"] = (0.02, 0.60)
            minuit_obj.limits["ups_frac_excited"] = (0.0, 1.0)
            minuit_obj.limits["ups_frac_3s_in_excited"] = (0.0, 1.0)
            for idx in range(1, ups_background_order + 1):
                minuit_obj.limits[f"ups_c{idx}"] = (-2.0, 2.0)
            for idx in range(ups_background_order + 1, 5):
                minuit_obj.fixed[f"ups_c{idx}"] = True
        else:
            for name in ("ups_mean", "ups_sigma", "ups_frac_excited", "ups_frac_3s_in_excited", "ups_c1", "ups_c2", "ups_c3", "ups_c4"):
                minuit_obj.fixed[name] = True
        if fix_n_sss_zero:
            minuit_obj.fixed["N_sss"] = True
        if fixed_ups_params:
            for name in fixed_ups_params:
                minuit_obj.fixed[name] = True
        minuit_obj.strategy = 1
        return minuit_obj

    m = make_minuit()
    m.simplex(ncall=5000)
    m.migrad(ncall=20000)
    m.hesse(ncall=20000)

    nominal_start = {name: float(m.values[name]) for name in m.parameters}
    m_null = make_minuit(start_values=nominal_start, fix_n_sss_zero=True)
    m_null.migrad(ncall=20000)
    m_null.hesse(ncall=20000)
    significance = one_sided_profile_significance(full_nll=m.fval, null_nll=m_null.fval, signal_yield=m.values["N_sss"])
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
    ups_null_results: dict[str, dict[str, object]] = {}
    if "Ups_mass" in fit_branches:
        no_2s = make_minuit(
            start_values=nominal_start,
            fixed_ups_params={"ups_frac_3s_in_excited": 1.0},
        )
        no_2s.migrad(ncall=20000)
        no_2s.hesse(ncall=20000)
        no_3s = make_minuit(
            start_values=nominal_start,
            fixed_ups_params={"ups_frac_3s_in_excited": 0.0},
        )
        no_3s.migrad(ncall=20000)
        no_3s.hesse(ncall=20000)
        no_excited = make_minuit(
            start_values=nominal_start,
            fixed_ups_params={"ups_frac_excited": 0.0, "ups_frac_3s_in_excited": 0.0},
        )
        no_excited.migrad(ncall=20000)
        no_excited.hesse(ncall=20000)
        ups_null_results = {
            "Ups_2S": {
                "null_hypothesis": "no 2S model",
                "nll": float(no_2s.fval),
                "fit_valid": bool(no_2s.fmin.is_valid),
                "minuit": no_2s,
            },
            "Ups_3S": {
                "null_hypothesis": "no 3S model",
                "nll": float(no_3s.fval),
                "fit_valid": bool(no_3s.fmin.is_valid),
                "minuit": no_3s,
            },
            "Ups_excited": {
                "null_hypothesis": "no 2S or 3S model",
                "nll": float(no_excited.fval),
                "fit_valid": bool(no_excited.fmin.is_valid),
                "minuit": no_excited,
            },
        }
        ups_peak_significance_table = build_ups_peak_significance_table(
            full_nll=float(m.fval),
            full_fit_valid=bool(m.fmin.is_valid),
            total_signal_yield=float(m.values["N_sss"]),
            frac_excited=float(m.values["ups_frac_excited"]),
            frac_3s_in_excited=float(m.values["ups_frac_3s_in_excited"]),
            null_results=ups_null_results,
        )

    yield_names = ["N_sss", "N_ssb", "N_sbs", "N_bss", "N_sbb", "N_bsb", "N_bbs", "N_bbb"]
    used_parameters = set(yield_names) | {"j1_mean", "j1_sigma_cb", "j1_alpha", "j1_n", "j1_sigma_g", "j1_frac", "j1_slope"}
    if "Jpsi_2_mass" in fit_branches:
        used_parameters |= {"j2_mean", "j2_sigma_cb", "j2_alpha", "j2_n", "j2_sigma_g", "j2_frac", "j2_slope"}
    if "Phi_mass" in fit_branches:
        used_parameters |= {"phi_mean", "phi_sigma", "phi_c1", "phi_c2", "phi_c3", "phi_c4"}
    if "Ups_mass" in fit_branches:
        used_parameters |= {"ups_mean", "ups_sigma", "ups_frac_excited", "ups_frac_3s_in_excited"}
        used_parameters |= {f"ups_c{idx}" for idx in range(1, ups_background_order + 1)}

    fit_summary_rows = [
        {"quantity": "analysis_mode", "value": analysis_mode},
        {"quantity": "is_valid", "value": m.fmin.is_valid},
        {"quantity": "has_accurate_covar", "value": m.fmin.has_accurate_covar},
        {"quantity": "edm", "value": m.fmin.edm},
        {"quantity": "fval", "value": m.fval},
        {"quantity": "N_total_dataset", "value": len(fit_df)},
        {"quantity": "jpsi1_locked_params", "value": ",".join(locked_param_names(jpsi1_pdf_config)) or "(none)"},
        {"quantity": "jpsi2_locked_params", "value": ",".join(locked_param_names(jpsi2_pdf_config)) if "Jpsi_2_mass" in fit_branches else "(n/a)"},
        {"quantity": "N_sss (3D signal yield)", "value": m.values["N_sss"]},
        {"quantity": "N_sss_err", "value": m.errors["N_sss"]},
        {"quantity": "null_fit_valid", "value": m_null.fmin.is_valid},
        {"quantity": "null_fval", "value": m_null.fval},
        {"quantity": "delta_nll", "value": significance["delta_nll"]},
        {"quantity": "q0", "value": significance["q0"]},
        {"quantity": "p0", "value": significance["p0"]},
        {"quantity": "N_sss_significance_sigma", "value": significance["N_sss_significance_sigma"]},
    ]
    if "Phi_mass" in fit_branches:
        fit_summary_rows.append({"quantity": "phi_background_degree", "value": 4})
    if "Ups_mass" in fit_branches:
        fit_summary_rows.append({"quantity": "ups_background_order", "value": ups_background_order})
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

    return {
        "minuit": m,
        "fit_summary": pd.DataFrame(fit_summary_rows),
        "parameter_table": minuit_parameter_table(m, used_parameters=used_parameters),
        "yield_table": pd.DataFrame([{"yield": name, "value": m.values[name], "error": m.errors[name]} for name in yield_names]),
        "yield_names": yield_names,
        "jpsi1_pdf_config": jpsi1_pdf_config,
        "jpsi2_pdf_config": jpsi2_pdf_config,
        "null_minuit": m_null,
        "ups_peak_significance_table": ups_peak_significance_table,
        "ups_null_results": ups_null_results,
    }
