from __future__ import annotations

import awkward as ak
import numpy as np

from .config import OfflineSelectionConfig


def eta_from_pxyz(px, py, pz):
    p = np.sqrt(px * px + py * py + pz * pz)
    safe = p > np.abs(pz)
    ratio = ak.where(safe, (p + pz) / (p - pz), 1.0)
    eta = 0.5 * np.log(ratio)
    return ak.where(safe, eta, ak.where(pz >= 0.0, 1.0e6, -1.0e6))


def rapidity_from_pxyzm(px, py, pz, mass):
    e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    good = (e + pz > 0.0) & (e - pz > 0.0)
    ratio = ak.where(good, (e + pz) / (e - pz), 1.0)
    y = 0.5 * np.log(ratio)
    return ak.where(good, y, np.nan)


def in_window(values, window: tuple[float, float]):
    low, high = window
    return np.isfinite(values) & (values >= low) & (values <= high)


def muon_pass_mask(pt, eta, soft, cfg: OfflineSelectionConfig):
    abs_eta = np.abs(eta)
    return (
        (soft == 1)
        & (
            ((abs_eta < 1.2) & (pt > cfg.mu_pt_barrel_min))
            | (
                (abs_eta >= 1.2)
                & (abs_eta < cfg.mu_abs_eta_max)
                & (pt > cfg.mu_pt_endcap_min)
            )
        )
    )


def flatten_candidate_field(array):
    return ak.to_numpy(ak.flatten(array, axis=1))
