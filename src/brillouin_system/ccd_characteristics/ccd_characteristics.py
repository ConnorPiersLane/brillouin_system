"""Measured characteristics of THE camera — the pure readout chain.

Everything in this package describes the physical camera's READOUT CHAIN
(Andor iXon Ultra 897, DU897_BV serial 9303): gain, read noise, dark
level. NOT fitting configuration, and NOT the PSF — the PSF is a property
of how the spectral lines land on the sensor and belongs to the peaks
(user rule 2026-08-24: record + measurement script live in
spectrum_fitting/peak_fitting_config/psf_measurement.py). Each value here
was measured on the instrument, is frozen between measurements, and must
be re-measured after a hardware / readout-mode change. The rule
(2026-08-20): any parameter that has to be FITTED or OBTAINED from a
dedicated measurement is written in ccd_characteristics.toml, and the
script that obtains it lives in measurement_scripts/ next to it — so the
numbers and the recipe that produced them can never drift apart.

Layout:
    ccd_characteristics.toml   the numbers + provenance (dates, methods)
    measurement_scripts/       one runnable script per number:
        measure_gain_photon_transfer.py   sensitivity g [e-/count] (+ eps)
        measure_read_noise_and_dark.py    read noise + dark median [counts]

Consumers read `ccd_config` below (noise_analysis: g / read noise / dark
level). Scan-local measurements still take precedence where they exist (a
scan's own dark stack beats the TOML dark median / read noise — same
philosophy everywhere: per-scan at live settings wins, the TOML is the
documented reference and fallback).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path

import tomli
import tomli_w

from brillouin_system.helpers.thread_safe_config import LazyThreadSafeConfig, ThreadSafeConfig

CCD_TOML_PATH = Path(__file__).parent / "ccd_characteristics.toml"


@dataclass
class CcdCharacteristics:
    """[ccd] section: gain, noise and dark level of the readout chain.

    ALL values are per (output amplifier, readout speed, preamp) mode —
    they do NOT generalise to another mode; `valid_mode` documents the one
    they were measured in. The *_measured / *_method fields are provenance,
    loaded and saved with the numbers on purpose (TOML comments do not
    survive programmatic writes; provenance-as-data does).
    """
    # Photoelectrons per digitised count (ADU) at preamp 1x, Conventional
    # amplifier. The preamp MULTIPLIER divides this (see
    # noise_analysis.electrons_per_count).
    sensitivity_e_per_count_preamp_1x: float = 3.89
    # EM amplifier sensitivity: NEVER measured (different output amplifier —
    # the Conventional value does not transfer). 0.0 = unset; the EM path in
    # noise_analysis raises rather than guess.
    sensitivity_e_per_count_em_preamp_1x: float = 0.0
    # Per-pixel read noise rms [counts]; lives in the readout amplifier, so
    # it is exposure-independent but readout-mode-dependent.
    read_noise_counts: float = 1.10
    # Median pixel count of a closed-shutter frame [counts/px] — the
    # electronic dark level. It carries NO shot noise (offset, not
    # light): the Thompson bound subtracts it from the fitted background
    # before the background-light Poisson term (dark_counts).
    dark_median_counts: float = 200.2
    # EM register excess noise factor F^2 on the variance (stochastic
    # multiplication) — theory, not measured; unused until EM sensitivity is.
    em_excess_noise_factor: float = 2.0

    # --- provenance ---
    camera_model: str = ""
    camera_serial: str = ""
    valid_mode: str = ""
    sensitivity_measured: str = ""
    sensitivity_method: str = ""
    read_noise_measured: str = ""
    read_noise_method: str = ""
    dark_median_measured: str = ""
    dark_median_method: str = ""


def _load_section(path: Path, section: str, cls):
    with path.open("rb") as f:
        data = tomli.load(f)
    raw = data.get(section, {})
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def load_ccd_characteristics(path: Path = CCD_TOML_PATH) -> CcdCharacteristics:
    return _load_section(path, "ccd", CcdCharacteristics)


def save_ccd_section(section: str, config: ThreadSafeConfig,
                     path: Path = CCD_TOML_PATH):
    """Write one section ('ccd') back to the TOML."""
    with path.open("rb") as f:
        data = tomli.load(f)
    data[section] = asdict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Global instance
ccd_config = LazyThreadSafeConfig(lambda: load_ccd_characteristics(CCD_TOML_PATH))
