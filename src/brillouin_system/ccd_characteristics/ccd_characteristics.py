"""Measured characteristics of THE camera — one home for every number.

Everything in this package describes the physical camera (Andor iXon Ultra
897, DU897_BV serial 9303), NOT fitting configuration: each value was
measured on the instrument, is frozen between measurements, and must be
re-measured after a hardware / readout-mode / ROI change. The rule
(2026-08-20): any parameter that has to be FITTED or OBTAINED from a
dedicated measurement is written in ccd_characteristics.toml, and the
script that obtains it lives in measurement_scripts/ next to it — so the
numbers and the recipe that produced them can never drift apart.

Layout:
    ccd_characteristics.toml   the numbers + provenance (dates, methods)
    measurement_scripts/       one runnable script per number:
        measure_gain_photon_transfer.py   sensitivity g [e-/count] (+ eps)
        measure_read_noise_and_dark.py    read noise + dark median [counts]
        measure_psf_kernel.py             PSF sigma / tau [px]

Consumers read the ThreadSafeConfig instances below (`ccd_config`,
`psf_config`) — noise_analysis for g / read noise / dark level, the
spectrum fitter for the PSF kernel. Scan-local measurements still take
precedence where they exist (a scan's own dark stack beats the TOML dark
median / read noise — same philosophy everywhere: per-scan at live
settings wins, the TOML is the documented reference and fallback).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from pathlib import Path

import tomli
import tomli_w

from brillouin_system.helpers.thread_safe_config import ThreadSafeConfig

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
    # electronic dark/bias pedestal. It carries NO shot noise (offset, not
    # light): the Thompson bound subtracts it from the fitted background
    # before the pedestal Poisson term (pedestal_bias_counts).
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


@dataclass
class PsfConstants:
    """[psf] section: the frozen camera PSF (the 'lorentzian_x_psf' model).

    GLOBAL — one camera, one kernel, shared by the sample and reference fits
    (different kernels would define different peak-centre conventions, which
    is exactly the model-mixing artifact the fitter guards against). These
    ENTER the fit but are never fitted per frame: they are properties of the
    camera, not of the data.
      psf_sigma_px     Gaussian charge-diffusion blur.
      psf_tau_*_px     one-sided exponential readout smear, per peak, toward
                       higher pixel numbers (the charge-transfer direction).
    Measured 2026-07 on the fine EOM sweeps (see
    measurement_scripts/measure_psf_kernel.py): 0.25 / 0.40 / 0.20 px,
    stable across 6 calibrations over 7 weeks. Re-measure after any
    camera/ROI change; the model refuses to run while all three are 0.
    """
    psf_sigma_px: float = 0.0
    psf_tau_left_px: float = 0.0
    psf_tau_right_px: float = 0.0
    # MEASURED reference values of the kernel, kept separately so they can
    # never be lost: the working psf_* fields above may be tuned in the
    # config GUI and saved, but the GUI never writes these — they change
    # only on a re-measurement (measurement_scripts/measure_psf_kernel.py).
    # The GUI shows them in brackets next to each working field.
    psf_sigma_px_measured: float = 0.25
    psf_tau_left_px_measured: float = 0.40
    psf_tau_right_px_measured: float = 0.20
    psf_measured: str = ""
    psf_method: str = ""
    # Outer-order tails (the opt-in n_peaks=4 fit only). The tail is a
    # POSITION property falling ~linearly toward the readout side — measured
    # 2026-08-20 from the outer calibration lines of four 4-peak-ROI sessions
    # (256 lines per position, Data/Figure3/fit_outer_kernel.py): tau ≈ +0.50
    # (outer left, px ~35) / +0.40 / +0.20 / +0.00 (outer right, px ~148).
    # PROVISIONAL (per-frame sigma/tau/gamma are degenerate; sweep medians
    # only): fine for positions and intensities, do not hang width claims on
    # outer-peak lineshapes.
    psf_tau_outer_left_px: float = 0.50
    psf_tau_outer_right_px: float = 0.0


def _load_section(path: Path, section: str, cls):
    with path.open("rb") as f:
        data = tomli.load(f)
    raw = data.get(section, {})
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in names})


def load_ccd_characteristics(path: Path = CCD_TOML_PATH) -> CcdCharacteristics:
    return _load_section(path, "ccd", CcdCharacteristics)


def load_psf_constants(path: Path = CCD_TOML_PATH) -> PsfConstants:
    return _load_section(path, "psf", PsfConstants)


def save_ccd_section(section: str, config: ThreadSafeConfig,
                     path: Path = CCD_TOML_PATH):
    """Write one section ('ccd' or 'psf') back to the TOML."""
    with path.open("rb") as f:
        data = tomli.load(f)
    data[section] = asdict(config.get_raw())
    with path.open("wb") as f:
        tomli_w.dump(data, f)


# Global instances
ccd_config = ThreadSafeConfig(load_ccd_characteristics(CCD_TOML_PATH))
psf_config = ThreadSafeConfig(load_psf_constants(CCD_TOML_PATH))
