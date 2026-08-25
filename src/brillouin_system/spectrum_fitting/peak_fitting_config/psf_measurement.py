"""The MEASURED camera PSF kernel — a frozen measurement record.

USER RULE (2026-08-24): the PSF belongs to the PEAKS — it exists because
of how the spectral lines land on the sensor and lives entirely in the
peak-fitting domain. ccd_characteristics is the pure readout chain (gain,
read noise, dark level) and holds nothing PSF-related.

The WORKING kernel the fitter uses is the [global] fitting config
(SlineFromFrameConfig.psf_* — one fitting config, no nested sub-config;
user decision 2026-08-20). This record keeps the measured values +
provenance so experimentation over there can never lose the measurement:
the config GUI shows these in brackets and never writes them. A
re-measurement (measure_psf_kernel.py, next to this file) updates BOTH
this record and the working values.

  psf_sigma_px     Gaussian charge-diffusion blur.
  psf_tau_*_px     one-sided exponential readout smear, per peak, toward
                   higher pixel numbers (the charge-transfer direction).

Measured 2026-07 on the fine EOM sweeps: 0.25 / 0.40 / 0.20 px, stable
across 6 calibrations over 7 weeks (re-confirmed alignment-stable across
3 months in the 2026-08 background-model work). The outer taus (n_peaks=4
fit only) were measured 2026-08-20 from the outer calibration lines of
four 4-peak-ROI sessions — PROVISIONAL (per-frame sigma/tau/gamma are
degenerate; sweep medians only): fine for positions and intensities, do
not hang width claims on outer-peak lineshapes. Re-measure after any
camera / ROI / readout change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PsfMeasurement:
    psf_sigma_px: float = 0.25
    psf_tau_left_px: float = 0.40
    psf_tau_right_px: float = 0.20
    psf_tau_outer_left_px: float = 0.50
    psf_tau_outer_right_px: float = 0.0
    psf_measured: str = "2026-07"
    psf_method: str = (
        "Fine EOM sweeps: grid over (sigma, tau_l, tau_r) minimizing the "
        "per-pixel residual sine of the calibration tracks (from ~7 MHz to "
        "~2 MHz rms); stable across 6 calibrations over 7 weeks. See "
        "measure_psf_kernel.py."
    )


PSF_MEASURED = PsfMeasurement()
