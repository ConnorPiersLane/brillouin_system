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

  psf_sigma_*_px   Gaussian charge-diffusion blur, per peak (a position
                   property on the sensor).
  psf_tau_*_px     one-sided exponential readout smear, per peak, toward
                   higher pixel numbers (the charge-transfer direction;
                   the direction was TESTED, not assumed — a mirrored
                   tail is worse than no tail on both peaks,
                   Data/Figure2/tau_direction_scan.txt, 2026-08-31).

FINAL values user-adopted 2026-08-31: the two-decimal means of the
three-sweep refit (Data/SectionS3/constants_summary.txt over
cal2_5-18 / cal1_5-22 / cal2_5-22: tau*_AS 0.399 +- 0.007, tau*_S
0.185 +- 0.002, sigma*_AS 0.252 +- 0.005, sigma*_S 0.285 +- 0.003 ->
adopted 0.40 / 0.18 / 0.25 / 0.28). Consistent with the original
2026-07 shared-kernel measurement (0.25 / 0.40 / 0.20, stable across
6 calibrations over 7 weeks, re-confirmed alignment-stable across
3 months in the 2026-08 background-model work). Validated on Figure 2:
the Stokes folded residual sine drops ~0.9 -> ~0.4 MHz, anti-Stokes
panels unchanged. The
outer taus (n_peaks=4 fit only) were measured 2026-08-20 from the outer
calibration lines of four 4-peak-ROI sessions — PROVISIONAL (per-frame
sigma/tau/gamma are degenerate; sweep medians only): fine for positions
and intensities, do not hang width claims on outer-peak lineshapes.
Re-measure after any camera / ROI / readout change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PsfMeasurement:
    psf_sigma_left_px: float = 0.25
    psf_sigma_right_px: float = 0.28
    psf_tau_left_px: float = 0.40
    psf_tau_right_px: float = 0.18
    psf_tau_outer_left_px: float = 0.50
    psf_tau_outer_right_px: float = 0.0
    psf_measured: str = "2026-08-31 (three-sweep per-peak refit)"
    psf_method: str = (
        "Fine EOM sweeps: scans of the kernel constants minimizing the "
        "per-pixel residual sine of the calibration tracks (from ~7 MHz to "
        "~2 MHz rms). Adopted = two-decimal means of the 2026-08-31 "
        "three-sweep refit (Data/SectionS3/constants_summary.txt, sd "
        "0.007/0.002/0.005/0.003 px); consistent with the 2026-07 "
        "shared-sigma measurement, stable across 6 calibrations over "
        "7 weeks. See measure_psf_kernel.py."
    )


PSF_MEASURED = PsfMeasurement()
