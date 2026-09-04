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

ALL EIGHT constants user-adopted 2026-09-03 from the four-peak
4001-point fine-sweep determination of 2026-09-02 (four runs,
calibration_4001_1..4.h5, adaptive per-peak scans minimizing each
peak's folded once-per-pixel sine; agreement +-0.02 px across the four
runs INCLUDING a realignment before run 4; second coordinate-descent
pass puts outer_left / left / right at 0.06 / 0.07 / 0.09 MHz residual
sine — records in Data/2026-9-2/determine_fourpeak_summary.txt,
phase3_outer_tau_summary.txt, constants per run in
determine_calibration_4001_*_summary.txt). The previous 2026-08-31
inner values (0.25/0.28, 0.40/0.18, May epoch) differ by <=0.01 px.

The OUTER_RIGHT order carries an intrinsic NEAR-CORE SATELLITE — a
scaled displaced copy of the main line (same gamma, same kernel) with
ratio psf_sat_ratio_outer_right at psf_sat_delta_outer_right_px.
Without it that peak's fitted position wobbles once per pixel by
~3.2 MHz, immune to any (sigma, tau); with it 0.32-0.41 MHz, validated
BLIND on the three runs it was not tuned on (Data/2026-9-2/
ghost_model_opt.txt, validate_companion_run*.txt). Nine alternative
mechanisms were eliminated by measurement (sampling/estimator bias via
synthetic data, window placement, joint-fit crosstalk, row tilt, column
gains, stray contamination, track curvature, distant ghost, kernel
shape). The other three orders need no satellite (their residual floors
leave nothing to determine).
Re-measure after any camera / ROI / readout change.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PsfMeasurement:
    psf_sigma_left_px: float = 0.26
    psf_sigma_right_px: float = 0.27
    psf_tau_left_px: float = 0.39
    psf_tau_right_px: float = 0.17
    psf_sigma_outer_left_px: float = 0.39
    psf_sigma_outer_right_px: float = 0.36
    psf_tau_outer_left_px: float = 0.95
    psf_tau_outer_right_px: float = 0.0
    psf_sat_ratio_outer_right: float = 0.037
    psf_sat_delta_outer_right_px: float = -1.23
    psf_measured: str = "2026-09-03 (four-peak 4001-pt determination, 9-2)"
    psf_method: str = (
        "Four 4001-point four-peak fine sweeps (Data/2026-9-2): adaptive "
        "per-peak scans of each (sigma, tau) minimizing that peak's folded "
        "once-per-pixel sine; four-run agreement +-0.02 px incl. a "
        "realignment; residual sines 0.06-0.09 MHz (outer_left/left/right). "
        "outer_right additionally carries an intrinsic near-core satellite "
        "(scaled displaced copy, ratio 0.037 at -1.23 px), optimized on "
        "run 1 and validated blind on runs 2-4 (0.32-0.41 MHz). See "
        "determine_fourpeak_kernel.py and ghost_model_opt.py with the data."
    )


PSF_MEASURED = PsfMeasurement()
