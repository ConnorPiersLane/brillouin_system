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

  psf_box_*_px     ROW-TILT boxcar: each line is tilted against the CCD
                   columns, so the 13-row sum smears it. The four tilts
                   are ONE physical constant — a ~27 MHz/row frequency
                   shear (a VIPA property, NOT camera rotation) divided
                   by each track's local dispersion — measured stable to
                   0.1-3 percent across all four sweeps incl. a
                   realignment (Data/2026-9-2/tilt_all_runs.py). Only
                   the OUTER orders carry the literal top-hat: the
                   row-intensity profile is bell-shaped (sd ~2 rows), so
                   the INNER peaks' tilt smear is Gauss+tail-shaped and
                   already lives in their sigma/tau (forcing a top-hat
                   there ruins their sines 0.09/0.25 -> 4.8/2.9 MHz).

Inner constants user-adopted 2026-09-03 from the four-peak 4001-point
fine-sweep determination of 2026-09-02 (four runs, agreement +-0.02 px
incl. a realignment; residual sines 0.07/0.09 MHz — records in
Data/2026-9-2/determine_fourpeak_summary.txt and per-run summaries).
OUTER constants redetermined and user-adopted 2026-09-04 under the
measured boxcars (determine_allbox.py, four runs: outer_right tau
agrees to +-0.001 px, sigma +-0.005; records in
determine_allbox*_summary.txt). At the adopted set ALL SIXTEEN
peak x run sine amplitudes are <1 MHz (verify_final_constants.py) and
the four orders' instrument widths group by SIDE in frequency space
(anti-Stokes 219/236, Stokes 283/298 MHz FWHM — cross-order
consistency, the calibration-only sigma selector). Held-out width
validation: water 29 C all four peaks 0.93-1.03 of Holmes (outer_left
0.81-0.84 at 40-45 C = the known narrow-line corner); glycerol
10-40 wt outer/inner 0.95-1.04 (Data/2026-9-3/
outer_final_water_closure.csv, Data/2026-8-28/wg_fourpeak_final.csv).

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
    # PHYSICAL outer constants (2026-09-04 determination under the
    # measured row-tilt boxcars, four-run agreement, outer_right tau
    # +-0.001 px; PRODUCTION again since 2026-09-05 together with the
    # outer boxes). The boxes are geometry: ONE ~27 MHz/row frequency
    # shear (a VIPA property, stable across sweeps and a realignment)
    # divided by each track's local dispersion, x 13 rows. Inner boxes
    # are 0 BY MEASUREMENT (bell-shaped row profile: their smear is
    # Gauss+tail-shaped and already inside sigma/tau).
    psf_sigma_outer_left_px: float = 0.14
    psf_sigma_outer_right_px: float = 0.15
    psf_tau_outer_left_px: float = 0.13
    psf_tau_outer_right_px: float = 0.08
    psf_box_outer_left_px: float = 1.95
    psf_box_outer_right_px: float = 0.85
    psf_sat_ratio_outer_right: float = 0.037
    psf_sat_delta_outer_right_px: float = -1.23
    psf_measured: str = (
        "2026-09-04 (boxcar-decomposed outer kernels, 9-2 sweeps; "
        "inner 2026-09-03); production 2026-09-05")
    psf_method: str = (
        "Four 4001-point four-peak fine sweeps (Data/2026-9-2): row tilts "
        "measured per peak on all four runs (tilt_all_runs.py); outer "
        "boxes frozen at tilt x 13 rows; (sigma, tau) per peak from "
        "adaptive scans minimizing each peak's folded once-per-pixel "
        "sine (four-run agreement incl. a realignment). outer_right "
        "additionally carries an intrinsic near-core satellite (ratio "
        "0.037 at -1.23 px, blind-validated). Applied in production to "
        "lorentzian_x_psf AND the four-peak dho_x_psf (2026-09-05): the "
        "DHO removes the lineshape-lean shift systematic, the boxes fix "
        "the outer width closure, the satellite the outer_right centre."
    )


PSF_MEASURED = PsfMeasurement()
