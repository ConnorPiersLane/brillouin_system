"""ONE function that turns an AxialScan into the numbers a figure needs.

summarize_axial_scan(scan) -> AxialScanSummary: everything the paper
figures (2, 3, 4) read per scan, computed exclusively with the production
classes — fit_axial_scan for the fits/shifts/widths/photons,
theoretical_precision for the Thompson bound (one bound per scan at the
scan-MEAN fit parameters, the settled recipe), na_mean_shift_ratio for the
post-hoc NA cone factor. No parallel analysis chain: what this returns IS
what the production pipeline says under the CURRENT configs (model preset,
beta, reference model), so set the configs first when a figure must
reproduce a specific recipe.

Conventions baked in (the hard-won rules):
* measured per-frame scatter = consecutive-difference sd / sqrt(2)
  (drift-immune) — the number the Thompson bound is compared against;
  the plain sd is reported alongside.
* ONE Thompson bound per scan, at the scan-mean amplitude/width/position/
  background — never an average of per-frame bounds.
* frames are fitted RAW; the dark/bias level enters only the bound
  (scan's own dark stack, else the ccd_characteristics reference).
* error bars on a scan's MEAN belong to scan-to-scan scatter, not the
  within-scan s.e.m. — this summary gives per-scan numbers; combining
  scans is the figure's job.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from brillouin_system.my_dataclasses.human_interface_measurements import (
    AnalyzedSpectrum,
    AxialScan,
    calibration_for_scan,
    fit_axial_scan,
    fitter_for_scan,
)
from brillouin_system.spectrum_fitting.na_lineshape import na_mean_shift_ratio
from brillouin_system.spectrum_fitting.noise_analysis import (
    PixelCountsAndPhotons,
    theoretical_precision,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter


@dataclass
class AxialScanSummary:
    """One scan reduced to one row. GHz for values, MHz for scatter/bounds."""
    id: str
    i: int
    n_frames: int
    n_success: int
    exposure_s: float
    emccd_gain: float
    preamp_gain: float

    # Shift observables: mean over frames, plain sd, diff-sd, Thompson bound.
    shift_left_ghz: float | None = None
    shift_right_ghz: float | None = None
    shift_distance_ghz: float | None = None
    sd_left_mhz: float | None = None
    sd_right_mhz: float | None = None
    sd_distance_mhz: float | None = None
    diff_sd_left_mhz: float | None = None
    diff_sd_right_mhz: float | None = None
    diff_sd_distance_mhz: float | None = None
    thompson_left_mhz: float | None = None
    thompson_right_mhz: float | None = None
    thompson_distance_mhz: float | None = None

    # Widths, mean over frames (GHz, HWHM as everywhere in the chain).
    hwhm_left_ghz: float | None = None
    hwhm_right_ghz: float | None = None
    instrument_hwhm_left_ghz: float | None = None
    instrument_hwhm_right_ghz: float | None = None
    linewidth_left_ghz: float | None = None
    linewidth_right_ghz: float | None = None

    # Photons per frame, mean over frames.
    photons_left: float | None = None
    photons_right: float | None = None
    photons_total: float | None = None

    # Post-hoc NA cone factor under the current sample config (divide the
    # measured shift by it); None when na_weighting is "none"/unconfigured.
    na_shift_ratio: float | None = None


def _series(analyzed: list[AnalyzedSpectrum], get) -> np.ndarray:
    vals = np.array([get(a) if get(a) is not None else np.nan
                     for a in analyzed], dtype=float)
    return vals[np.isfinite(vals)]


def _stats(vals: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """(mean [same unit], plain sd [MHz], diff-sd [MHz]) of a GHz series."""
    if vals.size == 0:
        return None, None, None
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) * 1000.0 if vals.size >= 2 else None
    dsd = (float(np.std(np.diff(vals), ddof=1) / np.sqrt(2.0)) * 1000.0
           if vals.size >= 3 else None)
    return mean, sd, dsd


def _mean_or_none(vals: np.ndarray) -> float | None:
    return float(np.mean(vals)) if vals.size else None


def summarize_axial_scan(
    scan: AxialScan,
    fitter: SpectrumFitter | None = None,
) -> AxialScanSummary:
    """The one entry point: fit the scan with the production chain and
    reduce it to a figure-ready row."""
    fitter = fitter if fitter is not None else fitter_for_scan(scan)
    calc = calibration_for_scan(scan, fitter)
    analyzed = fit_axial_scan(scan, fitter=fitter, calibration_calculator=calc)

    info = scan.system_state.andor_camera_info
    fits = [a.fitted_spectrum for a in analyzed if a.fitted_spectrum.is_success]

    out = AxialScanSummary(
        id=scan.id, i=scan.i,
        n_frames=len(scan.measurements), n_success=len(fits),
        exposure_s=float(info.exposure), emccd_gain=float(info.gain),
        preamp_gain=float(info.preamp_gain),
    )

    # --- shifts ---
    for name, get in (
            ("left", lambda a: a.analyzed_shifts.freq_shift_left_peak_ghz),
            ("right", lambda a: a.analyzed_shifts.freq_shift_right_peak_ghz),
            ("distance", lambda a: a.analyzed_shifts.freq_shift_peak_distance_ghz)):
        mean, sd, dsd = _stats(_series(analyzed, get))
        setattr(out, f"shift_{name}_ghz", mean)
        setattr(out, f"sd_{name}_mhz", sd)
        setattr(out, f"diff_sd_{name}_mhz", dsd)

    # --- widths + photons (means over successful frames) ---
    for name, get in (
            ("hwhm_left_ghz", lambda a: a.analyzed_shifts.hwhm_left_peak_ghz),
            ("hwhm_right_ghz", lambda a: a.analyzed_shifts.hwhm_right_peak_ghz),
            ("instrument_hwhm_left_ghz",
             lambda a: a.analyzed_shifts.instrument_hwhm_left_peak_ghz),
            ("instrument_hwhm_right_ghz",
             lambda a: a.analyzed_shifts.instrument_hwhm_right_peak_ghz),
            ("linewidth_left_ghz",
             lambda a: a.analyzed_shifts.linewidth_left_peak_ghz),
            ("linewidth_right_ghz",
             lambda a: a.analyzed_shifts.linewidth_right_peak_ghz),
            ("photons_left", lambda a: a.photons.left_peak_photons),
            ("photons_right", lambda a: a.photons.right_peak_photons),
            ("photons_total", lambda a: a.photons.total_photons)):
        setattr(out, name, _mean_or_none(_series(analyzed, get)))

    # --- ONE Thompson bound at the scan-mean fit parameters ---
    if fits:
        def mean_of(attr):
            vals = np.array([getattr(f, attr) for f in fits
                             if getattr(f, attr) is not None], dtype=float)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else None

        mean_fs = replace(
            fits[0],
            left_peak_center_px=mean_of("left_peak_center_px"),
            left_peak_width_px=mean_of("left_peak_width_px"),
            left_peak_amplitude=mean_of("left_peak_amplitude"),
            right_peak_center_px=mean_of("right_peak_center_px"),
            right_peak_width_px=mean_of("right_peak_width_px"),
            right_peak_amplitude=mean_of("right_peak_amplitude"),
            inter_peak_distance=mean_of("inter_peak_distance"),
            left_peak_bg_counts=mean_of("left_peak_bg_counts"),
            right_peak_bg_counts=mean_of("right_peak_bg_counts"),
        )
        photons = PixelCountsAndPhotons.from_fit(
            fs=mean_fs, preamp_gain=info.preamp_gain, emccd_gain=info.gain)
        # mean_fs carries the row band (sline_rows, inherited from the
        # fits); the camera numbers (read noise, dark level) come from
        # ccd_characteristics inside theoretical_precision.
        theo = theoretical_precision(
            fs=mean_fs, photons=photons, calibration_calculator=calc,
            preamp_gain=info.preamp_gain, emccd_gain=info.gain)
        out.thompson_left_mhz = theo.left_peak_total_mhz
        out.thompson_right_mhz = theo.right_peak_total_mhz
        out.thompson_distance_mhz = theo.distance_total_mhz

    # --- NA cone factor (post-hoc; never inside the fit) ---
    try:
        out.na_shift_ratio = float(na_mean_shift_ratio(fitter.sample_config))
    except Exception:
        out.na_shift_ratio = None

    return out
