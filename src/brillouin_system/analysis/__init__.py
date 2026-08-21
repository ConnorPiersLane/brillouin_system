"""The analysis layer: everything that consumes a FIT plus a CALIBRATION.

Sits strictly above spectrum_fitting (pixel domain) and calibration
(px -> GHz), so the package layering is one-directional:

    my_dataclasses -> spectrum_fitting -> calibration -> analysis

Modules:
    pixel_counts_and_photons    per-peak counts/photons behind a fit
    thompson_shot_noise_limit   the precision bound (generic + production)
    monte_carlo_noise_simulation  synthetic frames with exactly-known noise
    analyzed_spectrum           one frame's full analysis result
    fit_axial_scan              a scan through the whole chain
    scan_summary                one scan reduced to one figure-ready row
"""
from brillouin_system.analysis.analyzed_spectrum import AnalyzedSpectrum
from brillouin_system.analysis.fit_axial_scan import analyze_frame, fit_axial_scan
from brillouin_system.analysis.monte_carlo_noise_simulation import (
    MonteCarloFrames,
)
from brillouin_system.analysis.pixel_counts_and_photons import (
    PixelCountsAndPhotons,
    electrons_per_count,
    count_to_electrons,
)
from brillouin_system.analysis.scan_summary import (
    AxialScanSummary,
    summarize_axial_scan,
)
from brillouin_system.analysis.thompson_shot_noise_limit import (
    PeakPrecision,
    peak_precision,
    distance_precision,
    TheoreticalPeakStdError,
    theoretical_precision,
    LORENTZIAN_PHOTON_FACTOR,
)

__all__ = [
    "AnalyzedSpectrum",
    "analyze_frame",
    "fit_axial_scan",
    "MonteCarloFrames",
    "PixelCountsAndPhotons",
    "electrons_per_count",
    "count_to_electrons",
    "AxialScanSummary",
    "summarize_axial_scan",
    "PeakPrecision",
    "peak_precision",
    "distance_precision",
    "TheoreticalPeakStdError",
    "theoretical_precision",
    "LORENTZIAN_PHOTON_FACTOR",
]
