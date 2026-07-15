from dataclasses import dataclass


@dataclass
class ElasticAnchors:
    """Pixel positions of the two Rayleigh-order elastic lines (zero-shift),
    extrapolated from the calibration polynomials (CalibrationCalculator
    .elastic_anchors()). Needed by fitting models that anchor each Brillouin
    peak at its own elastic order (na_lorentzian*).

    Lives in spectrum_fitting (not calibration) so SpectrumFitter can type
    against it without importing calibration, which imports SpectrumFitter.
    """
    rayleigh_left_px: float
    rayleigh_right_px: float
