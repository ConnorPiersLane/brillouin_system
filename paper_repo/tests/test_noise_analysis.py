"""Tests for brillouin_peak_fitting.noise_analysis.

Run with paper_repo/src on PYTHONPATH:  pytest tests/test_noise_analysis.py
"""
import math

import numpy as np
from scipy.optimize import curve_fit

from brillouin_peak_fitting.noise_analysis import (
    MonteCarloFrames,
    peak_photons,
    peak_precision,
    peaks_precision,
    distance_precision,
)

GAIN = 3.89          # e-/count
READ = 1.10          # counts rms


def test_generator_matches_its_own_noise_model():
    """Per-pixel scatter of the frames must reproduce expected_std()."""
    mean = np.full((4, 50), 1000.0)
    mc = MonteCarloFrames(mean_frame=mean, gain_e_per_count=GAIN,
                          read_noise_counts=READ, n_images=3000, seed=1)
    stack = mc.stack()
    measured = stack.std(axis=0, ddof=1).mean()
    expected = mc.expected_std().mean()   # sqrt(1000/3.89 + 1.1^2) ~ 16.07
    assert abs(measured / expected - 1.0) < 0.02
    assert abs(stack.mean() - 1000.0) < 0.5


def test_thompson_terms_are_the_documented_formulas():
    s, n, b, a = 3.0, 20000.0, 4.3, 1.0
    p = peak_precision(width=s, n_photons=n, bg_rms=b, pixel_size=a)
    assert math.isclose(p.photons, math.sqrt(2.0 * s * s / n))
    assert math.isclose(p.pixelation, math.sqrt(a * a / 12.0 / n))
    assert math.isclose(
        p.background,
        math.sqrt(4.0 * math.sqrt(math.pi) * s ** 3 * b ** 2 / (a * n * n)))
    assert math.isclose(
        p.total,
        math.sqrt(p.photons ** 2 + p.pixelation ** 2 + p.background ** 2))


def test_distance_precision():
    assert math.isclose(distance_precision(3.0, 4.0), 5.0)
    # Positive correlation of the centre errors tightens the distance.
    assert distance_precision(3.0, 4.0, correlation=0.5) < 5.0


def test_peak_photons_is_the_exact_lorentzian_area():
    # Area of the pixel-integrated Lorentzian = pi * amp * width, exactly.
    assert math.isclose(peak_photons(500.0, 3.0, GAIN),
                        GAIN * math.pi * 500.0 * 3.0)


def test_peaks_precision_covers_four_peaks():
    widths = [3.0, 2.8, 2.9, 3.1]
    photons = [8000.0, 20000.0, 18000.0, 9000.0]
    results = peaks_precision(widths, photons, bg_rms=4.3)
    assert len(results) == 4
    for r, w, n in zip(results, widths, photons):
        single = peak_precision(width=w, n_photons=n, bg_rms=4.3)
        assert math.isclose(r.total, single.total)
    # Per-peak pixel_size (different local dispersion per track).
    per_track = peaks_precision(widths, photons, bg_rms=4.3,
                                pixel_size=[1.0, 1.1, 0.9, 1.0])
    assert per_track[1].pixelation > results[1].pixelation


def test_monte_carlo_closes_on_the_thompson_bound():
    """A plain least-squares centre fit on MC frames must land in the known
    band above the bound (~1.0-1.4x for unweighted LSQ; the Lorentzian
    factor 2 is already in the bound)."""
    px = np.arange(200, dtype=float)
    amp, cen, wid = 500.0, 100.0, 3.0     # counts, px, HWHM px
    sline = amp / (1.0 + ((px - cen) / wid) ** 2)

    mc = MonteCarloFrames(mean_frame=sline[None, :], gain_e_per_count=GAIN,
                          read_noise_counts=READ, n_images=400, seed=7)

    def lorentzian(x, a, c, w):
        return a / (1.0 + ((x - c) / w) ** 2)

    def fit_center(frame):
        popt, _ = curve_fit(lorentzian, px, frame[0],
                            p0=[amp, cen + 0.5, wid])
        return popt[1]

    centers = np.asarray(mc.run(fit_center))
    measured_std = centers.std(ddof=1)

    n_photons = GAIN * sline.sum()
    bg_rms_e = READ * GAIN                # no pedestal: read noise only
    bound = peak_precision(width=wid, n_photons=n_photons,
                           bg_rms=bg_rms_e, pixel_size=1.0).total

    ratio = measured_std / bound
    print(f"MC/Thompson closure: measured {measured_std:.4f} px, "
          f"bound {bound:.4f} px, ratio {ratio:.3f}")
    assert 0.85 < ratio < 1.6, ratio


if __name__ == "__main__":
    test_generator_matches_its_own_noise_model()
    test_thompson_terms_are_the_documented_formulas()
    test_distance_precision()
    test_monte_carlo_closes_on_the_thompson_bound()
    print("ALL NOISE-ANALYSIS TESTS PASSED")
