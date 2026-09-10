"""The Epsf class (spectrum_fitting/epsf.py): the measured instrument
response of a calibration, built from spectra alone.

Synthetic elastic lines generated with the PARAMETRIC kernel (Lorentzian x
Gauss x one-sided tail x pixel, so the line is asymmetric like the real one)
walk across the detector as a 4-8 GHz sweep. The tests check

  * HELD OUT: an ePSF built on every other frame fits the remaining frames
    with no once-per-pixel wobble (the plain-Lorentzian first guess wobbles
    by 15-20 MHz on the same frames);
  * the template centre sits at a CONSTANT offset from the parametric core
    centre whatever the sub-pixel phase (the convention offset that must
    cancel between axis and kernel);
  * the stored node table round-trips through save/load.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pytest

from brillouin_system.spectrum_fitting.epsf import Epsf
from brillouin_system.spectrum_fitting.measured_kernel import MeasuredKernel
from brillouin_system.spectrum_fitting.measured_kernel import DX

from synthetic_lines import (
    CEN_LEFT, CEN_RIGHT, PX, SLOPE_LEFT, SLOPE_RIGHT, elastic_sline,
    parametric_kernel)


def synthetic_sweep(rng, n_points=81, step_ghz=0.05):
    """(freqs, pxs, slines) of an EOM sweep from 4 GHz: the left line walks
    +0.18 px/step, the right -0.14 px/step, photon + read noise as measured."""
    freqs, pxs, slines = [], [], []
    for k in range(n_points):
        f = 4.0 + step_ghz * k
        cl = CEN_LEFT + (f - 5.07) / SLOPE_LEFT
        cr = CEN_RIGHT + (f - 5.07) / SLOPE_RIGHT
        s = elastic_sline(cl, cr)
        s = s + rng.normal(0.0, np.sqrt(s / 3.89 + 1.37))
        freqs.append(f)
        pxs.append(PX)
        slines.append(s)
    return freqs, pxs, slines


def test_held_out_sine_is_gone():
    freqs, pxs, slines = synthetic_sweep(np.random.default_rng(7))
    r = Epsf.held_out_sine(freqs, pxs, slines, 2)
    assert r.n_train >= 40 and r.n_test >= 40
    for line in range(2):
        seed_amp, _ = r.seed[line]
        test_amp, test_sd = r.test[line]
        train_amp, _ = r.train[line]
        # the plain-Lorentzian first guess wobbles once per pixel ...
        assert seed_amp > 10.0
        # ... the ePSF template, on frames it never saw, does not
        assert test_amp < 0.5, (line, r.test[line])
        assert test_sd < 1.5, (line, r.test[line])
        # and in-sample is no better than held out by more than the noise
        assert abs(train_amp - test_amp) < 0.5


def test_template_centre_offset_is_a_constant_convention():
    freqs, pxs, slines = synthetic_sweep(np.random.default_rng(8))
    e = Epsf(freqs, pxs, slines, 2)
    assert e.names == ("left", "right")
    offs = []
    for d in np.linspace(-0.5, 0.5, 6):            # every sub-pixel phase
        s = elastic_sline(CEN_LEFT + d, CEN_RIGHT - d)      # noise-free
        _, cl, _ = e.fit_line(0, PX, s, CEN_LEFT + d)
        _, cr, _ = e.fit_line(1, PX, s, CEN_RIGHT - d)
        offs.append((cl - (CEN_LEFT + d), cr - (CEN_RIGHT - d)))
    offs = np.array(offs)
    # a constant per line (the tail pulls the template centre off the
    # Lorentzian core by a fixed amount), not a function of the phase
    assert offs.std(axis=0).max() < 0.005
    assert 0.1 < offs[:, 0].mean() < 0.6
    assert 0.05 < offs[:, 1].mean() < 0.4
    # the kernel is the parametric kernel shifted by that same offset
    for idx, c0 in ((0, CEN_LEFT), (1, CEN_RIGHT)):
        k = e.kernel(idx, c0)
        assert isinstance(k, MeasuredKernel)
        assert abs(k.k.sum() * DX - 1.0) < 1e-9
        grid, k_par = parametric_kernel(idx)
        shifted = np.interp(grid + offs[:, idx].mean(), grid, k_par)
        core = np.abs(grid) <= 1.5
        assert np.max(np.abs(k.k[core] - shifted[core])) < 0.05 * k_par.max()


def test_call_matches_kernel_and_applies_the_envelope_once():
    freqs, pxs, slines = synthetic_sweep(np.random.default_rng(9), n_points=61)
    e = Epsf(freqs, pxs, slines, 2, env_slope=[0.02, -0.03])
    x = np.arange(70.0, 90.0, 0.25)
    prof = e(0, CEN_LEFT + 0.3, x, amp=2.0, offset=1.0)
    k = e.kernel(0, CEN_LEFT + 0.3)
    u = x - (CEN_LEFT + 0.3)
    # zero beyond the +-6 px grid, as the kernel is
    expect = 1.0 + 2.0 * np.interp(u, k.u, k.k, left=0.0, right=0.0) / k.k.max() * np.exp(0.02 * u)
    assert np.allclose(prof, expect, atol=0.01)
    assert abs(prof.max() - 3.0) < 0.1


def test_save_load_roundtrip(tmp_path):
    freqs, pxs, slines = synthetic_sweep(np.random.default_rng(10), n_points=61)
    e = Epsf(freqs, pxs, slines, 2, env_slope=[0.01, -0.01])
    path = tmp_path / "epsf.csv"
    e.save(path)
    e2 = Epsf.load(path)
    for line, c in ((0, CEN_LEFT + 0.3), (1, CEN_RIGHT - 0.7)):
        assert np.allclose(e2.kernel(line, c).k, e.kernel(line, c).k)
        assert np.allclose(e2(line, c, PX), e(line, c, PX))
        assert e2.env_slope(line, c) == pytest.approx(e.env_slope(line, c))
    with pytest.raises(ValueError, match="No template node"):
        e2.kernel(0, CEN_LEFT + 40.0)
