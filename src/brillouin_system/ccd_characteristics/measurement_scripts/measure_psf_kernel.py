"""Measure the camera PSF kernel (sigma, tau_left, tau_right) [px].

Results: the [psf] section of ccd_characteristics.toml. Measured 2026-07
on the fine EOM sweeps: sigma = 0.25, tau_left = 0.40, tau_right = 0.20 px,
stable across 6 calibrations over 7 weeks (and the sigma/tau values were
re-confirmed alignment-stable across 3 months in the 2026-08 background-
model work). Re-measure after any camera / ROI / readout change.

THE METHOD, and why it looks like this
--------------------------------------
The EOM calibration sidebands are spectrally sharp next to anything the
spectrometer can resolve, so a dense calibration sweep samples the SAME
line at many sub-pixel positions. If the fit model's lineshape is wrong,
the fitted centre picks up a bias that depends on WHERE inside a pixel the
peak sits — visible as a per-pixel-periodic residual "sine" (~±7 MHz) in
the px→GHz calibration tracks. That sine was traced to the camera pixel
response (2026-05/07): a Gaussian charge-diffusion blur (sigma) plus a
one-sided exponential readout smear (tau, toward higher pixel numbers /
the charge-transfer direction), different for the left and right peak
because the smear is a POSITION property on the sensor.

The kernel is therefore measured by MINIMIZING THE RESIDUAL SINE: fit the
whole calibration with 'lorentzian_x_psf' at trial (sigma, tau_l, tau_r),
and take the triple that minimizes the rms residual of the left+right
tracks against their polynomials. Per-frame sigma/tau/gamma are DEGENERATE
— only the sweep-level residual identifies the kernel, which is why this
is a grid search over calibrate() calls and not a per-frame fit.

Use a DENSE sweep (fine EOM steps; the 2026-5-18 data in
Data/2026-5-18-calibration/ is the reference set). Expect a few minutes:
every grid point re-fits every calibration frame.

USAGE
    python measure_psf_kernel.py calibration.pkl [--degree 2] [--coarse-only]
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from dataclasses import replace

import numpy as np


def load_calibration(path: str):
    if path.endswith((".h5", ".hdf5")):
        from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
        from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
            dict_to_dataclass_tree, load_dict_from_hdf5)
        return dict_to_dataclass_tree(load_dict_from_hdf5(path), known_classes)
    with open(path, "rb") as f:
        return pickle.load(f)


def residual_rms_mhz(data, degree: int, sigma: float, tau_l: float,
                     tau_r: float) -> float:
    """rms of the left+right track residuals [MHz] at one kernel triple."""
    from brillouin_system.calibration.calibration import calibrate
    from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

    fitter = SpectrumFitter()
    fitter.update_reference_config(
        replace(fitter.reference_config, fitting_model="lorentzian_x_psf"))
    # The kernel working values ride in the [global] sline config.
    fitter.update_sline_config(replace(
        fitter.sline_config, psf_sigma_px=sigma,
        psf_tau_left_px=tau_l, psf_tau_right_px=tau_r))

    p = calibrate(data=data, poyfit_degree=degree, fitter=fitter)

    res = []
    for px, freqs, coeffs in (
            (p.left_px_points, p.left_freq_points, p.freq_left_peak),
            (p.right_px_points, p.right_freq_points, p.freq_right_peak)):
        px = np.asarray(px, dtype=float)
        freqs = np.asarray(freqs, dtype=float)
        res.append((freqs - np.polyval(coeffs, px)) * 1000.0)
    res = np.concatenate(res)
    return float(np.sqrt(np.mean(res ** 2)))


def grid_search(data, degree: int, sigmas, taus_l, taus_r) -> tuple:
    best = None
    for sigma, tl, tr in itertools.product(sigmas, taus_l, taus_r):
        try:
            rms = residual_rms_mhz(data, degree, sigma, tl, tr)
        except Exception as e:
            print(f"  ({sigma:.2f}, {tl:.2f}, {tr:.2f}) failed: {e}")
            continue
        print(f"  sigma {sigma:.3f}  tau_l {tl:.3f}  tau_r {tr:.3f}  "
              f"-> rms {rms:.2f} MHz")
        if best is None or rms < best[0]:
            best = (rms, sigma, tl, tr)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="CalibrationData .pkl/.h5 (dense EOM sweep)")
    ap.add_argument("--degree", type=int, default=2,
                    help="track polynomial degree (default 2)")
    ap.add_argument("--coarse-only", action="store_true",
                    help="skip the refinement pass")
    args = ap.parse_args(argv)

    data = load_calibration(args.file)

    print("Baseline (no kernel, i.e. plain-Lorentzian-equivalent tiny kernel):")
    base = residual_rms_mhz(data, args.degree, 0.01, 0.0, 0.0)
    print(f"  rms {base:.2f} MHz — the sine the kernel must remove\n")

    print("Coarse grid:")
    best = grid_search(
        data, args.degree,
        sigmas=np.arange(0.10, 0.46, 0.05),
        taus_l=np.arange(0.0, 0.61, 0.10),
        taus_r=np.arange(0.0, 0.61, 0.10))
    if best is None:
        print("All grid points failed.")
        return 1

    if not args.coarse_only:
        _, s0, tl0, tr0 = best
        print("\nRefinement:")
        best = grid_search(
            data, args.degree,
            sigmas=np.arange(max(s0 - 0.05, 0.01), s0 + 0.051, 0.025),
            taus_l=np.arange(max(tl0 - 0.10, 0.0), tl0 + 0.101, 0.05),
            taus_r=np.arange(max(tr0 - 0.10, 0.0), tr0 + 0.101, 0.05))

    rms, sigma, tl, tr = best
    print(f"\n==== Best kernel ====")
    print(f"sigma {sigma:.3f} px, tau_left {tl:.3f} px, tau_right {tr:.3f} px "
          f"-> rms {rms:.2f} MHz (baseline {base:.2f})")
    print("\nA re-measurement updates BOTH files:")
    print("ccd_characteristics.toml [psf] (the measurement record):")
    print(f"psf_sigma_px = {sigma:.3g}")
    print(f"psf_tau_left_px = {tl:.3g}")
    print(f"psf_tau_right_px = {tr:.3g}")
    print('psf_measured = "<date>"')
    print("find_peaks_config.toml [global] (the working values the fitter "
          "uses):")
    print(f"psf_sigma_px = {sigma:.3g}")
    print(f"psf_tau_left_px = {tl:.3g}")
    print(f"psf_tau_right_px = {tr:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
