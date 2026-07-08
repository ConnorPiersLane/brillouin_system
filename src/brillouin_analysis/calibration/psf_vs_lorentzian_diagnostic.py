#!/usr/bin/env python3
"""
Diagnostic: does PSF centering flatten the calibration residuals?

Loads one saved calibration (raw reference frames), rebuilds the dual-chain
calibration once, and compares its two chains — the Lorentzian-centered main
chain (classic) and the PSF-centered variant (empirical instrument
response):

  1. the pixel-phase residual sinusoid amplitude for each (left / right /
     distance). This is the metric from the weekly-update slides: a smaller
     amplitude means the reference centers track the true frequency better at
     every sub-pixel phase.
  2. the reconstructed per-order PSF shapes (skew is visible as an asymmetric
     profile / non-zero centroid).

This is calibration-internal (reference vs reference), so it isolates whether
PSF centering removes the bias at its source, independent of sample fitting.

Usage:
    python psf_vs_lorentzian_diagnostic.py [path_to_calibration_file]
If no path is given, a file dialog opens. Accepts .pkl / .h5 / .hdf5.
"""
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from brillouin_system.calibration.calibration import (
    CalibrationCalculator,
    CalibrationData,
    get_calibration_calculator_from_data,
)
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.saving_and_loading.known_dataclasses_lookup import known_classes
from brillouin_system.saving_and_loading.safe_and_load_hdf5 import (
    dict_to_dataclass_tree,
    load_dict_from_hdf5,
)
from brillouin_system.spectrum_fitting.dho_model import epsf_grid


def load_calibration_file(file_path: str):
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    if file_path.endswith((".h5", ".hdf5")):
        native = load_dict_from_hdf5(file_path)
        return dict_to_dataclass_tree(native, known=known_classes)
    raise ValueError(f"Unsupported file type: {file_path}")


def extract_calibration_data(obj) -> CalibrationData:
    """Accept either a bare CalibrationData or a container that holds one."""
    if isinstance(obj, CalibrationData):
        return obj
    for attr in ("calibration_data", "measured_freqs"):
        if hasattr(obj, "measured_freqs"):
            return obj
        if hasattr(obj, "calibration_data") and obj.calibration_data is not None:
            return obj.calibration_data
    raise ValueError(
        "File does not contain raw calibration frames (CalibrationData). "
        "PSF reconstruction needs the reference frames, not just the fitted "
        "parameters — this must be a calibration saved with frames."
    )


def sinusoid(x, amplitude, phase, offset):
    return amplitude * np.sin(2 * np.pi * x + phase) + offset


def fit_sinusoid_amplitude(px_points, residuals_mhz):
    x = np.asarray(px_points, dtype=float) % 1.0
    y = np.asarray(residuals_mhz, dtype=float)
    from scipy.optimize import curve_fit
    p0 = [0.5 * (np.max(y) - np.min(y)), 0.0, float(np.mean(y))]
    try:
        popt, _ = curve_fit(sinusoid, x, y, p0=p0)
        return x, abs(float(popt[0])), popt
    except Exception:
        return x, float("nan"), None


def residuals_for(calculator, side):
    p = calculator.p
    if side == "left":
        px, freq, model = p.left_px_points, p.left_freq_points, calculator.freq_left_peak
    elif side == "right":
        px, freq, model = p.right_px_points, p.right_freq_points, calculator.freq_right_peak
    else:
        px, freq, model = p.dist_px_points, p.dist_freq_points, calculator.freq_peak_distance
    px = np.asarray(px, dtype=float)
    freq = np.asarray(freq, dtype=float)
    resid_mhz = (np.asarray(model(px), dtype=float) - freq) * 1000.0
    return px, resid_mhz


def run(data: CalibrationData):
    degree = calibration_config.get().degree

    # One dual-chain calibration: the top-level parameters are the
    # Lorentzian-centered main chain, the PSF-centered chain is the variant.
    calc_lor = get_calibration_calculator_from_data(data, degree)

    variant = calc_lor.p.psf_variant
    psf_available = variant is not None and variant.psf_left is not None
    calc_psf = CalibrationCalculator(variant) if psf_available else calc_lor
    print("=" * 60)
    print(f"Calibration degree: {degree}")
    print(f"PSF reconstruction succeeded: {psf_available}")
    if not psf_available:
        print("  -> no PSF variant chain (reconstruction not usable). The two "
              "rows below will look identical; see the console warning "
              "printed during calibrate() for the reason.")
    print("=" * 60)

    sides = ["left", "right", "dist"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True)

    for row, (label, calc) in enumerate([("lorentzian", calc_lor), ("psf", calc_psf)]):
        for col, side in enumerate(sides):
            px, resid = residuals_for(calc, side)
            xph, amp, popt = fit_sinusoid_amplitude(px, resid)
            rms = float(np.std(resid))
            print(f"[{label:10s}] {side:5s}: sinusoid amp = {amp:6.2f} MHz | rms = {rms:6.2f} MHz")

            ax = axes[row, col]
            ax.scatter(xph, resid, s=6)
            if popt is not None:
                xx = np.linspace(0, 1, 400)
                ax.plot(xx, sinusoid(xx, *popt), linewidth=2)
            ax.axhline(0.0, linestyle="--", linewidth=0.8)
            ax.set_title(f"{label} — {side} (amp {amp:.1f} MHz)")
            ax.set_xlabel("pixel phase")
            ax.set_ylabel("model - measured (MHz)")
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
        print("-" * 60)

    fig.suptitle("Calibration residuals folded into one pixel: lorentzian (top) vs PSF (bottom)")

    # PSF shapes
    if psf_available:
        step = calc_psf.p.psf_grid_step_px
        pl = np.asarray(calc_psf.p.psf_left, dtype=float)
        pr = np.asarray(calc_psf.p.psf_right, dtype=float)
        ul, ur = epsf_grid(pl, step), epsf_grid(pr, step)
        cl = float(np.sum(ul * pl) / np.sum(pl))
        cr = float(np.sum(ur * pr) / np.sum(pr))
        print(f"PSF centroid (skew) left = {cl:+.3f} px, right = {cr:+.3f} px")
        print("  (skew shows as a non-zero centroid; the slides predict the "
              "LEFT order is skewed and the right is ~symmetric)")

        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        ax2[0].plot(ul, pl / pl.max())
        ax2[0].axvline(0, linestyle="--", linewidth=0.8)
        ax2[0].set_title(f"left ePSF (centroid {cl:+.3f} px)")
        ax2[0].set_xlabel("pixel offset")
        ax2[0].grid(True, alpha=0.3)
        ax2[1].plot(ur, pr / pr.max())
        ax2[1].axvline(0, linestyle="--", linewidth=0.8)
        ax2[1].set_title(f"right ePSF (centroid {cr:+.3f} px)")
        ax2[1].set_xlabel("pixel offset")
        ax2[1].grid(True, alpha=0.3)

    plt.show()


def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    if file_path is None:
        from PyQt5.QtWidgets import QApplication, QFileDialog
        app = QApplication(sys.argv)
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Open Calibration File (with frames)", "",
            "Calibration Files (*.pkl *.h5 *.hdf5);;All Files (*)",
        )
        if not file_path:
            return
    obj = load_calibration_file(file_path)
    data = extract_calibration_data(obj)
    run(data)


if __name__ == "__main__":
    main()
