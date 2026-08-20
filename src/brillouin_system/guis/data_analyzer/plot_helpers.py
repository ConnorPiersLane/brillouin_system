"""Shared plotting for the data-analyzer viewers: the raw Andor frame and
a fitted spectrum with clearly marked fit windows."""
from __future__ import annotations

import numpy as np

from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum


def show_frame(fig, ax, frame: np.ndarray, colorbar=None,
               title: str = "Andor Frame (raw)"):
    """Draw a raw camera frame with percentile contrast; returns the colorbar
    (pass it back on the next call so it is reused, not stacked)."""
    ax.cla()
    frame = np.asarray(frame, dtype=float)
    vmin, vmax = np.percentile(frame, [1.0, 99.7])
    if vmax <= vmin:
        vmin, vmax = float(frame.min()), float(frame.max() or 1.0)
    im = ax.imshow(frame, cmap="magma", aspect="auto", interpolation="none",
                   origin="upper", vmin=vmin, vmax=vmax)
    if colorbar is None:
        colorbar = fig.colorbar(im, ax=ax, pad=0.01, label="Counts")
    else:
        colorbar.update_normal(im)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Pixel (X)")
    ax.set_ylabel("Pixel (Y)")
    return colorbar


def _mask_spans(x: np.ndarray, mask: np.ndarray):
    """Contiguous True runs of the fit mask as (x_lo, x_hi) spans."""
    spans = []
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            spans.append((x[start] - 0.5, x[i - 1] + 0.5))
            start = None
    if start is not None:
        spans.append((x[start] - 0.5, x[-1] + 0.5))
    return spans


def plot_fitted_spectrum(ax, fit: FittedSpectrum, title: str | None = None):
    """Spectrum + fit with the used pixels made unmissable: the fit windows
    are shaded and the used points drawn as filled orange markers on top of
    the open gray data circles."""
    ax.cla()
    ax.plot(fit.x_pixels, fit.sline, "o", ms=3.5, mfc="none", mec="0.55",
            mew=0.8, label="Spectrum")

    if fit.is_success and fit.mask_for_fitting is not None:
        first = True
        for lo, hi in _mask_spans(fit.x_pixels, fit.mask_for_fitting):
            ax.axvspan(lo, hi, color="#ff7f0e", alpha=0.12, lw=0,
                       label="Fit window" if first else None)
            first = False
        ax.plot(fit.x_pixels[fit.mask_for_fitting],
                fit.sline[fit.mask_for_fitting],
                "o", ms=4.5, color="#ff7f0e", mec="#b25000", mew=0.5,
                label="Used for fit")
        ax.plot(fit.x_fit_refined, fit.y_fit_refined, "-",
                color="#d62728", lw=1.4, label="Fit")
        for px in (fit.left_peak_center_px, fit.right_peak_center_px):
            if px is not None and np.isfinite(px):
                ax.axvline(px, color="#d62728", ls=":", lw=0.8, alpha=0.6)

    if title:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("Pixel (X)")
    ax.set_ylabel("Intensity (Σ rows)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
