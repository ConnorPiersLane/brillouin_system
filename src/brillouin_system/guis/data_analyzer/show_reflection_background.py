"""Viewer for the reflection background the prmr fits will use.

Top: the template's 2D frame (the stored bias-subtracted mean image).
Bottom: its row-band sline, with the session's calibration sideband
positions marked (the frequency anchor of the template).

At fit time the template is collapsed over the SAMPLE scan's row band
(user decision 2026-08-26 — see ReflectionBackground.sline). A manual
config band is exactly that band, so the viewer shows it. An auto band is
located per scan on the sample frames and cannot be known from the template
alone, so the viewer then shows the all-rows sum and says so.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from brillouin_system.guis.data_analyzer.plot_helpers import show_frame
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackground,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

log = get_logger(__name__)


def _meta_line(bg: ReflectionBackground) -> str:
    m = bg.meta
    parts = []
    if m.get("scan_id"):
        parts.append(f"scan {m.get('scan_i', '?')} - {m['scan_id']}")
    if m.get("source"):
        parts.append(f"source: {m['source']}")
    if m.get("built"):
        parts.append(f"built {m['built']}")
    n_scans = m.get("n_scans")
    n_frames = m.get("frames_per_scan")
    if n_scans and n_frames:
        parts.append(f"{n_scans} scan(s) x {n_frames} frames")
    if m.get("bias_counts") is not None:
        parts.append(f"bias {m['bias_counts']:.1f} counts")
    parts.append(f"cal {bg.cal_freqs.min():.2f}-{bg.cal_freqs.max():.2f} GHz "
                 f"({len(bg.cal_freqs)} points)")
    if m.get("notes"):
        parts.append(m["notes"])
    return "  |  ".join(parts)


class ReflectionBackgroundViewer(QWidget):
    def __init__(self, background: ReflectionBackground,
                 title: str = "Reflection Background"):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(850, 650)
        self.background = background

        layout = QVBoxLayout(self)
        meta = QLabel(_meta_line(background))
        meta.setWordWrap(True)
        layout.addWidget(meta)

        self.figure = Figure(figsize=(8, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        self._plot()

    def _band_rows(self) -> list[int] | None:
        """The ROW INDICES of the live fitting config, or None when the band
        cannot be known here (auto mode locates it per scan, on the sample
        frames — the viewer then falls back to the all-rows sum)."""
        try:
            return SpectrumFitter().get_selected_rows()
        except Exception as e:
            log.info(f"[ReflectionBackgroundViewer] Row band not available "
                     f"from the config alone ({e}) — showing all rows.")
            return None

    def _plot(self):
        bg = self.background
        # The colorbar gets its own column so it does not shrink the frame
        # axes: frame (top) and sline (bottom) then share one x axis and the
        # pixel columns line up vertically.
        gs = self.figure.add_gridspec(2, 2, width_ratios=[40, 1], wspace=0.03)
        ax_frame = self.figure.add_subplot(gs[0, 0])
        cax = self.figure.add_subplot(gs[0, 1])
        ax_sline = self.figure.add_subplot(gs[1, 0], sharex=ax_frame)

        show_frame(self.figure, ax_frame, bg.frame,
                   title="Template frame (bias-subtracted mean)", cax=cax)
        ax_frame.set_xlabel("")
        ax_frame.tick_params(labelbottom=False)

        rows = self._band_rows()
        try:
            sline = bg.sline(rows)
            band_label = (f"Row-band sline (rows {min(rows)}-{max(rows)})"
                          if rows is not None else
                          "Sline (all rows — auto band is per-scan)")
        except ValueError as e:
            log.warning(f"[ReflectionBackgroundViewer] None of the configured "
                        f"rows exist on the template frame ({e}) — showing "
                        f"the full sum.")
            sline = bg.sline(None)
            band_label = "Sline (all rows)"
        ax_sline.plot(bg.px, sline, "-", color="0.3", lw=1.0,
                      label=band_label)
        for freqs_px, color, name in ((bg.cal_left_px, "#1f77b4", "AS-order cal"),
                                      (bg.cal_right_px, "#d62728", "S-order cal")):
            first = True
            for x in freqs_px:
                ax_sline.axvline(x, color=color, ls=":", lw=0.7, alpha=0.6,
                                 label=name if first else None)
                first = False
        ax_sline.set_xlabel("Pixel (X)")
        ax_sline.set_ylabel("Intensity (Σ rows)")
        ax_sline.grid(True, alpha=0.3)
        ax_sline.legend(fontsize=8, loc="upper right")

        self.canvas.draw_idle()
