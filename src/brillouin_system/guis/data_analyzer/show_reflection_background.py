"""Viewer for the reflection background the prmr fits will use.

Top: the template's 2D frame (the stored bias-subtracted mean image).
Bottom: its row-band sline — summed with the row count the live fitting
config uses, selected on the template's OWN frame, exactly as the mapper
does at fit time — with the session's calibration sideband positions marked
(the frequency anchor of the template).
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

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        self._plot()

    def _n_rows(self) -> int:
        """The row COUNT of the live fitting config — the band itself is
        re-selected on the template's own frame (its y alignment)."""
        try:
            return len(SpectrumFitter().get_selected_rows(self.background.frame))
        except Exception as e:
            log.warning(f"[ReflectionBackgroundViewer] Could not resolve the "
                        f"configured row band ({e}) — showing all rows.")
            return self.background.frame.shape[0]

    def _plot(self):
        bg = self.background
        ax_frame = self.figure.add_subplot(211)
        ax_sline = self.figure.add_subplot(212)

        show_frame(self.figure, ax_frame, bg.frame,
                   title="Template frame (bias-subtracted mean)")

        n_rows = self._n_rows()
        try:
            sline = bg.sline(n_rows)
            band_label = f"Row-band sline ({n_rows} rows)"
        except ValueError as e:
            log.warning(f"[ReflectionBackgroundViewer] Row band not locatable "
                        f"on the template frame ({e}) — showing the full sum.")
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

        self.figure.tight_layout()
        self.canvas.draw_idle()
