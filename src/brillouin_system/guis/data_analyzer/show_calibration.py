"""Calibration viewer for the data analyzer.

Top: the raw calibration frames themselves, browsable one EOM frequency at
a time, each with its reference fit — same look as the sample viewer.
Bottom: the calibration map (measured sideband points, fitted polynomial,
residuals in MHz — the pixel-response sine lives there), with the selected
frame's point highlighted.

Works for a standalone calibration file (re-fitted from its raw frames with
the current configs), for the calibration carried by an axial scan, and —
frames section hidden — for a legacy scan that stores only the polynomial.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)
from PyQt5.QtCore import Qt

from brillouin_system.calibration.calibration import CalibrationCalculator, CalibrationData
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.guis.data_analyzer.log_panel import LogPanel
from brillouin_system.guis.data_analyzer.plot_helpers import plot_fitted_spectrum, show_frame
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.spectrum_fitting.spectrum_fitter import SpectrumFitter

log = get_logger(__name__)

# outer_* tracks exist only on four-peak calibrations; their views show
# "No calibration model for this reference" on two-peak data.
REFERENCES = ("left", "right", "distance", "outer_left", "outer_right")

_AXIS_LABELS = {
    "left": "Left peak center (px)",
    "right": "Right peak center (px)",
    "distance": "Inter-peak distance (px)",
    "outer_left": "Outer-left peak center (px)",
    "outer_right": "Outer-right peak center (px)",
}


def calibration_points(calc: CalibrationCalculator, reference: str):
    """The measured (px, GHz) points stored with the parameters, or None."""
    p = calc.p
    px, freqs = {
        "left": (p.left_px_points, p.left_freq_points),
        "right": (p.right_px_points, p.right_freq_points),
        "distance": (p.dist_px_points, p.dist_freq_points),
        "outer_left": (p.outer_left_px_points, p.outer_left_freq_points),
        "outer_right": (p.outer_right_px_points, p.outer_right_freq_points),
    }[reference]
    if px is None or freqs is None or len(np.atleast_1d(px)) == 0:
        return None
    px = np.asarray(px, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    valid = np.isfinite(px) & np.isfinite(freqs)
    if not np.any(valid):
        return None
    return px[valid], freqs[valid]


def poly_for_reference(calc: CalibrationCalculator, reference: str):
    return {
        "left": calc.p.freq_left_peak,
        "right": calc.p.freq_right_peak,
        "distance": calc.p.freq_peak_distance,
        "outer_left": calc.p.freq_outer_left_peak,
        "outer_right": calc.p.freq_outer_right_peak,
    }[reference]


@dataclass
class _CalFrame:
    frame: np.ndarray
    set_freq_ghz: float
    microwave_freq: float
    fit: FittedSpectrum


class CalibrationViewer(QWidget):
    """Frame browser + fit + residual view of one calibration, with a marker
    for the point a sample fit currently sits at (optional)."""

    def __init__(self, calculator: CalibrationCalculator, title: str,
                 calibration_data: CalibrationData | None = None,
                 fitter: SpectrumFitter | None = None,
                 current_fit: FittedSpectrum | None = None,
                 parent=None):
        super().__init__(parent)
        self.calc = calculator
        # A sample fit whose point is marked on the map — re-evaluated for
        # whichever reference is selected, so it always lands on that track.
        self.current_fit = current_fit
        self.fitter = fitter if fitter is not None else SpectrumFitter()
        self._colorbar = None

        self.setWindowTitle(title)

        self.cal_frames: list[_CalFrame] = []
        if calibration_data is not None:
            self._fit_calibration_frames(calibration_data)
        self.frame_index = 0
        self.have_frames = len(self.cal_frames) > 0

        self.setMinimumSize(1100, 850 if self.have_frames else 600)
        self._init_ui(title)

        log.info(f"[CalibrationViewer] {title}"
                 + ("" if self.have_frames else " — no raw frames stored, "
                    "showing the stored calibration only"))
        self.calc.print_all_models()
        self.update_display()

    # ---------------- Frame fitting ----------------

    def _fit_calibration_frames(self, data: CalibrationData):
        """One reference fit per stored calibration frame, for display —
        the same fitter (and row band) the calibration itself was fitted
        with, so what is shown IS what calibrate() saw."""
        entries = []
        for freq_block in data.measured_freqs:
            for point in freq_block.cali_meas_points:
                px, sline = self.fitter.get_px_sline_from_image(point.frame)
                fit = self.fitter.fit(px=px, sline=sline,
                                      is_reference_mode=True)
                entries.append(_CalFrame(
                    frame=np.asarray(point.frame, dtype=float),
                    set_freq_ghz=freq_block.set_freq_ghz,
                    microwave_freq=float(point.microwave_freq),
                    fit=fit))
        entries.sort(key=lambda e: e.microwave_freq)
        n_fail = sum(1 for e in entries if not e.fit.is_success)
        log.info(f"[CalibrationViewer] fitted {len(entries)} calibration "
                 f"frames ({n_fail} failed)")
        self.cal_frames = entries

    # ---------------- UI ----------------

    def _init_ui(self, title: str):
        splitter = QSplitter(Qt.Horizontal, self)
        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

        left = QWidget()
        vbox = QVBoxLayout(left)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Reference:"))
        self.ref_combo = QComboBox()
        self.ref_combo.addItems(REFERENCES)
        default_ref = calibration_config.get().reference
        if default_ref in REFERENCES:
            self.ref_combo.setCurrentText(default_ref)
        self.ref_combo.currentTextChanged.connect(lambda _: self.update_display())
        controls.addWidget(self.ref_combo)
        controls.addSpacing(24)

        if self.have_frames:
            self.left_btn = QPushButton("←")
            self.left_btn.setFixedWidth(36)
            self.left_btn.clicked.connect(self._on_prev_frame)
            controls.addWidget(self.left_btn)

            controls.addWidget(QLabel("Frame #:"))
            self.frame_spinner = QSpinBox()
            self.frame_spinner.setRange(0, len(self.cal_frames) - 1)
            self.frame_spinner.valueChanged.connect(self._on_frame_changed)
            controls.addWidget(self.frame_spinner)

            self.right_btn = QPushButton("→")
            self.right_btn.setFixedWidth(36)
            self.right_btn.clicked.connect(self._on_next_frame)
            controls.addWidget(self.right_btn)

            self.frame_label = QLabel()
            self.frame_label.setStyleSheet("font-weight: bold;")
            controls.addSpacing(12)
            controls.addWidget(self.frame_label)

        controls.addStretch()
        vbox.addLayout(controls)

        if self.have_frames:
            self.fig = Figure(figsize=(7, 9), constrained_layout=True)
            gs = self.fig.add_gridspec(4, 1, height_ratios=[1.1, 1.0, 1.4, 0.55])
            self.ax_frame = self.fig.add_subplot(gs[0])
            self.ax_spec = self.fig.add_subplot(gs[1])
            self.ax_fit = self.fig.add_subplot(gs[2])
            self.ax_res = self.fig.add_subplot(gs[3], sharex=self.ax_fit)
        else:
            self.fig = Figure(figsize=(7, 5), constrained_layout=True)
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
            self.ax_frame = None
            self.ax_spec = None
            self.ax_fit = self.fig.add_subplot(gs[0])
            self.ax_res = self.fig.add_subplot(gs[1], sharex=self.ax_fit)

        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(NavigationToolbar(self.canvas, left))
        vbox.addWidget(self.canvas)

        splitter.addWidget(left)
        splitter.addWidget(LogPanel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([800, 300])

    # ---------------- Navigation ----------------

    def _on_frame_changed(self, value: int):
        self.frame_index = value
        self.update_display()

    def _on_prev_frame(self):
        if self.frame_index > 0:
            self.frame_spinner.setValue(self.frame_index - 1)

    def _on_next_frame(self):
        if self.frame_index < len(self.cal_frames) - 1:
            self.frame_spinner.setValue(self.frame_index + 1)

    # ---------------- Plotting ----------------

    def _current_sample_point(self, reference: str):
        """(px, GHz) of the sample fit on the selected reference track."""
        fit = self.current_fit
        if fit is None or not fit.is_success:
            return None
        px = {
            "left": fit.left_peak_center_px,
            "right": fit.right_peak_center_px,
            "distance": fit.inter_peak_distance,
            "outer_left": fit.outer_left_peak_center_px,
            "outer_right": fit.outer_right_peak_center_px,
        }[reference]
        if px is None or not np.isfinite(px):
            return None
        if reference.startswith("outer") and not self.calc.has_outer_tracks():
            return None
        freq = {
            "left": self.calc.freq_left_peak,
            "right": self.calc.freq_right_peak,
            "distance": self.calc.freq_peak_distance,
            "outer_left": self.calc.freq_outer_left_peak,
            "outer_right": self.calc.freq_outer_right_peak,
        }[reference](px)
        if freq is None or not np.isfinite(freq):
            return None
        return float(px), float(freq)

    def _selected_frame_point(self, reference: str):
        """(px, GHz) of the selected frame's fit for the chosen reference."""
        if not self.have_frames:
            return None
        entry = self.cal_frames[self.frame_index]
        if not entry.fit.is_success:
            return None
        px = {
            "left": entry.fit.left_peak_center_px,
            "right": entry.fit.right_peak_center_px,
            "distance": entry.fit.inter_peak_distance,
            "outer_left": entry.fit.outer_left_peak_center_px,
            "outer_right": entry.fit.outer_right_peak_center_px,
        }[reference]
        if px is None or not np.isfinite(px):
            return None
        return float(px), entry.microwave_freq

    def update_display(self):
        reference = self.ref_combo.currentText()

        if self.have_frames:
            entry = self.cal_frames[self.frame_index]
            self.frame_label.setText(
                f"{self.frame_index + 1} / {len(self.cal_frames)} | "
                f"EOM {entry.microwave_freq:.3f} GHz"
                + ("" if entry.fit.is_success else " | FIT FAILED"))

            self._colorbar = show_frame(
                self.fig, self.ax_frame, entry.frame, colorbar=self._colorbar,
                title=f"Calibration frame — EOM {entry.microwave_freq:.3f} GHz")
            try:
                rows = self.fitter.get_selected_rows(entry.frame)
            except Exception:
                rows = None
            if rows:
                for edge in (min(rows) - 0.5, max(rows) + 0.5):
                    self.ax_frame.axhline(edge, color="cyan", ls="--",
                                          lw=1.0, alpha=0.9)
                self.ax_frame.axhspan(min(rows) - 0.5, max(rows) + 0.5,
                                      color="cyan", alpha=0.12, lw=0)

            plot_fitted_spectrum(self.ax_spec, entry.fit,
                                 title="Reference fit")

        self._plot_calibration_map(reference)
        self.canvas.draw()

    def _plot_calibration_map(self, reference: str):
        ax, axr = self.ax_fit, self.ax_res
        ax.cla()
        axr.cla()

        points = calibration_points(self.calc, reference)
        coeffs = poly_for_reference(self.calc, reference)
        have_poly = coeffs is not None and np.all(
            np.isfinite(np.asarray(coeffs, dtype=float)))

        if points is not None:
            px, freqs = points
            ax.plot(px, freqs, "o", ms=4, color="#1f77b4", alpha=0.6,
                    label=f"Measured sidebands (n={len(px)})")

            if have_poly:
                x_min, x_max = float(np.min(px)), float(np.max(px))
                cur = self._current_sample_point(reference)
                if cur is not None:
                    x_min = min(x_min, cur[0])
                    x_max = max(x_max, cur[0])
                x_fit = np.linspace(x_min, x_max, 400)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="#d62728",
                        label=f"Polynomial, degree {len(coeffs) - 1}")

                residuals_mhz = (freqs - np.polyval(coeffs, px)) * 1000.0
                axr.axhline(0.0, color="0.6", lw=0.8)
                axr.plot(px, residuals_mhz, ".", ms=4, color="#1f77b4")
                rms = float(np.sqrt(np.mean(residuals_mhz ** 2)))
                axr.set_title(f"Residuals, rms = {rms:.1f} MHz", fontsize=9)
        elif have_poly:
            # Parameters without stored points (legacy scan) — draw the
            # polynomial over a nominal range so at least the model shows.
            x_fit = np.linspace(0.0, 512.0, 400)
            ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="#d62728",
                    label="Polynomial (no stored measured points)")
        else:
            ax.text(0.5, 0.5, "No calibration model for this reference",
                    ha="center", va="center", transform=ax.transAxes)

        # The selected frame's own fitted point, tied to the map.
        sel = self._selected_frame_point(reference)
        if sel is not None:
            ax.plot(sel[0], sel[1], "o", ms=11, mfc="none", mec="#2ca02c",
                    mew=2, label="Selected frame")
            if points is not None and have_poly:
                res = (sel[1] - float(np.polyval(coeffs, sel[0]))) * 1000.0
                axr.plot(sel[0], res, "o", ms=9, mfc="none", mec="#2ca02c",
                         mew=2)

        cur = self._current_sample_point(reference)
        if cur is not None:
            ax.axvline(cur[0], color="#9467bd", ls=":", lw=1)
            ax.plot(cur[0], cur[1], "D", ms=8, color="#9467bd",
                    label=f"Current sample fit: {cur[0]:.2f} px → "
                          f"{cur[1]:.4f} GHz")

        ax.set_ylabel("Frequency (GHz)")
        ax.set_title(f"Calibration — {reference}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        axr.set_xlabel(_AXIS_LABELS[reference])
        axr.set_ylabel("MHz")
        axr.grid(True, alpha=0.3)
