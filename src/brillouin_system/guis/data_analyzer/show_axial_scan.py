"""Viewer for one axial scan: frame, spectrum fit, and depth profile,
with a right-hand log column and an SNR analysis that puts the measured
scatter next to the Thompson shot-noise bound and a Monte Carlo re-fit
of synthetic frames.

SNR methodology (user-confirmed 2026-08-20, thompson-scan-series recipe):
* Thompson: ONE bound per scan, evaluated at the scan-MEAN fit parameters
  (mean amplitude/width/position/pedestal) — never an average of per-frame
  bounds. Intensity/common-mode noise stays out of the shift bound (it is
  multiplicative and moves no centre), which is why N uses the Poisson
  gain, never a linear-PTC slope.
* Measured: drift-immune sd from consecutive-frame differences / sqrt(2),
  next to the plain sd.
* Monte Carlo: truth = the SCAN-MEAN frame with the dark level removed
  (the Figure-3 recipe), so real-frame structure the model misses —
  pedestal, side-order tails — is in the synthetic frames; noise is
  Poisson in electrons at the measured gain plus per-pixel read noise,
  re-fit through the untouched production pipeline.

Frames are never dark-subtracted for fitting or display (user rule
2026-08-20): the fit's background absorbs the dark level, and the
Thompson bound removes it analytically (pedestal_bias_counts).
"""
from __future__ import annotations

import traceback
from dataclasses import replace as dc_replace

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)
from scipy.stats import norm
from tifffile import imwrite

from brillouin_system.calibration.calibration import (
    AnalyzedFreqShifts, CalibrationCalculator,
)
from brillouin_system.calibration.config.calibration_config import calibration_config
from brillouin_system.guis.data_analyzer.excel_export_axial_scan import (
    BrillouinExport, export_to_excel, get_excel_row_data, load_from_excel,
)
from brillouin_system.guis.data_analyzer.log_panel import LogPanel
from brillouin_system.guis.data_analyzer.plot_helpers import plot_fitted_spectrum, show_frame
from brillouin_system.guis.data_analyzer.show_calibration import CalibrationViewer
from brillouin_system.logging_utils.logging_setup import get_logger
from brillouin_system.my_dataclasses.fitted_spectrum import FittedSpectrum
from brillouin_system.my_dataclasses.human_interface_measurements import (
    AnalyzedSpectrum, AxialScan, calibration_for_scan, fit_axial_scan,
    fitter_for_scan,
)
from brillouin_system.ccd_characteristics import ccd_config
from brillouin_system.spectrum_fitting.noise_analysis import (
    MonteCarloFrames, PixelCountsAndPhotons,
    TheoreticalPeakStdError, electrons_per_count, theoretical_precision,
)
from brillouin_system.spectrum_fitting.reflection_background import (
    ReflectionBackground, ReflectionBackgroundMapper,
)
from brillouin_system.spectrum_fitting.spectrum_fitter import (
    config_requires_reflection_background,
)

log = get_logger(__name__)


def fmt(val, precision=3):
    return f"{val:.{precision}f}" if val is not None else "N/A"


def reference_freq(shifts: AnalyzedFreqShifts, reference: str) -> float | None:
    if reference == "left":
        return shifts.freq_shift_left_peak_ghz
    if reference == "right":
        return shifts.freq_shift_right_peak_ghz
    if reference == "distance":
        return shifts.freq_shift_peak_distance_ghz
    raise ValueError(
        f"Unknown reference '{reference}'. Use 'left', 'right', or 'distance'.")


def reference_theo_total_mhz(theo, reference: str) -> float | None:
    """The Thompson total of the observable the reference selects."""
    if reference == "left":
        return theo.left_peak_total_mhz
    if reference == "right":
        return theo.right_peak_total_mhz
    if reference == "distance":
        return theo.distance_total_mhz
    return None


class AxialScanViewer(QWidget):
    """GUI for visualizing and analyzing axial scan data."""

    def __init__(self, axial_scan: AxialScan):
        super().__init__()
        self.axial_scan: AxialScan = axial_scan
        self.setWindowTitle(f"Axial Scan Viewer - ID: {axial_scan.id}")
        self.setMinimumSize(1200, 800)

        self.peak_reference = calibration_config.get().reference

        # One fitter + one calibration re-fit, shared by the whole viewer
        # (fit pipeline, calibration plot, Monte Carlo).
        self.fitter = fitter_for_scan(axial_scan)
        self.calc: CalibrationCalculator = calibration_for_scan(axial_scan, self.fitter)
        self.list_analyzed_spectras: list[AnalyzedSpectrum] = fit_axial_scan(
            axial_scan, fitter=self.fitter, calibration_calculator=self.calc)
        self.freq_shifts: list[float | None] = [
            reference_freq(a.analyzed_shifts, self.peak_reference)
            for a in self.list_analyzed_spectras
        ]
        self.z_positions = np.array(
            [m.lens_zaber_position for m in axial_scan.measurements], dtype=float)

        self.current_index = 0
        self.open_windows = []
        self._colorbar = None

        self.init_ui()
        self.print_scan_overview()
        self.update_display()

    # ---------------- UI Setup ----------------

    def init_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        left = QWidget()
        layout = QVBoxLayout(left)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.info_label)

        layout.addWidget(self.setup_plot_area())

        controls = QVBoxLayout()
        controls.addLayout(self.setup_navigation())
        controls.addLayout(self.setup_second_row_controls())
        layout.addLayout(controls)

        splitter.addWidget(left)
        splitter.addWidget(LogPanel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 300])

    def setup_plot_area(self) -> QGroupBox:
        group = QGroupBox("Axial Scan Data")
        vbox = QVBoxLayout()

        self.fig = Figure(figsize=(8, 7), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.2, 1, 1])
        self.ax_img = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1])
        self.ax_axial = self.fig.add_subplot(gs[2])

        self.canvas = FigureCanvas(self.fig)
        vbox.addWidget(NavigationToolbar(self.canvas, group))
        vbox.addWidget(self.canvas)
        group.setLayout(vbox)
        return group

    def setup_navigation(self) -> QHBoxLayout:
        nav_layout = QHBoxLayout()

        self.left_btn = QPushButton("←")
        self.left_btn.clicked.connect(self.on_left_clicked)
        nav_layout.addWidget(self.left_btn)

        nav_layout.addWidget(QLabel("Measurement #:"))
        self.index_spinner = QSpinBox()
        self.index_spinner.setRange(0, len(self.axial_scan.measurements) - 1)
        self.index_spinner.valueChanged.connect(self.on_index_changed)
        nav_layout.addWidget(self.index_spinner)

        self.right_btn = QPushButton("→")
        self.right_btn.clicked.connect(self.on_right_clicked)
        nav_layout.addWidget(self.right_btn)

        self.analyze_btn = QPushButton("Analyze SNR")
        self.analyze_btn.clicked.connect(self.on_analyze_snr)
        nav_layout.addWidget(self.analyze_btn)

        nav_layout.addWidget(QLabel("MC frames:"))
        self.mc_spinner = QSpinBox()
        self.mc_spinner.setRange(20, 2000)
        self.mc_spinner.setValue(100)
        self.mc_spinner.setToolTip(
            "Number of synthetic frames the Monte Carlo re-fits.")
        nav_layout.addWidget(self.mc_spinner)

        nav_layout.addStretch()
        return nav_layout

    def setup_second_row_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.show_cal_plot_btn = QPushButton("Show Calibration Plot")
        self.show_cal_plot_btn.clicked.connect(self.on_show_calibration_plot)
        row.addWidget(self.show_cal_plot_btn)

        self.save_new_btn = QPushButton("Save New Excel")
        self.save_new_btn.clicked.connect(self.save_new_excel)
        row.addWidget(self.save_new_btn)

        self.add_existing_btn = QPushButton("Add to Existing Excel")
        self.add_existing_btn.clicked.connect(self.add_to_existing_excel)
        row.addWidget(self.add_existing_btn)

        self.export_frame_btn = QPushButton("Export Frame (uint8)")
        self.export_frame_btn.clicked.connect(self.export_current_frame)
        row.addWidget(self.export_frame_btn)

        row.addStretch()
        return row

    # ---------------- Plotting ----------------

    def _raw_frame(self, index: int) -> np.ndarray:
        """The ORIGINAL Andor frame, exactly as recorded — nothing
        subtracted anywhere: display and fit both use raw frames; the
        dark level is handled analytically in the Thompson bound."""
        return np.asarray(self.axial_scan.measurements[index].frame_andor,
                          dtype=float)

    def update_display(self):
        mp = self.axial_scan.measurements[self.current_index]
        self.print_measurement_info()

        self.info_label.setText(
            f"ID: {self.axial_scan.id} | "
            f"Index: {self.current_index + 1} / {len(self.axial_scan.measurements)} | "
            f"Z pos: {mp.lens_zaber_position:.2f} µm | "
            f"Reference: {self.peak_reference}"
        )

        self.plot_frame(self._raw_frame(self.current_index))
        self.plot_spectrum()
        self.plot_axial_scan()

        self.canvas.draw()

    def plot_frame(self, frame: np.ndarray):
        self._colorbar = show_frame(self.fig, self.ax_img, frame,
                                    colorbar=self._colorbar)

        # Summed row band (the band the ANALYSIS used — live config) and
        # fitted peak columns.
        try:
            rows = self.fitter.get_selected_rows(frame)
        except Exception:
            rows = None
        if rows:
            for edge in (min(rows) - 0.5, max(rows) + 0.5):
                self.ax_img.axhline(edge, color="cyan", ls="--", lw=1.0,
                                    alpha=0.9)
            self.ax_img.axhspan(min(rows) - 0.5, max(rows) + 0.5,
                                color="cyan", alpha=0.12, lw=0)
            self.ax_img.text(
                0.995, 0.03, f"summed rows {min(rows)}-{max(rows)}",
                transform=self.ax_img.transAxes, ha="right", va="bottom",
                fontsize=7, color="cyan")

        fit = self.list_analyzed_spectras[self.current_index].fitted_spectrum
        if fit.is_success:
            for px in (fit.left_peak_center_px, fit.right_peak_center_px):
                self.ax_img.axvline(px, color="w", ls=":", lw=0.8, alpha=0.7)

    def plot_spectrum(self):
        fit: FittedSpectrum = self.list_analyzed_spectras[self.current_index].fitted_spectrum
        z = self.axial_scan.measurements[self.current_index].lens_zaber_position
        plot_fitted_spectrum(self.ax_spec, fit,
                             title=f"Spectrum at Z = {round(z)} µm")

    def plot_axial_scan(self):
        self.ax_axial.cla()

        shifts = np.array(
            [s if s is not None else np.nan for s in self.freq_shifts],
            dtype=float)
        theo_mhz = np.array([
            reference_theo_total_mhz(a.theoretical_precisions, self.peak_reference)
            or np.nan
            for a in self.list_analyzed_spectras
        ], dtype=float)

        # Measured axis: the lens position, with the Thompson bound of each
        # point as an error bar (the floor, not the expected scatter).
        self.ax_axial.errorbar(
            self.z_positions, shifts, yerr=theo_mhz / 1000.0,
            fmt="o-", ms=4, lw=1, color="#1f77b4", ecolor="#1f77b4",
            elinewidth=0.8, capsize=2, alpha=0.85,
            label=f"Shift ({self.peak_reference}) ± Thompson bound")

        y_val = shifts[self.current_index]
        if np.isfinite(y_val):
            self.ax_axial.plot(self.z_positions[self.current_index], y_val,
                               "o", ms=10, mfc="none", mec="#d62728", mew=2,
                               label="Current")
            self.ax_axial.set_title(f"Freq (GHz): {y_val:.4f}", fontsize=10)

        self.ax_axial.set_xlabel("Lens Z position (µm)")
        self.ax_axial.set_ylabel("Frequency Shift (GHz)")
        self.ax_axial.grid(True, alpha=0.3)
        self.ax_axial.legend(fontsize=8)

    # ---------------- Navigation ----------------

    def on_index_changed(self, value: int):
        self.current_index = value
        self.update_display()

    def on_left_clicked(self):
        if self.current_index > 0:
            self.index_spinner.setValue(self.current_index - 1)

    def on_right_clicked(self):
        if self.current_index < len(self.axial_scan.measurements) - 1:
            self.index_spinner.setValue(self.current_index + 1)

    # ---------------- SNR analysis ----------------

    def _scan_mean_thompson(self) -> TheoreticalPeakStdError | None:
        """ONE Thompson bound for the scan, at the scan-MEAN fit parameters.

        The thompson-scan-series recipe: average the fitted amplitude,
        width, position and pedestal over the frames first, then evaluate
        the bound once — never average per-frame bounds.
        """
        fits = [a.fitted_spectrum for a in self.list_analyzed_spectras
                if a.fitted_spectrum.is_success]
        if not fits:
            return None

        def mean_of(attr):
            vals = [getattr(f, attr) for f in fits]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            return float(np.mean(vals)) if vals else None

        mean_fs = dc_replace(
            fits[0],
            left_peak_center_px=mean_of("left_peak_center_px"),
            left_peak_width_px=mean_of("left_peak_width_px"),
            left_peak_amplitude=mean_of("left_peak_amplitude"),
            right_peak_center_px=mean_of("right_peak_center_px"),
            right_peak_width_px=mean_of("right_peak_width_px"),
            right_peak_amplitude=mean_of("right_peak_amplitude"),
            inter_peak_distance=mean_of("inter_peak_distance"),
            left_peak_bg_counts=mean_of("left_peak_bg_counts"),
            right_peak_bg_counts=mean_of("right_peak_bg_counts"),
        )

        info = self.axial_scan.system_state.andor_camera_info
        photons = PixelCountsAndPhotons.from_fit(
            fs=mean_fs, preamp_gain=info.preamp_gain, emccd_gain=info.gain)
        dark = self.axial_scan.system_state.dark_image
        # Raw-frame fits: the fitted pedestal contains the dark/bias level,
        # which carries no shot noise (same handling as fit_axial_scan) —
        # the scan's own dark stack wins, the ccd_characteristics reference
        # value is the fallback, a frame median only if even that is unset.
        first_frame = np.asarray(
            self.axial_scan.measurements[0].frame_andor, dtype=float)
        if dark is not None:
            level = float(np.median(dark.mean_image))
        else:
            level = (ccd_config.get().dark_median_counts
                     or float(np.median(first_frame)))
        rows = self.fitter.get_selected_rows(first_frame)
        bias_counts = level * len(rows)
        return theoretical_precision(
            fs=mean_fs, photons=photons, calibration_calculator=self.calc,
            dark_frame_std=dark.std_image if dark is not None else None,
            preamp_gain=info.preamp_gain, emccd_gain=info.gain,
            pedestal_bias_counts=bias_counts,
            sline_rows=rows)

    def _measured_background_for_fit(self, px: np.ndarray) -> np.ndarray | None:
        """The reflection background rendered for this scan, when the live
        sample config asks for it (prmr) — same construction as
        fit_axial_scan."""
        if self.axial_scan.system_state.is_reference_mode:
            return None
        if not config_requires_reflection_background(self.fitter.sample_config):
            return None
        rows = self.fitter.get_selected_rows(
            np.asarray(self.axial_scan.measurements[0].frame_andor))
        mapper = ReflectionBackgroundMapper(
            ReflectionBackground.load_default(), self.calc,
            n_rows=len(rows))
        return mapper.render(px)

    def _scan_mean_frame(self) -> np.ndarray:
        """The MC truth (Fig-3 recipe): the scan-mean frame with the dark
        level taken out.

        The DATA are fitted raw, but the MC truth must be light-only — the
        dark/bias pedestal carries no shot noise (only read noise, which
        the generator adds separately). Per-pixel dark-stack mean when
        darks were taken; a scalar median otherwise.
        """
        dark = self.axial_scan.system_state.dark_image
        mean_frame = np.mean([np.asarray(m.frame_andor, dtype=float)
                              for m in self.axial_scan.measurements], axis=0)
        if dark is not None:
            mean_frame = mean_frame - dark.mean_image
        else:
            bias = (ccd_config.get().dark_median_counts
                    or float(np.median(mean_frame)))
            log.info(f"[MC] no dark stack — subtracting the reference dark "
                     f"level {bias:.1f} counts/px (ccd_characteristics)")
            mean_frame = mean_frame - bias
        return np.clip(mean_frame, 0.0, None)

    def _warn_if_scan_moves(self):
        """Mean-frame truth assumes repeated measurements of ONE spectrum."""
        z = self.z_positions[np.isfinite(self.z_positions)]
        if z.size and (z.max() - z.min()) > 1.0:
            log.warning(f"[MC] Z positions span {z.max() - z.min():.1f} µm — "
                        f"the scan-mean-frame truth blurs a moving spectrum; "
                        f"MC width will be pessimistic.")
        centers = np.array([a.fitted_spectrum.left_peak_center_px
                            for a in self.list_analyzed_spectras
                            if a.fitted_spectrum.is_success], dtype=float)
        if centers.size and (centers.max() - centers.min()) > 0.3:
            log.warning(f"[MC] fitted peak centre wanders "
                        f"{centers.max() - centers.min():.2f} px over the "
                        f"scan — mean-frame truth is broadened by the drift.")

    def _run_monte_carlo(self, n_frames: int) -> np.ndarray:
        """Re-fit synthetic frames around the scan-mean frame and return the
        resulting reference-frequency samples (GHz).

        The Figure-3 MC recipe: truth = the measured scan-mean frame (so
        real-frame structure the model misses — pedestal, side-order tails —
        is in the synthetic data), shot noise Poisson in electrons at the
        measured gain, per-pixel read noise, then the untouched production
        pipeline (row sum -> fit -> calibration).
        """
        if not any(a.fitted_spectrum.is_success
                   for a in self.list_analyzed_spectras):
            raise ValueError("No successful fits in this scan.")

        self._warn_if_scan_moves()

        info = self.axial_scan.system_state.andor_camera_info
        e_per_count = electrons_per_count(
            preamp_gain=info.preamp_gain, emccd_gain=info.gain)

        dark = self.axial_scan.system_state.dark_image
        if dark is not None:
            read_rms = float(np.median(dark.std_image))
        else:
            read_rms = ccd_config.get().read_noise_counts

        truth = self._scan_mean_frame()
        is_ref = self.axial_scan.system_state.is_reference_mode

        log.info(f"[MC] {n_frames} synthetic frames: truth = scan-mean frame "
                 f"({len(self.axial_scan.measurements)} frames averaged), "
                 f"gain {e_per_count:.3f} e-/count, read noise "
                 f"{read_rms:.2f} counts/px")

        mc = MonteCarloFrames(mean_frame=truth,
                              gain_e_per_count=e_per_count,
                              read_noise_counts=read_rms,
                              n_images=n_frames, seed=0)

        samples = []
        n_failed = 0
        measured_bg = None
        for i, noisy in enumerate(mc.frames()):
            try:
                px, sline = self.fitter.get_px_sline_from_image(noisy)
                if measured_bg is None:
                    measured_bg = self._measured_background_for_fit(px)
                fit = self.fitter.fit(px=px, sline=sline,
                                      is_reference_mode=is_ref,
                                      measured_background=measured_bg)
            except Exception:
                n_failed += 1
                continue
            if not fit.is_success:
                n_failed += 1
                continue
            f = reference_freq(self.calc.analyze(fit), self.peak_reference)
            if f is not None and np.isfinite(f):
                samples.append(f)
            if (i + 1) % 50 == 0:
                log.info(f"[MC] {i + 1}/{n_frames} frames fitted")

        if n_failed:
            log.info(f"[MC] {n_failed} synthetic fits failed/discarded")
        return np.asarray(samples, dtype=float)

    def on_analyze_snr(self):
        """Measured scatter next to the Thompson bound and a Monte Carlo."""
        try:
            log.info("==== Analyze SNR ====")

            freq_shifts = np.array(
                [fs for fs in self.freq_shifts if fs is not None], dtype=float)
            freq_shifts = freq_shifts[np.isfinite(freq_shifts)]
            if freq_shifts.size < 2:
                QMessageBox.warning(self, "No Data",
                                    "Not enough valid frequency shifts to analyze.")
                return

            mu = float(np.mean(freq_shifts))
            sigma = float(np.std(freq_shifts, ddof=1))
            # Drift-immune measured scatter: consecutive-frame differences.
            # This is THE measured number the bound is compared against.
            sigma_diff = float(np.std(np.diff(freq_shifts), ddof=1) / np.sqrt(2.0))
            n = len(freq_shifts)

            theo = self._scan_mean_thompson()
            sigma_thompson_mhz = (
                reference_theo_total_mhz(theo, self.peak_reference)
                if theo is not None else None)

            mc_samples = self._run_monte_carlo(self.mc_spinner.value())
            sigma_mc_mhz = (float(np.std(mc_samples, ddof=1)) * 1000.0
                            if mc_samples.size >= 2 else None)

            log.info(f"Reference: {self.peak_reference}")
            log.info(f"Measured:  mean {mu:.4f} GHz (n={n})")
            log.info(f"  plain sd          {sigma * 1000:.2f} MHz "
                     f"(includes drift)")
            log.info(f"  diff-sd / sqrt(2) {sigma_diff * 1000:.2f} MHz "
                     f"(drift-immune — compare THIS to the bound)")
            if sigma_thompson_mhz is not None:
                log.info(f"Thompson bound (scan-mean parameters): "
                         f"{sigma_thompson_mhz:.2f} MHz")
                if sigma_thompson_mhz > 0:
                    log.info(f"  diff-sd / Thompson: "
                             f"{sigma_diff * 1000 / sigma_thompson_mhz:.2f}x "
                             f"(pipeline runs ~1.3-1.5x above the bound)")
            if sigma_mc_mhz is not None:
                log.info(f"Monte Carlo sd ({mc_samples.size} fits): "
                         f"{sigma_mc_mhz:.2f} MHz")
            log.info("=====================")

            # --- Figure ---
            fig = Figure(figsize=(7, 5), constrained_layout=True)
            ax = fig.add_subplot(111)
            counts, bins, _ = ax.hist(
                freq_shifts, bins=15, color="#9ecae1", edgecolor="black",
                alpha=0.7,
                label=f"Measured (n={n})\nσ = {sigma * 1000:.2f} MHz, "
                      f"diff-σ = {sigma_diff * 1000:.2f} MHz")
            bin_width = bins[1] - bins[0]

            if sigma > 0:
                x = np.linspace(freq_shifts.min(), freq_shifts.max(), 300)
                ax.plot(x, norm.pdf(x, mu, sigma) * n * bin_width, "-",
                        color="#1f77b4", lw=2, label="Gaussian fit")

            if mc_samples.size >= 2:
                # Widths are the comparison — recenter the MC (its truth is
                # one measurement, not the scan mean) onto the measured mean.
                mc_centered = mc_samples - float(np.mean(mc_samples)) + mu
                weights = np.full(mc_samples.size,
                                  n / mc_samples.size)
                ax.hist(mc_centered, bins=bins, weights=weights,
                        histtype="step", lw=2, color="#2ca02c",
                        label=f"Monte Carlo (recentered)\n"
                              f"σ = {sigma_mc_mhz:.2f} MHz")

            if sigma_thompson_mhz is not None and sigma_thompson_mhz > 0:
                s_t = sigma_thompson_mhz / 1000.0
                x = np.linspace(mu - 4 * max(sigma, s_t),
                                mu + 4 * max(sigma, s_t), 300)
                ax.plot(x, norm.pdf(x, mu, s_t) * n * bin_width, "--",
                        color="#d62728", lw=2,
                        label=f"Thompson shot-noise limit\n"
                              f"σ = {sigma_thompson_mhz:.2f} MHz")

            ax.set_xlabel(f"Frequency shift, {self.peak_reference} (GHz)")
            ax.set_ylabel("Count")
            ax.set_title("Measured scatter vs shot-noise limit vs Monte Carlo")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            hist_window = QWidget()
            hist_window.setWindowTitle("Analyze SNR - Frequency Shift Histogram")
            layout = QVBoxLayout(hist_window)
            canvas = FigureCanvas(fig)
            layout.addWidget(NavigationToolbar(canvas, hist_window))
            layout.addWidget(canvas)
            hist_window.resize(700, 500)
            hist_window.show()
            self.open_windows.append(hist_window)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to analyze SNR:\n{e}")

    # ---------------- Printing ----------------

    def print_scan_overview(self):
        scan = self.axial_scan
        log.info("==== Axial Scan Overview ====")
        log.info(f"ID: {scan.id}, Internal tracker i: {scan.i}")
        log.info(f"Number of measurements: {len(scan.measurements)}")

        self.calc.print_all_models()
        ss = scan.system_state
        log.info(f"System State: reference_mode={ss.is_reference_mode}, "
                 f"exposure={ss.andor_camera_info.exposure}, "
                 f"emccd_gain={ss.andor_camera_info.gain}, "
                 f"preamp_gain={ss.andor_camera_info.preamp_gain}")
        used = self.fitter.get_selected_rows(
            np.asarray(scan.measurements[0].frame_andor))
        log.info(f"Analysis sline rows (live config): "
                 f"{used[0]}-{used[-1]} ({len(used)} rows)")
        if scan.reflection_result_forwards is not None:
            z = scan.reflection_result_forwards.event_z_um
            log.info(f"Plane (forwards): {round(z) if z is not None else None}")
        if scan.reflection_result_backwards is not None:
            z = scan.reflection_result_backwards.event_z_um
            log.info(f"Plane (backwards): {round(z) if z is not None else None}")

        if scan.eye_tracker_results is not None:
            if scan.eye_tracker_results.laser_position is not None:
                er = scan.eye_tracker_results
                log.info("Laser Position [mm]: "
                         f"X={fmt(er.laser_position[0], 2)}, "
                         f"Y={fmt(er.laser_position[1], 2)}, "
                         f"Z={fmt(er.laser_position[2], 2)}")
            else:
                log.info("Eye Tracker Position is None")
        log.info("=============================")

    def print_measurement_info(self):
        mp = self.axial_scan.measurements[self.current_index]
        analyzed = self.list_analyzed_spectras[self.current_index]
        freq_shift = analyzed.analyzed_shifts
        photons = analyzed.photons
        theo = analyzed.theoretical_precisions

        log.info(f"--- Measurement {self.current_index} ---")
        log.info(f"Zaber position: {fmt(mp.lens_zaber_position, precision=0)} µm")
        log.info(f"Freq shifts: left={fmt(freq_shift.freq_shift_left_peak_ghz)}, "
                 f"right={fmt(freq_shift.freq_shift_right_peak_ghz)}, "
                 f"distance={fmt(freq_shift.freq_shift_peak_distance_ghz)}")
        log.info(f"HWHM fitted (GHz): left={fmt(freq_shift.hwhm_left_peak_ghz)}, "
                 f"right={fmt(freq_shift.hwhm_right_peak_ghz)}")
        log.info(f"HWHM instrument (GHz): left={fmt(freq_shift.instrument_hwhm_left_peak_ghz)}, "
                 f"right={fmt(freq_shift.instrument_hwhm_right_peak_ghz)}")
        log.info(f"Linewidth sample (GHz): left={fmt(freq_shift.linewidth_left_peak_ghz)}, "
                 f"right={fmt(freq_shift.linewidth_right_peak_ghz)}")
        log.info(f"Photons: left={fmt(photons.left_peak_photons, precision=0)}, "
                 f"right={fmt(photons.right_peak_photons, precision=0)}, "
                 f"total={fmt(photons.total_photons, precision=0)}")
        log.info(f"Thompson total (MHz): left={fmt(theo.left_peak_total_mhz, 1)}, "
                 f"right={fmt(theo.right_peak_total_mhz, 1)}, "
                 f"distance={fmt(theo.distance_total_mhz, 1)}")
        log.info("----------------------------")

    # ---------------- Calibration plot ----------------

    def on_show_calibration_plot(self):
        try:
            fit: FittedSpectrum = self.list_analyzed_spectras[
                self.current_index].fitted_spectrum

            viewer = CalibrationViewer(
                self.calc,
                title=f"Calibration - Scan {self.axial_scan.id}",
                calibration_data=self.axial_scan.calibration_data,
                fitter=self.fitter,
                current_fit=fit)
            viewer.show()
            self.open_windows.append(viewer)

        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(self, "Error",
                                 f"Failed to show calibration plot:\n{e}")

    # --- Export to Excel ---

    def _build_export_rows(self) -> list[BrillouinExport]:
        return [
            get_excel_row_data(axial_scan=self.axial_scan,
                               analyzed_spectrum=self.list_analyzed_spectras[i],
                               idx=i)
            for i in range(len(self.list_analyzed_spectras))
        ]

    def _get_default_excel_name(self) -> str:
        return f"{self.axial_scan.id}_brillouin_export.xlsx"

    @staticmethod
    def _write_rows_to_excel(file_path: str, rows: list[BrillouinExport]) -> str:
        if not file_path.lower().endswith(".xlsx"):
            file_path += ".xlsx"
        export_to_excel(rows, file_path)
        return file_path

    def save_new_excel(self):
        try:
            rows = self._build_export_rows()
            if not rows:
                QMessageBox.warning(self, "No Data", "There is no data to export.")
                return

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save New Excel", self._get_default_excel_name(),
                "Excel Files (*.xlsx)")
            if not file_path:
                return

            file_path = self._write_rows_to_excel(file_path, rows)
            log.info(f"Saved {len(rows)} rows to {file_path}")
            QMessageBox.information(self, "Excel Saved",
                                    f"Saved {len(rows)} rows to new file:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed",
                                 f"Could not save Excel file:\n{e}")

    def add_to_existing_excel(self):
        try:
            new_rows = self._build_export_rows()
            if not new_rows:
                QMessageBox.warning(self, "No Data", "There is no data to export.")
                return

            file_path, _ = QFileDialog.getOpenFileName(
                self, "Add to Existing Excel", "", "Excel Files (*.xlsx)")
            if not file_path:
                return

            existing_rows = load_from_excel(file_path, sheet_name=0)
            combined_rows = existing_rows + new_rows
            self._write_rows_to_excel(file_path, combined_rows)
            log.info(f"Added {len(new_rows)} rows to {file_path} "
                     f"({len(combined_rows)} total)")
            QMessageBox.information(
                self, "Excel Updated",
                f"Added {len(new_rows)} rows.\n"
                f"Workbook now contains {len(combined_rows)} rows:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Append Failed",
                                 f"Could not update Excel file:\n{e}")

    def export_current_frame(self):
        try:
            frame = self._raw_frame(self.current_index)

            if frame.max() > frame.min():
                normed = (frame - frame.min()) / (frame.max() - frame.min())
            else:
                normed = np.zeros_like(frame)
            frame_uint8 = (normed * 255).astype(np.uint8)

            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Frame", f"frame_{self.current_index}.tiff",
                "TIFF (*.tiff *.tif);;PNG (*.png)")
            if not file_path:
                return

            if file_path.lower().endswith(".png"):
                from PIL import Image
                Image.fromarray(frame_uint8).save(file_path)
            else:
                if not file_path.lower().endswith((".tif", ".tiff")):
                    file_path += ".tiff"
                imwrite(file_path, frame_uint8)

            log.info(f"Frame {self.current_index} exported to {file_path}")
            QMessageBox.information(self, "Saved", f"Saved to:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed:\n{e}")
