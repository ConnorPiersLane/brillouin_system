from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QApplication, QMessageBox, QCheckBox,
    QFileDialog,
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from brillouin_system.spectrum_fitting.peak_fitting_config.psf_measurement import (
    PSF_MEASURED,
)
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    find_peaks_sample_config, find_peaks_reference_config, sline_from_frame_config,
    save_config_section, FIND_PEAKS_TOML_PATH,
    FITTING_MODELS_SAMPLE, FITTING_MODELS_REFERENCE, BACKGROUNDS,
    NA_WEIGHTINGS, ROW_SELECTIONS, FittingConfigs,
    DHO_KERNELS, CENTRE_METHODS, ENVELOPE_SOURCES, KERNEL_SOURCES,
    KERNEL_FILE_LINES,
)


class FindPeaksConfigDialog(QDialog):
    def __init__(self, on_apply=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find Peaks Configuration")
        # self.setMinimumSize(500, 500)

        self.sample_inputs = {}
        self.reference_inputs = {}
        self.global_inputs = {}
        self.on_apply = on_apply

        layout = QVBoxLayout()
        layout.addLayout(self.create_dual_form())
        layout.addLayout(self.create_global_inputs())
        layout.addLayout(self.create_buttons())
        self.setLayout(layout)

        self.load_values()

        # The parametric camera-PSF constants are read only by the legacy
        # paths; grey them out whenever the self-calibrating chain is on.
        for combo in (self.sample_inputs["fitting_model"],
                      self.sample_inputs["dho_kernel"],
                      self.reference_inputs["centre_method"]):
            combo.currentTextChanged.connect(
                lambda _text: self._update_psf_fields_enabled())
        self._update_psf_fields_enabled()

    def field_names(self):
        # beta is not here: it renders indented under the use_window checkbox
        # (it is a parameter of the windowing).
        return [
            "prominence_fraction", "min_peak_width", "min_peak_height",
            "rel_height", "wlen_pixels",
        ]

    def pr_field_names(self):
        # 'lorentzian_x_psf' model: camera PSF working values — Gaussian
        # charge diffusion and the one-sided readout tail, both per peak
        # (position properties on the sensor). Part of the [global] fitting
        # config (one camera, one kernel, shared by sample and reference
        # fits). Not fitted per frame; the MEASURED record is
        # psf_measurement.PSF_MEASURED (shown in brackets).
        return ["psf_sigma_left_px", "psf_sigma_right_px",
                "psf_tau_left_px", "psf_tau_right_px",
                "psf_sigma_outer_left_px", "psf_sigma_outer_right_px",
                "psf_tau_outer_left_px", "psf_tau_outer_right_px",
                "psf_box_outer_left_px", "psf_box_outer_right_px",
                "psf_sat_ratio_outer_right", "psf_sat_delta_outer_right_px"]

    def na_field_names(self):
        # NA collection model (post-hoc scalar correction only, never in the
        # fit), sample only: aperture-clip NA; Gaussian coupling
        # geometry (na_weighting = uniform_gaussian: fiber-mode beam diameter
        # at pupil [session-calibrated on water] + objective focal length);
        # sample refractive index. The weighting itself is a combo box, not a
        # float field — see create_config_group.
        return ["na_collection", "na_beam_diameter_mm", "na_focal_length_mm", "na_n_sample"]

    def create_dual_form(self):
        layout = QHBoxLayout()
        layout.addWidget(self.create_config_group(
            "Sample", self.sample_inputs, FITTING_MODELS_SAMPLE, extra_fields=self.na_field_names()))
        layout.addWidget(self.create_config_group(
            "Reference", self.reference_inputs, FITTING_MODELS_REFERENCE))
        return layout

    def create_config_group(self, label, inputs, models, extra_fields=()):
        group = QGroupBox(label)
        vlayout = QVBoxLayout()
        for field in self.field_names():
            row = QHBoxLayout()
            row.addWidget(QLabel(field.replace("_", " ").capitalize()))
            edit = QLineEdit()
            if self._is_float_field(field):
                # floats (>=0), allow up to 100.0 with 5 decimal precision
                edit.setValidator(QDoubleValidator(0.0, 100.0, 5))
            else:
                edit.setValidator(QIntValidator(0, 9999))
            inputs[field] = edit
            row.addWidget(edit)
            vlayout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Fitting Model"))
        combo = QComboBox()
        combo.addItems(models)
        inputs["fitting_model"] = combo
        row.addWidget(combo)
        vlayout.addLayout(row)

        if label == "Sample":
            # instrument kernel of the DHO model
            row = QHBoxLayout()
            row.addSpacing(24)
            row.addWidget(QLabel("DHO kernel"))
            k_combo = QComboBox()
            k_combo.addItems(DHO_KERNELS)
            k_combo.setToolTip(
                "Instrument kernel the 'dho_x_psf' model convolves the DHO "
                "core with:\n"
                "measured: the profile stacked from THIS scan's calibration "
                "frames at each peak's position (per frame, no constants) — "
                "needs Centre method = template on the reference side.\n"
                "parametric: Lorentzian x Gauss x tail x pixel from the "
                "calibration width polynomial and the camera PSF constants "
                "below (legacy)."
            )
            inputs["dho_kernel"] = k_combo
            row.addWidget(k_combo)
            vlayout.addLayout(row)
        else:
            # how the calibration line centres are measured
            row = QHBoxLayout()
            row.addSpacing(24)
            row.addWidget(QLabel("Centre method"))
            c_combo = QComboBox()
            c_combo.addItems(CENTRE_METHODS)
            c_combo.setToolTip(
                "How the calibration lines are located (the frequency axis):\n"
                "template: plain-Lorentzian first guess, centres smoothed in "
                "drive frequency, measured profile stacked per position and "
                "refitted as a template (no instrument model; also supplies "
                "the DHO kernels and the envelope) — production.\n"
                "parametric: every calibration frame fitted with the "
                "Lorentzian x Gauss x tail x pixel model using the camera "
                "PSF constants below (legacy)."
            )
            inputs["centre_method"] = c_combo
            row.addWidget(c_combo)
            vlayout.addLayout(row)

        # NA correction (sample group only): the weighting selects the model,
        # the indented fields below are its parameters.
        if "na_collection" in extra_fields:
            row = QHBoxLayout()
            row.addWidget(QLabel("NA weighting"))
            na_combo = QComboBox()
            na_combo.addItems(NA_WEIGHTINGS)
            na_combo.setToolTip(
                "Collection weight over the NA cone (post-hoc scalar "
                "correction):\n"
                "none: no NA correction (ratio = 1); the fields below are "
                "ignored.\n"
                "uniform: hard pupil only — the NA 0.14 recipe (~ +3.5 MHz on "
                "water, parameter-free).\n"
                "uniform_gaussian: adds the Gaussian fiber-coupling apodization "
                "from na_beam_diameter_mm / na_focal_length_mm — required at "
                "NA 0.42."
            )
            inputs["na_weighting"] = na_combo
            row.addWidget(na_combo)
            vlayout.addLayout(row)

            for field in extra_fields:
                row = QHBoxLayout()
                row.addSpacing(24)  # parameters of the weighting above
                row.addWidget(QLabel(field.replace("_", " ").capitalize()))
                edit = QLineEdit()
                edit.setValidator(QDoubleValidator(0.0, 100.0, 5))
                inputs[field] = edit
                row.addWidget(edit)
                vlayout.addLayout(row)

            na_combo.currentTextChanged.connect(
                lambda _text, i=inputs: self._update_na_fields_enabled(i))
            self._update_na_fields_enabled(inputs)

        # Windowing and baseline apply to any lineshape.
        row = QHBoxLayout()
        row.addWidget(QLabel("Background"))
        bg_combo = QComboBox()
        bg_combo.addItems(BACKGROUNDS)
        inputs["background"] = bg_combo
        row.addWidget(bg_combo)
        vlayout.addLayout(row)

        check = QCheckBox("Fit only within +-beta*width of each peak")
        inputs["use_window"] = check
        vlayout.addWidget(check)

        row = QHBoxLayout()
        row.addSpacing(24)  # parameter of the windowing above
        row.addWidget(QLabel("Beta"))
        beta_edit = QLineEdit()
        beta_edit.setValidator(QDoubleValidator(0.0, 100.0, 5))
        beta_edit.setToolTip(
            "Window half-width in units of the found peak width. The prm "
            "presets pin beta = 3.0 (the width recipe is only valid there)."
        )
        inputs["beta"] = beta_edit
        row.addWidget(beta_edit)
        vlayout.addLayout(row)

        check.toggled.connect(beta_edit.setEnabled)
        beta_edit.setEnabled(check.isChecked())

        group.setLayout(vlayout)
        return group

    def create_global_inputs(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Global Settings"))

        # How the summed row band is chosen
        row = QHBoxLayout()
        row.addWidget(QLabel("Row selection"))
        combo = QComboBox()
        combo.addItems(ROW_SELECTIONS)
        combo.setToolTip(
            "manual: use the row list below.\n"
            "auto: take 'N rows' centred on the line's intensity centroid, "
            "located once per scan and then frozen."
        )
        self.global_inputs["row_selection"] = combo
        row.addWidget(combo)
        layout.addLayout(row)

        # Number of rows for the automatic band
        row = QHBoxLayout()
        row.addWidget(QLabel("N rows (auto)"))
        edit = QLineEdit()
        edit.setValidator(QIntValidator(1, 9999))
        edit.setToolTip(
            "Rows summed when row selection is 'auto'. 13 captures ~97% of "
            "the signal; precision plateaus from about 11."
        )
        self.global_inputs["n_rows"] = edit
        row.addWidget(edit)
        layout.addLayout(row)

        # Selected Rows (comma-separated), used when row selection is manual
        row = QHBoxLayout()
        row.addWidget(QLabel("Selected Rows (manual)"))
        edit = QLineEdit()
        edit.setPlaceholderText("e.g. 2, 3, 4, 5")
        self.global_inputs["selected_rows"] = edit
        row.addWidget(edit)
        layout.addLayout(row)

        # How many VIPA orders to fit — GLOBAL: one ROI, one peak count,
        # shared by the sample and reference fits.
        row = QHBoxLayout()
        row.addWidget(QLabel("N peaks (VIPA orders)"))
        n_peaks_combo = QComboBox()
        n_peaks_combo.addItems(["2", "4"])
        n_peaks_combo.setToolTip(
            "2: the inner main pair only.\n"
            "4: all four VIPA orders jointly — each order gets its own "
            "calibration track and the analysis reports the per-order "
            "shifts plus their inverse-variance combination. Requires an "
            "ROI containing the outer orders; on two-peak data the "
            "calibration stops with an error and sample fits fail loudly."
        )
        self.global_inputs["n_peaks"] = n_peaks_combo
        row.addWidget(n_peaks_combo)
        layout.addLayout(row)

        # Pixel Offsets
        for key in ["pixel_offset_left", "pixel_offset_right"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(key.replace("_", " ").capitalize()))
            edit = QLineEdit()
            edit.setValidator(QIntValidator(0, 9999))
            self.global_inputs[key] = edit
            row.addWidget(edit)
            layout.addLayout(row)

        # VIPA envelope slopes: per scan from the calibration, or constants
        row = QHBoxLayout()
        row.addWidget(QLabel("Envelope source"))
        env_combo = QComboBox()
        env_combo.addItems(ENVELOPE_SOURCES)
        env_combo.setToolTip(
            "Where the VIPA envelope slopes applied to each peak come from:\n"
            "measured: from this scan's own calibration frames (same-sideband "
            "line-area ratios along the sweep, degree-4 ln envelope; needs "
            "the four-order ROI, falls back to the constants otherwise).\n"
            "config: the four env_slope_*_perpx constants in the TOML."
        )
        self.global_inputs["envelope_source"] = env_combo
        row.addWidget(env_combo)
        layout.addLayout(row)

        # DHO sample kernels: the scan's own node table, or a stored one
        # (fine sweep) for the outer orders / all lines
        row = QHBoxLayout()
        row.addWidget(QLabel("Kernel source"))
        k_src = QComboBox()
        k_src.addItems(KERNEL_SOURCES)
        k_src.setToolTip(
            "Where the measured DHO sample kernels come from (dho_kernel = "
            "measured, centre_method = template). "
            "scan: this scan's own 41-point calibration node table. "
            "file: the stored node table below (built on a 401-point fine "
            "sweep with Epsf.save) for the lines chosen; the frequency axis "
            "and the envelope slopes stay per scan."
        )
        self.global_inputs["kernel_source"] = k_src
        row.addWidget(k_src)
        row.addWidget(QLabel("for"))
        k_lines = QComboBox()
        k_lines.addItems(KERNEL_FILE_LINES)
        k_lines.setToolTip("outer: outer_left + outer_right from the file, inner "
                           "pair per scan. all: every fitted line from the file.")
        self.global_inputs["kernel_file_lines"] = k_lines
        row.addWidget(k_lines)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Kernel file"))
        k_file = QLineEdit()
        k_file.setToolTip("Path of the stored ePSF node table (CSV written by "
                          "Epsf.save / template_calibration.save_epsf_file).")
        self.global_inputs["kernel_file"] = k_file
        row.addWidget(k_file)
        load_btn = QPushButton("Load PSF...")
        load_btn.setToolTip("Pick a stored ePSF node table (CSV) and use it as the "
                            "kernel file (kernel source -> file).")
        load_btn.clicked.connect(self._pick_kernel_file)
        row.addWidget(load_btn)
        build_btn = QPushButton("Compile from sweep...")
        build_btn.setToolTip("Pick a calibration fine sweep (.h5, e.g. 401 points 4-8 "
                             "GHz), build its ePSF node table under the CURRENT "
                             "fitting config, store it as epsf_<name>.csv next to "
                             "the .h5 and use it as the kernel file.")
        build_btn.clicked.connect(self._compile_kernel_file)
        row.addWidget(build_btn)
        layout.addLayout(row)
        self._kernel_info = QLabel("")
        self._kernel_info.setWordWrap(True)
        layout.addWidget(self._kernel_info)
        k_src.currentTextChanged.connect(lambda _t: self._update_kernel_file_enabled())

        # Camera PSF working values — part of the [global] fitting config,
        # shared by the sample and reference fits. The label shows the
        # MEASURED value from psf_measurement.PSF_MEASURED in brackets: that
        # record is never touched by the GUI, so the measurement cannot be
        # lost by experimentation here. Read ONLY by the legacy parametric
        # paths (greyed out while the self-calibrating chain is selected).
        self._psf_label = QLabel("Camera PSF constants (legacy parametric kernel only)")
        layout.addWidget(self._psf_label)
        measured = PSF_MEASURED
        for key in self.pr_field_names():
            row = QHBoxLayout()
            ref = getattr(measured, key, None)
            label = key.replace("_", " ").capitalize()
            if ref is not None:
                label += f"  (measured: {ref:g})"
            row.addWidget(QLabel(label))
            edit = QLineEdit()
            edit.setValidator(QDoubleValidator(0.0, 100.0, 5))
            edit.setToolTip(
                "Camera constants for the 'lorentzian_x_psf' model — "
                "Gaussian charge-diffusion blur and the one-sided readout "
                "tails. Not fitted per frame; saved with the [global] "
                "fitting config. The bracketed value is the MEASURED record "
                "(psf_measurement.py, fine EOM sweeps; see "
                "measure_psf_kernel.py) — the GUI never writes it."
            )
            self.global_inputs[key] = edit
            row.addWidget(edit)
            layout.addLayout(row)

        return layout

    def load_values(self):
        sample = find_peaks_sample_config.get()
        reference = find_peaks_reference_config.get()
        global_cfg = sline_from_frame_config.get()

        for field in self.field_names():
            self.sample_inputs[field].setText(str(getattr(sample, field)))
            self.reference_inputs[field].setText(str(getattr(reference, field)))

        for field in self.na_field_names():
            self.sample_inputs[field].setText(str(getattr(sample, field)))
        self.sample_inputs["na_weighting"].setCurrentText(sample.na_weighting)

        for field in self.pr_field_names():
            self.global_inputs[field].setText(str(getattr(global_cfg, field)))

        for inputs, cfg in ((self.sample_inputs, sample),
                            (self.reference_inputs, reference)):
            inputs["fitting_model"].setCurrentText(cfg.fitting_model)
            inputs["background"].setCurrentText(cfg.background)
            inputs["use_window"].setChecked(bool(cfg.use_window))
            inputs["beta"].setText(str(cfg.beta))
        self.sample_inputs["dho_kernel"].setCurrentText(sample.dho_kernel)
        self.reference_inputs["centre_method"].setCurrentText(reference.centre_method)
        self.global_inputs["envelope_source"].setCurrentText(global_cfg.envelope_source)
        self.global_inputs["kernel_source"].setCurrentText(global_cfg.kernel_source)
        self.global_inputs["kernel_file_lines"].setCurrentText(global_cfg.kernel_file_lines)
        self.global_inputs["kernel_file"].setText(str(global_cfg.kernel_file))
        self._update_kernel_file_enabled()

        # Global settings
        self.global_inputs["pixel_offset_left"].setText(str(global_cfg.pixel_offset_left))
        self.global_inputs["pixel_offset_right"].setText(str(global_cfg.pixel_offset_right))
        self.global_inputs["selected_rows"].setText(", ".join(str(x) for x in global_cfg.selected_rows))
        self.global_inputs["row_selection"].setCurrentText(global_cfg.row_selection)
        self.global_inputs["n_rows"].setText(str(global_cfg.n_rows))
        self.global_inputs["n_peaks"].setCurrentText(str(global_cfg.n_peaks))

    def create_buttons(self):
        layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_config)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_config)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addStretch()
        layout.addWidget(apply_btn)
        layout.addWidget(save_btn)
        layout.addWidget(close_btn)
        return layout

    def apply_config(self):
        try:
            # Global config
            global_kwargs = {
                "pixel_offset_left": self._parse(self.global_inputs["pixel_offset_left"].text(), "int"),
                "pixel_offset_right": self._parse(self.global_inputs["pixel_offset_right"].text(), "int"),
                "selected_rows": self._parse_selected_rows(self.global_inputs["selected_rows"].text()),
                "row_selection": self.global_inputs["row_selection"].currentText(),
                "n_rows": max(self._parse(self.global_inputs["n_rows"].text(), "int"), 1),
                "n_peaks": int(self.global_inputs["n_peaks"].currentText()),
                "envelope_source": self.global_inputs["envelope_source"].currentText(),
                "kernel_source": self.global_inputs["kernel_source"].currentText(),
                "kernel_file_lines": self.global_inputs["kernel_file_lines"].currentText(),
                "kernel_file": self.global_inputs["kernel_file"].text().strip(),
            }
            # Camera PSF working values ride in the same [global] config.
            global_kwargs.update({f: self._parse(self.global_inputs[f].text(), f)
                                  for f in self.pr_field_names()})

            # Sample
            sample_kwargs = {f: self._parse(self.sample_inputs[f].text(), f)
                             for f in (list(self.field_names()) + ["beta"]
                                       + list(self.na_field_names()))}
            sample_kwargs["na_weighting"] = self.sample_inputs["na_weighting"].currentText()
            sample_kwargs["dho_kernel"] = self.sample_inputs["dho_kernel"].currentText()
            sample_kwargs.update(self._model_kwargs(self.sample_inputs))

            # Reference
            reference_kwargs = {f: self._parse(self.reference_inputs[f].text(), f)
                                for f in list(self.field_names()) + ["beta"]}
            reference_kwargs["centre_method"] = self.reference_inputs["centre_method"].currentText()
            reference_kwargs.update(self._model_kwargs(self.reference_inputs))

            # Update all configs
            find_peaks_sample_config.update(**sample_kwargs)
            find_peaks_reference_config.update(**reference_kwargs)
            sline_from_frame_config.update(**global_kwargs)

            if self.on_apply:
                fitting_configs = FittingConfigs(
                    sline_config=sline_from_frame_config.get(),
                    sample_config=find_peaks_sample_config.get(),
                    reference_config=find_peaks_reference_config.get(),
                )
                self.on_apply(fitting_configs)

            QMessageBox.information(self, "Applied", "Settings applied (not saved to disk).")

        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply config:\n{e}")

    def save_config(self):
        try:
            self.apply_config()
            save_config_section(FIND_PEAKS_TOML_PATH, "sample", find_peaks_sample_config)
            save_config_section(FIND_PEAKS_TOML_PATH, "reference", find_peaks_reference_config)
            save_config_section(FIND_PEAKS_TOML_PATH, "global", sline_from_frame_config)
            QMessageBox.information(self, "Saved", "Settings saved to disk.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save config:\n{e}")

    def _update_na_fields_enabled(self, inputs):
        """Grey out the NA parameters the selected weighting does not read:
        none -> all off; uniform -> aperture + index only (na_collection is
        then the EFFECTIVE NA); uniform_gaussian -> all."""
        weighting = inputs["na_weighting"].currentText()
        gauss_only = ("na_beam_diameter_mm", "na_focal_length_mm")
        for field in self.na_field_names():
            if weighting == "none":
                enabled = False
            elif weighting == "uniform":
                enabled = field not in gauss_only
            else:
                enabled = True
            inputs[field].setEnabled(enabled)

    def _update_kernel_file_enabled(self):
        on = self.global_inputs["kernel_source"].currentText() == "file"
        self.global_inputs["kernel_file"].setEnabled(on)
        self.global_inputs["kernel_file_lines"].setEnabled(on)

    def set_kernel_file(self, path: str):
        """Use the stored ePSF table at `path`: read it (a bad file raises
        before anything changes), fill the field, switch the kernel source
        to 'file' and show what the table holds. Apply/Save as usual."""
        from brillouin_system.spectrum_fitting.epsf import load_epsf_file
        e = load_epsf_file(path)
        self.global_inputs["kernel_file"].setText(str(path))
        self.global_inputs["kernel_source"].setCurrentText("file")
        lines = ", ".join(f"{nm} {e.nodes[i].min():.0f}-{e.nodes[i].max():.0f} px "
                          f"({len(e.nodes[i])} nodes)" for i, nm in enumerate(e.names))
        self._kernel_info.setText(f"PSF table: {lines}; source {e.source or '-'}")
        return e

    def _pick_kernel_file(self):
        start = str(Path(self.global_inputs["kernel_file"].text() or ".").parent)
        path, _ = QFileDialog.getOpenFileName(self, "Stored ePSF node table", start,
                                              "ePSF table (*.csv);;All files (*)")
        if not path:
            return
        try:
            self.set_kernel_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Load PSF", f"Could not read {path}: {e}")

    def compile_kernel_file(self, calibration_h5: str) -> str:
        """Build the ePSF table of a calibration fine sweep under the CURRENT
        (applied) fitting config, store it as epsf_<stem>.csv next to the
        .h5 and use it. Returns the table's path."""
        from brillouin_system.spectrum_fitting.template_calibration import save_epsf_file
        out = Path(calibration_h5).with_name(f"epsf_{Path(calibration_h5).stem}.csv")
        save_epsf_file(calibration_h5, out)
        self.set_kernel_file(str(out))
        return str(out)

    def _compile_kernel_file(self):
        start = str(Path(self.global_inputs["kernel_file"].text() or ".").parent)
        path, _ = QFileDialog.getOpenFileName(self, "Calibration fine sweep", start,
                                              "calibration (*.h5);;All files (*)")
        if not path:
            return
        try:
            out = self.compile_kernel_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Compile PSF",
                                 f"Could not build the table from {path}: {e}")
            return
        QMessageBox.information(self, "Compile PSF", f"Stored {out} and selected it as the kernel file.")

    def psf_constants_in_use(self) -> bool:
        """True when some selected path still reads the parametric camera
        PSF constants: a parametric calibration, a parametric DHO kernel, or
        the 'lorentzian_x_psf' sample model."""
        return (self.reference_inputs["centre_method"].currentText() == "parametric"
                or self.sample_inputs["dho_kernel"].currentText() == "parametric"
                or self.sample_inputs["fitting_model"].currentText() == "lorentzian_x_psf")

    def _update_psf_fields_enabled(self):
        on = self.psf_constants_in_use()
        for key in self.pr_field_names():
            self.global_inputs[key].setEnabled(on)
        self._psf_label.setText(
            "Camera PSF constants (legacy parametric kernel"
            + (")" if on else " — not used by the selected chain)"))

    @staticmethod
    def _model_kwargs(inputs):
        """Lineshape + the options that apply to any lineshape."""
        return {
            "fitting_model": inputs["fitting_model"].currentText(),
            "background": inputs["background"].currentText(),
            "use_window": inputs["use_window"].isChecked(),
        }

    @staticmethod
    def _is_float_field(field):
        return (
            "fraction" in field or "rel" in field
            or field == "beta" or field.startswith("na_")
            or field.startswith("psf_")
        )

    def _parse(self, value, field):
        # Raise instead of silently coercing: a typo (or an empty field)
        # must surface in the Apply/Save error dialog, never become 0.
        value = value.strip()
        kind = "number" if self._is_float_field(field) else "integer"
        try:
            return float(value) if self._is_float_field(field) else int(value)
        except ValueError:
            raise ValueError(f"'{field}' must be a {kind}, got '{value}'.")

    def _parse_selected_rows(self, text):
        # Empty is fine (auto row selection); a malformed token is not.
        rows = []
        for token in text.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit():
                raise ValueError(
                    f"Selected rows must be comma-separated integers, "
                    f"got '{token}'."
                )
            rows.append(int(token))
        return rows


# ---------- Example usage ----------
if __name__ == "__main__":
    import sys

    def on_apply(configs: FittingConfigs):
        print("[Sline]", configs.sline_config)
        print("[Sample]", configs.sample_config)
        print("[Reference]", configs.reference_config)


    app = QApplication(sys.argv)
    dlg = FindPeaksConfigDialog(on_apply=on_apply)
    dlg.show()
    sys.exit(app.exec_())
