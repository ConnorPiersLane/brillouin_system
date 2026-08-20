from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QApplication, QMessageBox, QCheckBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    find_peaks_sample_config, find_peaks_reference_config, sline_from_frame_config,
    pixel_response_config,
    save_config_section, FIND_PEAKS_TOML_PATH,
    FITTING_MODELS_SAMPLE, FITTING_MODELS_REFERENCE, BACKGROUNDS,
    NA_WEIGHTINGS, ROW_SELECTIONS, FittingConfigs
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

    def field_names(self):
        return [
            "prominence_fraction", "min_peak_width", "min_peak_height",
            "rel_height", "wlen_pixels", "beta"  # added beta
        ]

    def pr_field_names(self):
        # 'pixel_response' model: frozen camera pixel-response constants —
        # Gaussian charge diffusion and the one-sided readout tail per peak.
        # GLOBAL (one camera, one kernel, shared by sample and reference
        # fits), edited in the Global Settings group. Not fitted per frame.
        return ["pr_sigma_px", "pr_tau_left_px", "pr_tau_right_px"]

    def na_field_names(self):
        # NA collection model (na_lorentzian* fitting model and the post-hoc
        # correction), sample only: aperture-clip NA; Gaussian coupling
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
        for field in list(self.field_names()) + list(extra_fields):
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

        # NA collection weight (sample group only, alongside the NA fields).
        if "na_collection" in extra_fields:
            row = QHBoxLayout()
            row.addWidget(QLabel("NA weighting"))
            na_combo = QComboBox()
            na_combo.addItems(NA_WEIGHTINGS)
            na_combo.setToolTip(
                "Collection weight over the NA cone (na_lorentzian model and "
                "the post-hoc correction):\n"
                "uniform: hard pupil only — the NA 0.14 recipe (~ +3.5 MHz on "
                "water, parameter-free).\n"
                "uniform_gaussian: adds the Gaussian fiber-coupling apodization "
                "from na_beam_diameter_mm / na_focal_length_mm — required at "
                "NA 0.42."
            )
            inputs["na_weighting"] = na_combo
            row.addWidget(na_combo)
            vlayout.addLayout(row)

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

        # Pixel Offsets
        for key in ["pixel_offset_left", "pixel_offset_right"]:
            row = QHBoxLayout()
            row.addWidget(QLabel(key.replace("_", " ").capitalize()))
            edit = QLineEdit()
            edit.setValidator(QIntValidator(0, 9999))
            self.global_inputs[key] = edit
            row.addWidget(edit)
            layout.addLayout(row)

        # Camera pixel-response constants — global: one camera, one kernel,
        # shared by the sample and reference fits.
        layout.addWidget(QLabel("Camera pixel response (shared by both fits)"))
        for key in self.pr_field_names():
            row = QHBoxLayout()
            row.addWidget(QLabel(key.replace("_", " ").capitalize()))
            edit = QLineEdit()
            edit.setValidator(QDoubleValidator(0.0, 100.0, 5))
            edit.setToolTip(
                "Frozen camera constants for the 'pixel_response' model — "
                "Gaussian charge-diffusion blur and the one-sided readout "
                "tails. Not fitted per frame; measured 2026-07 on the fine "
                "EOM sweeps: 0.25 / 0.40 / 0.20 px. Re-measure after any "
                "camera/ROI change."
            )
            self.global_inputs[key] = edit
            row.addWidget(edit)
            layout.addLayout(row)

        return layout

    def load_values(self):
        sample = find_peaks_sample_config.get()
        reference = find_peaks_reference_config.get()
        global_cfg = sline_from_frame_config.get()
        pr = pixel_response_config.get()

        for field in self.field_names():
            self.sample_inputs[field].setText(str(getattr(sample, field)))
            self.reference_inputs[field].setText(str(getattr(reference, field)))

        for field in self.na_field_names():
            self.sample_inputs[field].setText(str(getattr(sample, field)))
        self.sample_inputs["na_weighting"].setCurrentText(sample.na_weighting)

        for field in self.pr_field_names():
            self.global_inputs[field].setText(str(getattr(pr, field)))

        for inputs, cfg in ((self.sample_inputs, sample),
                            (self.reference_inputs, reference)):
            inputs["fitting_model"].setCurrentText(cfg.fitting_model)
            inputs["background"].setCurrentText(cfg.background)
            inputs["use_window"].setChecked(bool(cfg.use_window))

        # Global settings
        self.global_inputs["pixel_offset_left"].setText(str(global_cfg.pixel_offset_left))
        self.global_inputs["pixel_offset_right"].setText(str(global_cfg.pixel_offset_right))
        self.global_inputs["selected_rows"].setText(", ".join(str(x) for x in global_cfg.selected_rows))
        self.global_inputs["row_selection"].setCurrentText(global_cfg.row_selection)
        self.global_inputs["n_rows"].setText(str(global_cfg.n_rows))

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
            }

            # Sample
            sample_kwargs = {f: self._parse(self.sample_inputs[f].text(), f)
                             for f in list(self.field_names()) + list(self.na_field_names())}
            sample_kwargs["na_weighting"] = self.sample_inputs["na_weighting"].currentText()
            sample_kwargs.update(self._model_kwargs(self.sample_inputs))

            # Reference
            reference_kwargs = {f: self._parse(self.reference_inputs[f].text(), f)
                                for f in self.field_names()}
            reference_kwargs.update(self._model_kwargs(self.reference_inputs))

            # Camera pixel-response constants (global, shared by both fits)
            pr_kwargs = {f: self._parse(self.global_inputs[f].text(), f)
                         for f in self.pr_field_names()}

            # Update all configs
            find_peaks_sample_config.update(**sample_kwargs)
            find_peaks_reference_config.update(**reference_kwargs)
            sline_from_frame_config.update(**global_kwargs)
            pixel_response_config.update(**pr_kwargs)

            if self.on_apply:
                fitting_configs = FittingConfigs(
                    sline_config=sline_from_frame_config.get(),
                    sample_config=find_peaks_sample_config.get(),
                    reference_config=find_peaks_reference_config.get(),
                    pr_config=pixel_response_config.get(),
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
            save_config_section(FIND_PEAKS_TOML_PATH, "camera", pixel_response_config)
            QMessageBox.information(self, "Saved", "Settings saved to disk.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save config:\n{e}")

    @staticmethod
    def _model_kwargs(inputs):
        """Lineshape + the two options that apply to any lineshape."""
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
            or field.startswith("pr_")
        )

    def _parse(self, value, field):
        value = value.strip()
        try:
            return float(value) if self._is_float_field(field) else int(value)
        except ValueError:
            return 0

    def _parse_selected_rows(self, text):
        return [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]


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
