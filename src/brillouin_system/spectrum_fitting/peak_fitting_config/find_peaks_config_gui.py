from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QGroupBox, QApplication, QMessageBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from brillouin_system.spectrum_fitting.peak_fitting_config.find_peaks_config import (
    find_peaks_sample_config, find_peaks_reference_config, sline_from_frame_config,
    save_config_section, FIND_PEAKS_TOML_PATH,
    FITTING_MODELS_SAMPLE, FITTING_MODELS_REFERENCE, FittingConfigs
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
            "rel_height", "wlen_pixels"
        ]

    def create_dual_form(self):
        layout = QHBoxLayout()
        layout.addWidget(self.create_config_group("Sample", self.sample_inputs, FITTING_MODELS_SAMPLE))
        layout.addWidget(self.create_config_group("Reference", self.reference_inputs, FITTING_MODELS_REFERENCE))
        return layout

    def create_config_group(self, label, inputs, models):
        group = QGroupBox(label)
        vlayout = QVBoxLayout()
        for field in self.field_names():
            row = QHBoxLayout()
            row.addWidget(QLabel(field.replace("_", " ").capitalize()))
            edit = QLineEdit()
            if "fraction" in field or "rel" in field:
                edit.setValidator(QDoubleValidator(0.0, 1.0, 5))
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

        group.setLayout(vlayout)
        return group

    def create_global_inputs(self):
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Global Settings"))

        # Selected Rows (comma-separated)
        row = QHBoxLayout()
        row.addWidget(QLabel("Selected Rows"))
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

        return layout

    def load_values(self):
        sample = find_peaks_sample_config.get()
        reference = find_peaks_reference_config.get()
        global_cfg = sline_from_frame_config.get()

        for field in self.field_names():
            self.sample_inputs[field].setText(str(getattr(sample, field)))
            self.reference_inputs[field].setText(str(getattr(reference, field)))

        self.sample_inputs["fitting_model"].setCurrentText(sample.fitting_model)
        self.reference_inputs["fitting_model"].setCurrentText(reference.fitting_model)

        # Global settings
        self.global_inputs["pixel_offset_left"].setText(str(global_cfg.pixel_offset_left))
        self.global_inputs["pixel_offset_right"].setText(str(global_cfg.pixel_offset_right))
        self.global_inputs["selected_rows"].setText(", ".join(str(x) for x in global_cfg.selected_rows))

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
            }

            # Sample
            sample_kwargs = {f: self._parse(self.sample_inputs[f].text(), f) for f in self.field_names()}
            sample_kwargs["fitting_model"] = self.sample_inputs["fitting_model"].currentText()

            # Reference
            reference_kwargs = {f: self._parse(self.reference_inputs[f].text(), f) for f in self.field_names()}
            reference_kwargs["fitting_model"] = self.reference_inputs["fitting_model"].currentText()

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

                self.on_apply(
                    fitting_configs
                )

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

    def _parse(self, value, field):
        value = value.strip()
        try:
            return float(value) if "fraction" in field or "rel" in field else int(value)
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
