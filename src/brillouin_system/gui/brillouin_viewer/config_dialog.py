from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox,
    QPushButton, QApplication, QFormLayout, QButtonGroup, QRadioButton, QCheckBox
)
from PyQt5.QtGui import QIntValidator

from brillouin_system.config.config import (
    sample_config,
    reference_config,
    calibration_config,
    andor_frame_config,
    save_find_peaks_config_section,
    save_calibration_config,
    save_andor_frame_settings,
    find_peaks_config_toml_path,
)


class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Peak Detection Settings")
        self.setMinimumSize(900, 400)

        self.sample_inputs = {}
        self.reference_inputs = {}

        # Predefine Andor Frame inputs
        self.selected_rows_input = QLineEdit()
        self.n_dark_images_input = QLineEdit()
        self.n_dark_images_input.setValidator(QIntValidator(0, 999))
        self.do_subtract_dark_image_input = QCheckBox()
        self.n_bg_images_input = QLineEdit()
        self.n_bg_images_input.setValidator(QIntValidator(0, 999))

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.create_andor_frame_group())
        main_layout.addWidget(self.create_spectrum_fitting_group())
        main_layout.addWidget(self.create_calibration_group())
        main_layout.addLayout(self.create_save_button())

        self.setLayout(main_layout)
        self.load_initial_values()

    def create_andor_frame_group(self):
        group = QGroupBox("Andor Frame Settings")
        layout = QVBoxLayout()

        # Row 1: Selected pixel rows
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Selected Pixel Rows (e.g. 4,5,6):"))
        row1.addWidget(self.selected_rows_input)

        # Row 2: All other inputs in one line
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("BG images:"))
        row2.addWidget(self.n_bg_images_input)
        row2.addWidget(QLabel("Dark images:"))
        row2.addWidget(self.n_dark_images_input)
        row2.addWidget(QLabel("Subtract dark:"))
        row2.addWidget(self.do_subtract_dark_image_input)

        layout.addLayout(row1)
        layout.addLayout(row2)
        group.setLayout(layout)
        return group

    def create_spectrum_fitting_group(self):
        group = QGroupBox("Spectrum Fitting")
        layout = QHBoxLayout()
        layout.addWidget(self.create_config_group("Sample", self.sample_inputs))
        layout.addWidget(self.create_config_group("Reference", self.reference_inputs))
        group.setLayout(layout)
        return group

    def create_config_group(self, title, inputs_dict):
        group = QGroupBox(title)
        layout = QVBoxLayout()
        for field in self.field_names():
            row = QHBoxLayout()
            label = QLabel(field.replace("_", " ").capitalize() + ":")
            input_field = QLineEdit()
            inputs_dict[field] = input_field
            row.addWidget(label)
            row.addWidget(input_field)
            layout.addLayout(row)
        group.setLayout(layout)
        return group

    def create_calibration_group(self):
        group = QGroupBox("Calibration")
        layout = QFormLayout()

        self.n_per_freq_input = QLineEdit()
        self.n_per_freq_input.setValidator(QIntValidator(1, 999))
        layout.addRow("n_per_freq:", self.n_per_freq_input)

        self.calib_freqs_input = QLineEdit()
        layout.addRow("Frequencies for Calibration (GHz):", self.calib_freqs_input)

        toggle_row = QHBoxLayout()
        self.ref_group = QButtonGroup(self)
        self.left_radio = QRadioButton("Left Peak")
        self.right_radio = QRadioButton("Right Peak")
        self.dist_radio = QRadioButton("Peak Distance")
        self.ref_group.addButton(self.left_radio)
        self.ref_group.addButton(self.right_radio)
        self.ref_group.addButton(self.dist_radio)
        toggle_row.addWidget(self.left_radio)
        toggle_row.addWidget(self.right_radio)
        toggle_row.addWidget(self.dist_radio)
        layout.addRow("Reference:", toggle_row)

        group.setLayout(layout)
        return group

    def create_save_button(self):
        self.save_button = QPushButton("Save Values")
        self.save_button.clicked.connect(self.on_save_values_clicked)

        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self.save_button)
        return row

    def field_names(self):
        return [
            "prominence_fraction",
            "min_peak_width",
            "min_peak_height",
            "rel_height",
            "wlen_pixels",
        ]

    def load_initial_values(self):
        andor = andor_frame_config.get()
        self.selected_rows_input.setText(",".join(str(x) for x in andor.selected_rows))
        self.n_dark_images_input.setText(str(andor.n_dark_images))
        self.do_subtract_dark_image_input.setChecked(andor.do_subtract_dark_image)
        self.n_bg_images_input.setText(str(andor.n_bg_images))

        for field in self.field_names():
            self.sample_inputs[field].setText(str(getattr(sample_config.get(), field)))
            self.reference_inputs[field].setText(str(getattr(reference_config.get(), field)))

        calib = calibration_config.get()
        self.n_per_freq_input.setText(str(calib.n_per_freq))
        self.calib_freqs_input.setText(", ".join(f"{f:.3f}" for f in calib.calibration_freqs))

        ref = calib.reference
        if ref == "left":
            self.left_radio.setChecked(True)
        elif ref == "right":
            self.right_radio.setChecked(True)
        else:
            self.dist_radio.setChecked(True)

    def on_save_values_clicked(self):
        try:
            # Save Andor Frame Settings
            andor_frame_config.update(
                selected_rows=[int(x.strip()) for x in self.selected_rows_input.text().split(",") if x.strip().isdigit()],
                n_dark_images=int(self.n_dark_images_input.text()),
                do_subtract_dark_image=self.do_subtract_dark_image_input.isChecked(),
                n_bg_images=int(self.n_bg_images_input.text())
            )

            # Save FindPeaks sample/reference
            for field in self.field_names():
                sample_config.set(field, self._parse_value(self.sample_inputs[field].text(), field))
                reference_config.set(field, self._parse_value(self.reference_inputs[field].text(), field))

            # Save calibration
            if self.left_radio.isChecked():
                ref = "left"
            elif self.right_radio.isChecked():
                ref = "right"
            else:
                ref = "distance"

            calibration_config.update(
                n_per_freq=int(self.n_per_freq_input.text()),
                calibration_freqs=[float(f.strip()) for f in self.calib_freqs_input.text().split(",") if f.strip()],
                reference=ref
            )

            save_andor_frame_settings(find_peaks_config_toml_path, andor_frame_config)
            save_find_peaks_config_section(find_peaks_config_toml_path, "sample", sample_config)
            save_find_peaks_config_section(find_peaks_config_toml_path, "reference", reference_config)
            save_calibration_config(find_peaks_config_toml_path, calibration_config)

            print("[ConfigDialog] Saved successfully.")
        except Exception as e:
            print(f"[ConfigDialog] Error while saving: {e}")

    def _parse_value(self, value: str, field: str):
        value = value.strip()
        if value.lower() == "none":
            return None
        try:
            if "fraction" in field or "rel" in field:
                return float(value)
            return int(value)
        except ValueError:
            return None


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    dialog = ConfigDialog()
    dialog.show()
    sys.exit(app.exec_())
