from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QGroupBox,
    QPushButton, QApplication, QFormLayout, QButtonGroup, QRadioButton,
    QCheckBox, QComboBox
)
from PyQt5.QtGui import QIntValidator

from brillouin_system.config.config import (
    find_peaks_sample_config,
    find_peaks_reference_config,
    calibration_config,
    andor_frame_config,
    save_find_peaks_config_section,
    save_calibration_config,
    save_andor_frame_settings,
    find_peaks_config_toml_path,
    fitting_models_sample,
    fitting_models_reference
)



class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Peak Detection Settings")
        self.setMinimumSize(900, 600)

        self.sample_inputs = {}
        self.reference_inputs = {}

        # Andor Frame inputs
        self.selected_rows_input = QLineEdit()
        self.n_dark_images_input = QLineEdit()
        self.n_dark_images_input.setValidator(QIntValidator(0, 999))
        self.dark_image_input = QCheckBox()
        self.n_bg_images_input = QLineEdit()
        self.n_bg_images_input.setValidator(QIntValidator(0, 999))
        self.x_start_input = QLineEdit()
        self.x_start_input.setValidator(QIntValidator(0, 9999))
        self.x_end_input = QLineEdit()
        self.x_end_input.setValidator(QIntValidator(0, 9999))
        self.y_start_input = QLineEdit()
        self.y_start_input.setValidator(QIntValidator(0, 9999))
        self.y_end_input = QLineEdit()
        self.y_end_input.setValidator(QIntValidator(0, 9999))
        self.vbin_input = QLineEdit()
        self.vbin_input.setValidator(QIntValidator(0, 999))
        self.hbin_input = QLineEdit()
        self.hbin_input.setValidator(QIntValidator(0, 999))
        self.amp_mode_index_input = QLineEdit()
        self.amp_mode_index_input.setValidator(QIntValidator(0, 999))

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.create_andor_frame_group())
        main_layout.addWidget(self.create_spectrum_fitting_group())
        main_layout.addWidget(self.create_calibration_group())
        main_layout.addLayout(self.create_save_button())

        self.setLayout(main_layout)
        self.load_initial_values()

    def create_andor_frame_group(self):
        group = QGroupBox("Andor Frame Analysis Settings")
        layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Selected Pixel Rows (e.g. 4,5,6):"))
        row1.addWidget(self.selected_rows_input)
        row1.addWidget(QLabel("Amp Mode Index:"))
        row1.addWidget(self.amp_mode_index_input)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("X Start:"))
        row2.addWidget(self.x_start_input)
        row2.addWidget(QLabel("X End:"))
        row2.addWidget(self.x_end_input)
        row2.addWidget(QLabel("Y Start:"))
        row2.addWidget(self.y_start_input)
        row2.addWidget(QLabel("Y End:"))
        row2.addWidget(self.y_end_input)
        row2.addWidget(QLabel("VBin:"))
        row2.addWidget(self.vbin_input)
        row2.addWidget(QLabel("HBin:"))
        row2.addWidget(self.hbin_input)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("BG Images:"))
        row3.addWidget(self.n_bg_images_input)
        row3.addWidget(QLabel("Dark Images:"))
        row3.addWidget(self.n_dark_images_input)
        row3.addWidget(QLabel("Take Dark Images:"))
        row3.addWidget(self.dark_image_input)

        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addLayout(row3)
        group.setLayout(layout)
        return group

    def create_spectrum_fitting_group(self):
        group = QGroupBox("Find Peaks")
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

        # Add Peak Fitting combo at end
        row = QHBoxLayout()
        label = QLabel("Peak Fitting:")
        combo = QComboBox()
        combo.addItems(fitting_models_sample if title.lower() == "sample" else fitting_models_reference)
        row.addWidget(label)
        row.addWidget(combo)
        layout.addLayout(row)

        if title.lower() == "sample":
            self.sample_model_combo = combo
        else:
            self.reference_model_combo = combo

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
        self.dark_image_input.setChecked(andor.take_dark_image)
        self.n_bg_images_input.setText(str(andor.n_bg_images))
        self.x_start_input.setText(str(andor.x_start))
        self.x_end_input.setText(str(andor.x_end))
        self.y_start_input.setText(str(andor.y_start))
        self.y_end_input.setText(str(andor.y_end))
        self.vbin_input.setText(str(andor.vbin))
        self.hbin_input.setText(str(andor.hbin))
        self.amp_mode_index_input.setText(str(andor.amp_mode_index))
        self.sample_model_combo.setCurrentText(find_peaks_sample_config.get_field("fitting_model"))
        self.reference_model_combo.setCurrentText(find_peaks_reference_config.get_field("fitting_model"))

        for field in self.field_names():
            self.sample_inputs[field].setText(str(getattr(find_peaks_sample_config.get(), field)))
            self.reference_inputs[field].setText(str(getattr(find_peaks_reference_config.get(), field)))

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
            andor_frame_config.update(
                selected_rows=[int(x.strip()) for x in self.selected_rows_input.text().split(",") if x.strip().isdigit()],
                n_dark_images=int(self.n_dark_images_input.text()),
                take_dark_image=self.dark_image_input.isChecked(),
                n_bg_images=int(self.n_bg_images_input.text()),
                x_start=int(self.x_start_input.text()),
                x_end=int(self.x_end_input.text()),
                y_start=int(self.y_start_input.text()),
                y_end=int(self.y_end_input.text()),
                vbin=int(self.vbin_input.text()),
                hbin=int(self.hbin_input.text()),
                amp_mode_index=int(self.amp_mode_index_input.text())
            )

            # Sample
            sample_kwargs = {field: self._parse_value(self.sample_inputs[field].text(), field)
                             for field in self.field_names()}
            sample_kwargs["fitting_model"] = self.sample_model_combo.currentText()
            find_peaks_sample_config.update(**sample_kwargs)

            # Reference
            reference_kwargs = {field: self._parse_value(self.reference_inputs[field].text(), field)
                                for field in self.field_names()}
            reference_kwargs["fitting_model"] = self.reference_model_combo.currentText()
            find_peaks_reference_config.update(**reference_kwargs)

            # Calibration
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
            save_find_peaks_config_section(find_peaks_config_toml_path, "sample", find_peaks_sample_config)
            save_find_peaks_config_section(find_peaks_config_toml_path, "reference", find_peaks_reference_config)
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
