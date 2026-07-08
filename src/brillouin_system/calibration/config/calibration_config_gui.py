from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QFormLayout, QLineEdit, QRadioButton,
    QButtonGroup, QPushButton, QLabel, QMessageBox, QHBoxLayout, QCheckBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from brillouin_system.calibration.config.calibration_config import (
    calibration_config, save_calibration_config, CALIBRATION_TOML_PATH
)


class CalibrationConfigDialog(QDialog):
    def __init__(self, on_apply=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Settings")

        self.on_apply = on_apply

        layout = QVBoxLayout()
        form = QFormLayout()

        # Inputs
        self.n_per_freq_input = QLineEdit()
        self.n_per_freq_input.setValidator(QIntValidator(1, 999))

        self.degree_input = QLineEdit()
        self.degree_input.setValidator(QIntValidator(1, 99))

        self.start_input = QLineEdit()
        self.start_input.setValidator(QDoubleValidator(0.0, 1e6, 6))

        self.stop_input = QLineEdit()
        self.stop_input.setValidator(QDoubleValidator(0.0, 1e6, 6))

        self.step_input = QLineEdit()
        self.step_input.setValidator(QDoubleValidator(0.0, 1e6, 6))

        # Reference radio buttons
        self.ref_group = QButtonGroup(self)
        self.left_radio = QRadioButton("Left Peak")
        self.right_radio = QRadioButton("Right Peak")
        self.dist_radio = QRadioButton("Peak Distance")

        self.ref_group.addButton(self.left_radio)
        self.ref_group.addButton(self.right_radio)
        self.ref_group.addButton(self.dist_radio)

        ref_layout = QVBoxLayout()
        ref_layout.addWidget(self.left_radio)
        ref_layout.addWidget(self.right_radio)
        ref_layout.addWidget(self.dist_radio)

        # Mode radio buttons
        self.mode_group = QButtonGroup(self)
        self.poly_radio = QRadioButton("Polynomial")
        self.interp_radio = QRadioButton("Interpolation")

        self.mode_group.addButton(self.poly_radio)
        self.mode_group.addButton(self.interp_radio)

        mode_layout = QVBoxLayout()
        mode_layout.addWidget(self.poly_radio)
        mode_layout.addWidget(self.interp_radio)

        # Save calibration frames checkbox
        self.save_frames_checkbox = QCheckBox("Save calibration frames with each scan")

        # Form layout
        form.addRow("n_per_freq:", self.n_per_freq_input)
        form.addRow("Polynomial Degree:", self.degree_input)
        form.addRow("Start Frequency (GHz):", self.start_input)
        form.addRow("Stop Frequency (GHz):", self.stop_input)
        form.addRow("Step (GHz):", self.step_input)
        form.addRow(QLabel("Reference Method:"), ref_layout)
        form.addRow(QLabel("Mode:"), mode_layout)
        form.addRow(QLabel("Storage:"), self.save_frames_checkbox)

        layout.addLayout(form)

        # Buttons (Apply / Save / Close)
        layout.addLayout(self.create_buttons())

        self.setLayout(layout)
        self.load_values()

    def load_values(self):
        cfg = calibration_config.get()

        self.n_per_freq_input.setText(str(cfg.n_per_freq))
        self.degree_input.setText(str(cfg.degree))
        self.start_input.setText(str(cfg.start))
        self.stop_input.setText(str(cfg.stop))
        self.step_input.setText(str(cfg.step))

        if cfg.reference == "left":
            self.left_radio.setChecked(True)
        elif cfg.reference == "right":
            self.right_radio.setChecked(True)
        else:
            self.dist_radio.setChecked(True)

        if cfg.mode == "poly":
            self.poly_radio.setChecked(True)
        else:
            self.interp_radio.setChecked(True)

        self.save_frames_checkbox.setChecked(cfg.save_calibration_frames)

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
            reference = (
                "left" if self.left_radio.isChecked() else
                "right" if self.right_radio.isChecked() else
                "distance"
            )

            mode = "poly" if self.poly_radio.isChecked() else "interp"

            calibration_config.update(
                n_per_freq=int(self.n_per_freq_input.text()),
                degree=int(self.degree_input.text()),
                start=float(self.start_input.text()),
                stop=float(self.stop_input.text()),
                step=float(self.step_input.text()),
                reference=reference,
                mode=mode,
                save_calibration_frames=self.save_frames_checkbox.isChecked(),
            )

            if self.on_apply:
                self.on_apply(calibration_config.get())

            QMessageBox.information(self, "Applied", "Settings applied (not saved to disk).")

        except Exception as e:
            QMessageBox.critical(self, "Apply Error", f"Failed to apply config:\n{e}")

    def save_config(self):
        try:
            self.apply_config()
            save_calibration_config(CALIBRATION_TOML_PATH, calibration_config)
            QMessageBox.information(self, "Saved", "Calibration settings saved to disk.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save config:\n{e}")


if __name__ == "__main__":
    import sys

    def on_apply(cfg):
        print("[Calibration Applied]", cfg)

    app = QApplication(sys.argv)
    dlg = CalibrationConfigDialog(on_apply=on_apply)
    dlg.show()
    sys.exit(app.exec_())