# calibration_config_gui.py
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QFormLayout, QLineEdit, QRadioButton,
    QButtonGroup, QPushButton, QLabel, QMessageBox, QCheckBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from brillouin_system.calibration.config.calibration_config import (
    calibration_config, save_calibration_config, CALIBRATION_TOML_PATH
)


class CalibrationConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Settings")
        self.setMinimumSize(300, 300)

        layout = QVBoxLayout()
        form = QFormLayout()

        self.n_per_freq_input = QLineEdit(); self.n_per_freq_input.setValidator(QIntValidator(1, 999))
        self.degree_input = QLineEdit(); self.degree_input.setValidator(QIntValidator(1, 99))

        self.start_input = QLineEdit(); self.start_input.setValidator(QDoubleValidator(0.0, 1e6, 6))
        self.stop_input = QLineEdit(); self.stop_input.setValidator(QDoubleValidator(0.0, 1e6, 6))
        self.step_input = QLineEdit(); self.step_input.setValidator(QDoubleValidator(0.0, 1e6, 6))

        self.ref_group = QButtonGroup(self)
        self.left_radio = QRadioButton("Left Peak")
        self.right_radio = QRadioButton("Right Peak")
        self.dist_radio = QRadioButton("Peak Distance")

        self.ref_group.addButton(self.left_radio)
        self.ref_group.addButton(self.right_radio)
        self.ref_group.addButton(self.dist_radio)

        toggle_layout = QVBoxLayout()
        toggle_layout.addWidget(self.left_radio)
        toggle_layout.addWidget(self.right_radio)
        toggle_layout.addWidget(self.dist_radio)

        self.safe_checkbox = QCheckBox("Safe Calibration Data for each Scan")

        form.addRow("n_per_freq:", self.n_per_freq_input)
        form.addRow("Polynomial Degree:", self.degree_input)
        form.addRow("Start Frequency (GHz):", self.start_input)
        form.addRow("Stop Frequency (GHz):", self.stop_input)
        form.addRow("Step (GHz):", self.step_input)
        form.addRow(QLabel("Reference Method:"), toggle_layout)
        form.addRow(self.safe_checkbox)

        layout.addLayout(form)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_config)
        layout.addWidget(save_button)

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

        self.safe_checkbox.setChecked(cfg.safe_each_scan)

    def save_config(self):
        try:
            reference = (
                "left" if self.left_radio.isChecked() else
                "right" if self.right_radio.isChecked() else
                "distance"
            )

            calibration_config.update(
                n_per_freq=int(self.n_per_freq_input.text()),
                degree=int(self.degree_input.text()),
                start=float(self.start_input.text()),
                stop=float(self.stop_input.text()),
                step=float(self.step_input.text()),
                reference=reference,
                safe_each_scan=self.safe_checkbox.isChecked()
            )

            save_calibration_config(CALIBRATION_TOML_PATH, calibration_config)
            QMessageBox.information(self, "Saved", "Calibration settings saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    dlg = CalibrationConfigDialog()
    dlg.exec_()
